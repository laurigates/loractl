//! Explicit, lossless resume for the diffusion path (#180).
//!
//! Before this module a resumed run was *implicit and lossy*: the only trigger
//! was "`output.dir/output.name.safetensors` happens to exist", the step
//! counter restarted at 1, and AdamW re-warmed its moments from zero. Three
//! things follow from that, and this module fixes each:
//!
//! 1. **The trigger is now named.** [`crate::config::ResumeConfig`] carries
//!    `from` (an explicit artifact), `auto` (the historical
//!    resume-if-it-exists, kept as the documented default), and
//!    `allow_unfinished`.
//! 2. **Provenance is read, not assumed.** The `__metadata__` header this repo
//!    already writes ([`crate::metadata`]) says how many steps a file
//!    represents and whether the run that wrote it finished; [`plan_resume`]
//!    keys on that rather than on the file's mere existence.
//! 3. **The optimizer state round-trips** through a sidecar
//!    ([`optim_sidecar_path`]) that interop consumers ignore.
//!
//! ## Why a sidecar, and why not `export.rs`
//!
//! [`crate::export`] is the *interop* boundary: everything it writes is shaped
//! by what ComfyUI/kohya-ss read, and its key set is pinned against ComfyUI's
//! own `krea2_to_diffusers` map (`tests/krea2_lora_keys.rs`). AdamW moments
//! have no consumer out there, so putting them in that file would grow the
//! interop artifact by ~2× for tensors no reader wants — and any reader that
//! *did* pattern-match the new keys would be matching something we invented.
//! They live in a separate file, in burn's own (un-transposed) layout, under
//! keys that deliberately cannot collide with `.lora_down.weight` /
//! `.lora_up.weight` / `.alpha`.
//!
//! ## What resume does *not* restore
//!
//! The RNG stream. `AB::seed(&device, config.seed)` reseeds at run start
//! (`diffusion_trainer.rs`), and a resumed run then draws its timesteps and
//! noise from a stream that has **not** consumed the draws of the steps it is
//! skipping; burn 0.21 exposes no RNG save/restore. A resumed run is therefore
//! a *continuation*, not a bit-identical replay of an uninterrupted one, and
//! [`resume_message`] says so in the operator-visible event rather than
//! leaving it to be discovered.
//!
//! Per the crate invariant this module renders nothing: it returns `Result`s
//! and builds `String`s for the caller to put on the [`TrainEvent`
//! sink](crate::event::TrainEvent).

use crate::adapters::LoraAdapters;
use crate::config::{OutputConfig, ResumeConfig};
use crate::export::{OwnedF32Tensor, to_owned_f32};
use crate::metadata::{LoraMetadata, read_metadata};
use anyhow::{Context, Result, bail, ensure};
use burn::module::ParamId;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::record::AdaptorRecord;
use burn::optim::{AdamW, AdamWState, AdaptiveMomentumState, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorData};
use safetensors::tensor::Dtype;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The optimizer the diffusion trainer runs, spelled once so the sidecar's
/// signatures do not have to restate the three-parameter adaptor.
pub type LoraOptimizer<AB> = OptimizerAdaptor<AdamW, LoraAdapters<AB>, AB>;

/// `__metadata__` key naming the sidecar for what it is. Written so
/// `loractl inspect` (and a human with `strings`) can tell a 40 MB file of
/// optimizer moments from a LoRA at a glance — the sidecar sits next to the
/// adapter and will be copied around with it.
const ARTIFACT_KIND_KEY: &str = "loractl_artifact_kind";
/// Value of [`ARTIFACT_KIND_KEY`]; also the load-time sanity check, so
/// pointing the loader at an adapter export fails by name.
const ARTIFACT_KIND: &str = "optimizer-state";
/// AdamW's bias-correction step count (`AdaptiveMomentumState::time`). Not
/// derivable from the moments, and dropping it silently changes the very next
/// update's magnitude — which is exactly the "wrote a file, restored nothing"
/// failure this module exists to avoid.
const TIME_KEY: &str = "loractl_adamw_time";
/// The dynamic f16 loss scale at write time.
const LOSS_SCALE_KEY: &str = "loractl_loss_scale";
/// Consecutive non-overflowing steps at write time (the scale-growth counter).
const CLEAN_STREAK_KEY: &str = "loractl_clean_streak";
/// Steps completed at write time. Cross-checked against the adapter's
/// `ss_steps` by [`stale_sidecar`], so a sidecar left over from a *different*
/// run is caught — the file is written and read, and the check that makes it
/// mean something lives there, not here.
const STEPS_DONE_KEY: &str = "loractl_steps_done";

/// The two kohya factor names a site path is suffixed with, plus which of the
/// pair is the `A` (down) factor. The torch AdamW state names (`exp_avg`,
/// `exp_avg_sq`) are appended *after* these at the `format!` sites in
/// [`save_optimizer_state`] and [`load_optimizer_state`] — that second suffix
/// is what keeps a moment key from ending in `.weight`/`.alpha`, so no LoRA
/// loader's key map can match this file if it is ever dropped into a
/// `models/loras/` directory (the #137 failure shape: an unmatched key loads
/// without error and does nothing). [`moment_key`] is the one place that
/// composition happens.
const FACTOR_KEYS: [(&str, bool); 2] = [("lora_down", true), ("lora_up", false)];

/// The sidecar key for one moment tensor: site path, kohya factor name, torch
/// state name. One function rather than four `format!`s so the unit test that
/// pins the shape of these keys sees the *production* composition rather than
/// literals it chose itself.
fn moment_key(target: &str, suffix: &str, kind: &str) -> String {
    format!("{target}.{suffix}.{kind}")
}

/// Where the optimizer sidecar for `adapter` lives: beside it, same stem,
/// `.optim.safetensors`.
///
/// Beside the *source* rather than the destination, deliberately: resuming an
/// artifact from another directory into a clean `output.dir` is a real
/// recovery workflow, and the moments belong to the file being resumed.
pub fn optim_sidecar_path(adapter: &Path) -> PathBuf {
    let mut name = adapter
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "adapter".to_string());
    name.push_str(".optim.safetensors");
    adapter.with_file_name(name)
}

/// What the resume source's `__metadata__` says about the run that wrote it.
///
/// Three states, not two. "No header" is *not* "unfinished": a
/// `metadata.embed: false` run writes no `__metadata__` at all
/// (`export.rs` drops an empty map), and so does any third-party tool, so
/// folding it into the unfinished branch would refuse every `--no-metadata`
/// user's resume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumeProvenance {
    /// `ss_training_finished_at` is present — the run that wrote this file ran
    /// to completion, having done `steps_done` steps.
    Finished {
        /// The file's `ss_steps`.
        steps_done: u64,
    },
    /// `ss_steps` is present but `ss_training_finished_at` is not. Every
    /// mid-run `checkpoint-N.safetensors` is this by construction
    /// (`metadata_for(step, false)`), which is why it is not fatal on its own.
    Unfinished {
        /// The file's `ss_steps`.
        steps_done: u64,
        /// The file's `ss_max_train_steps` — the total the interrupted run was
        /// aiming at, when it recorded one.
        planned: Option<u64>,
    },
    /// No `ss_steps` in the header (or no header at all). The step counter
    /// cannot be honoured, so the resumed run restarts its numbering at 1.
    Unrecorded,
}

impl ResumeProvenance {
    /// Steps this file represents, or 0 when the header did not record it.
    pub fn steps_done(&self) -> u64 {
        match self {
            Self::Finished { steps_done } | Self::Unfinished { steps_done, .. } => *steps_done,
            Self::Unrecorded => 0,
        }
    }
}

/// A resolved resume: which file, why, what it claims, and whether an
/// optimizer sidecar sits beside it.
#[derive(Debug, Clone)]
pub struct ResumePlan {
    /// The adapter artifact to load.
    pub source: PathBuf,
    /// `true` when `resume.from` named it; `false` for the `auto` path.
    pub explicit: bool,
    /// What the source's `__metadata__` says (see [`ResumeProvenance`]).
    pub provenance: ResumeProvenance,
    /// The sidecar beside `source`, when one exists **and belongs to it**
    /// (see [`stale_sidecar`]).
    pub optim_state: Option<PathBuf>,
    /// Set instead of [`Self::optim_state`] when a sidecar sits beside the
    /// source but records a different run's step count. Carried rather than
    /// dropped so [`resume_message`] can say *why* the moments are missing.
    pub stale_optim: Option<StaleSidecar>,
}

/// A sidecar that exists beside the resume source but was left there by a
/// different run — see [`stale_sidecar`] for how that happens and why it is
/// skipped rather than restored or refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaleSidecar {
    /// The file that was skipped.
    pub path: PathBuf,
    /// Its own [`STEPS_DONE_KEY`].
    pub steps_done: u64,
    /// The `ss_steps` the resume source records.
    pub source_steps_done: u64,
}

impl ResumePlan {
    /// The step number the resumed run's loop starts at.
    pub fn start_step(&self) -> u64 {
        self.provenance.steps_done() + 1
    }
}

/// Resolve `resume` + `output` into a plan, or `None` for a fresh run.
///
/// Policy, and why it differs between the two paths:
///
/// - **`from` set** — the file must exist (a typo must not silently retrain
///   from scratch), and its provenance is reported but never refused. Naming a
///   `checkpoint-N.safetensors` *is* the statement of intent, and every one of
///   those is unfinished by construction.
/// - **`auto`** — an unfinished artifact at the final-export path is anomalous
///   (that path is only ever written with a finish timestamp), so it is a hard
///   error naming `allow_unfinished`. See the field docs for why an error and
///   not a warning.
pub fn plan_resume(resume: &ResumeConfig, output: &OutputConfig) -> Result<Option<ResumePlan>> {
    let (source, explicit) = match &resume.from {
        Some(from) => {
            if !from.exists() {
                bail!(
                    "resume.from names {} but no such file exists — refusing to \
                     silently start a fresh run under a resume config",
                    from.display()
                );
            }
            (from.clone(), true)
        }
        None => {
            if !resume.auto {
                return Ok(None);
            }
            let auto = output.dir.join(format!("{}.safetensors", output.name));
            if !auto.exists() {
                return Ok(None);
            }
            (auto, false)
        }
    };

    let meta = read_metadata(&source).with_context(|| {
        format!(
            "reading the resume source's metadata from {}",
            source.display()
        )
    })?;
    let provenance = provenance_of(&meta);

    if !explicit
        && !resume.allow_unfinished
        && let ResumeProvenance::Unfinished {
            steps_done,
            planned,
        } = &provenance
    {
        let planned = planned
            .map(|p| format!("{p}"))
            .unwrap_or_else(|| "unrecorded".to_string());
        bail!(
            "{} records {steps_done} of {planned} steps and carries no \
             training-finished timestamp — the final export is only ever written \
             on a completed run, so this file is a checkpoint copied into place \
             or an interrupted write. Set `resume.allow_unfinished: true` (or \
             pass --resume-unfinished) to use it anyway, name it explicitly with \
             `resume.from` / --resume, or pass --no-resume to start fresh.",
            source.display()
        );
    }

    let sidecar = optim_sidecar_path(&source);
    let (optim_state, stale_optim) = match sidecar.exists() {
        false => (None, None),
        true => match stale_sidecar(&sidecar, &provenance) {
            Some(stale) => (None, Some(stale)),
            None => (Some(sidecar), None),
        },
    };

    Ok(Some(ResumePlan {
        source,
        explicit,
        provenance,
        optim_state,
        stale_optim,
    }))
}

/// Does the sidecar beside a resume source actually belong to it?
///
/// [`save_optimizer_state`] records the writing run's step count in the
/// sidecar header ([`STEPS_DONE_KEY`]); the adapter records its own in
/// `ss_steps`. The documented recovery workflow — copy
/// `checkpoint-N.safetensors` over the final export and resume — moves the
/// *weights* and leaves the finished run's sidecar in place, so the pair
/// disagrees while nothing about the tensors can tell: the shapes match, and
/// the shape guard in [`load_optimizer_state`] passes. Restoring it anyway
/// pairs step-N weights with step-M moments, bias correction and loss scale,
/// and the run looks fine — a file written, read, and quietly meaning the
/// wrong thing (`.claude/rules/burn-optimizer-and-dropout.md` §3).
///
/// Skipped rather than fatal, deliberately: the workflow that produces the
/// mismatch is a *recovery*, and erroring would stop it dead over a file the
/// operator has no reason to suspect. Re-warming AdamW from zero is a
/// documented, survivable state; [`resume_message`] names it and names the
/// file.
///
/// Only decidable when both sides recorded a number. `Unrecorded` provenance
/// (`metadata.embed: false`, or a third-party export) has no `ss_steps` to
/// compare against, and a sidecar with no [`STEPS_DONE_KEY`] was not written
/// by this code path — neither is evidence of a mismatch, so neither skips.
fn stale_sidecar(path: &Path, provenance: &ResumeProvenance) -> Option<StaleSidecar> {
    let source_steps_done = match provenance {
        ResumeProvenance::Finished { steps_done }
        | ResumeProvenance::Unfinished { steps_done, .. } => *steps_done,
        ResumeProvenance::Unrecorded => return None,
    };
    let meta = read_metadata(path).ok()?;
    let steps_done = meta.get(STEPS_DONE_KEY)?.parse::<u64>().ok()?;
    (steps_done != source_steps_done).then(|| StaleSidecar {
        path: path.to_path_buf(),
        steps_done,
        source_steps_done,
    })
}

/// Classify a header. Split out so it is unit-testable without a file.
fn provenance_of(meta: &LoraMetadata) -> ResumeProvenance {
    let Some(steps_done) = meta.get("ss_steps").and_then(|s| s.parse::<u64>().ok()) else {
        return ResumeProvenance::Unrecorded;
    };
    if meta.get("ss_training_finished_at").is_some() {
        ResumeProvenance::Finished { steps_done }
    } else {
        ResumeProvenance::Unfinished {
            steps_done,
            planned: meta
                .get("ss_max_train_steps")
                .and_then(|s| s.parse::<u64>().ok()),
        }
    }
}

/// The scalars that ride alongside the moments — everything a step's behaviour
/// depends on that is *not* a parameter tensor.
#[derive(Debug, Clone, Copy)]
pub struct OptimProgress {
    /// Steps completed at write time.
    pub steps_done: u64,
    /// The dynamic f16 loss scale.
    pub loss_scale: f32,
    /// Consecutive clean (non-overflowing) steps.
    pub clean_streak: u32,
}

/// What [`load_optimizer_state`] actually put back.
#[derive(Debug, Clone, Copy)]
pub struct OptimRestored {
    /// Number of parameter tensors whose moments were restored (2 per delta).
    pub params: usize,
    /// AdamW's bias-correction step count.
    pub time: usize,
    /// The restored loss scale.
    pub loss_scale: f32,
    /// The restored clean streak.
    pub clean_streak: u32,
}

/// Write AdamW's per-parameter moments plus `progress` to `path`.
///
/// Keyed by **site path**, never by `ParamId`. `ParamId::new()` draws a random
/// 8-byte id (`burn-std`'s `IdGenerator`), so the ids a fresh `build_adapters`
/// mints in the next process share nothing with this one's — a `ParamId`-keyed
/// dump (which is what a naive `Recorder` gives you) would restore *nothing*
/// on the next run, silently. `set.targets[i]` is aligned with `set.deltas[i]`
/// by construction, and is the same key the export already uses.
///
/// Tensors are written in burn's own orientation (`A: [d_in, rank]`,
/// `B: [rank, d_out]`) with no transpose: this file is ours alone, so the
/// export's kohya-facing transpose would be pure ceremony here.
pub fn save_optimizer_state<AB: AutodiffBackend>(
    optim: &LoraOptimizer<AB>,
    set: &LoraAdapters<AB>,
    progress: OptimProgress,
    path: &Path,
) -> Result<()> {
    let records = optim.to_record();
    let mut tensors: Vec<(String, OwnedF32Tensor)> = Vec::with_capacity(set.deltas.len() * 4);
    let mut time: Option<usize> = None;

    for (delta, target) in set.deltas.iter().zip(&set.targets) {
        for (suffix, is_down) in FACTOR_KEYS {
            let id = if is_down {
                delta.lora_a.weight.id
            } else {
                delta.lora_b.weight.id
            };
            // Absent before the first update lands (and for any parameter the
            // gradient map skipped). Nothing to save then — an empty sidecar is
            // written rather than a wrong one.
            let Some(record) = records.get(&id) else {
                continue;
            };
            let state: AdamWState<AB::InnerBackend, 2> = record.clone().into_state::<2>();
            let momentum = state.momentum;
            // amsgrad is off (`AdamWConfig::new()` defaults it false) so this
            // is always `None`. Checked rather than assumed: turning it on
            // later must fail here loudly instead of silently halving the
            // state that round-trips.
            ensure!(
                momentum.max_moment_2.is_none(),
                "optimizer state at {target}.{suffix} carries an AMSGrad \
                 max_moment_2, which this sidecar does not serialize — \
                 enabling amsgrad needs the format extended first"
            );
            // One scalar `time` for the whole file: the trainer steps every
            // parameter together or skips them all (the dynamic-scale guard is
            // a single reduced scalar over every adapter gradient), so a
            // divergence here means that invariant broke and a single scalar
            // would quietly record a wrong bias correction for most tensors.
            match time {
                None => time = Some(momentum.time),
                Some(t) => ensure!(
                    t == momentum.time,
                    "optimizer state is inconsistent: {target}.{suffix} is at \
                     step {} while another parameter is at step {t} — AdamW's \
                     bias correction cannot be recorded as one scalar",
                    momentum.time
                ),
            }
            tensors.push((
                moment_key(target, suffix, "exp_avg"),
                to_owned_f32(momentum.moment_1),
            ));
            tensors.push((
                moment_key(target, suffix, "exp_avg_sq"),
                to_owned_f32(momentum.moment_2),
            ));
        }
    }

    // Nothing to record — the optimizer never stepped (a run that resumed an
    // already-complete config, or one whose every update was skipped by the
    // dynamic-scale guard). Writing an EMPTY sidecar here would be actively
    // destructive: it would clobber the moments of the file being resumed with
    // a file that `load_optimizer_state` then refuses as missing tensors.
    // Leaving the existing sidecar alone is also the correct content, since no
    // update was applied to the weights it belongs to.
    if tensors.is_empty() {
        return Ok(());
    }

    let mut header = HashMap::new();
    header.insert(ARTIFACT_KIND_KEY.to_string(), ARTIFACT_KIND.to_string());
    header.insert(TIME_KEY.to_string(), time.unwrap_or(0).to_string());
    header.insert(LOSS_SCALE_KEY.to_string(), progress.loss_scale.to_string());
    header.insert(
        CLEAN_STREAK_KEY.to_string(),
        progress.clean_streak.to_string(),
    );
    header.insert(STEPS_DONE_KEY.to_string(), progress.steps_done.to_string());

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating optimizer-state dir {}", parent.display()))?;
    }
    let views: Vec<(&str, &OwnedF32Tensor)> =
        tensors.iter().map(|(k, t)| (k.as_str(), t)).collect();
    // No sd-webui hashes here (unlike `export_adapters`): nothing indexes this
    // file, and computing them costs a second full serialization of the
    // largest artifact the run writes.
    safetensors::serialize_to_file(views, Some(header), path)
        .with_context(|| format!("writing optimizer state to {}", path.display()))?;
    Ok(())
}

/// Read a sidecar back into `optim`, rebuilding the `ParamId` map against the
/// **current** `set`.
///
/// Takes the optimizer by value because burn's `Optimizer::load_record`
/// consumes `self`. Call it *after* `import_adapters` — that replaces the
/// params (and so their ids) with fresh ones, and the ids the optimizer will
/// route by are the post-import ones.
///
/// Every target in `set` must be present with matching shapes; a missing site
/// or a changed rank is an error naming the site, not a partial restore.
pub fn load_optimizer_state<AB: AutodiffBackend>(
    optim: LoraOptimizer<AB>,
    set: &LoraAdapters<AB>,
    path: &Path,
    device: &AB::Device,
) -> Result<(LoraOptimizer<AB>, OptimRestored)> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading optimizer state {}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parsing optimizer state {}", path.display()))?;

    let meta = read_metadata(path)
        .with_context(|| format!("reading the optimizer state header of {}", path.display()))?;
    // Guards against being pointed at an adapter export, which would otherwise
    // fail as "missing tensor <site>.lora_down.exp_avg" — true but unhelpful.
    match meta.get(ARTIFACT_KIND_KEY) {
        Some(ARTIFACT_KIND) => {}
        other => bail!(
            "{} is not a loractl optimizer-state sidecar ({ARTIFACT_KIND_KEY} = {:?})",
            path.display(),
            other
        ),
    }
    let time: usize = meta
        .get(TIME_KEY)
        .and_then(|s| s.parse().ok())
        .with_context(|| format!("{} has no readable {TIME_KEY}", path.display()))?;
    let loss_scale: f32 = meta
        .get(LOSS_SCALE_KEY)
        .and_then(|s| s.parse().ok())
        .with_context(|| format!("{} has no readable {LOSS_SCALE_KEY}", path.display()))?;
    let clean_streak: u32 = meta
        .get(CLEAN_STREAK_KEY)
        .and_then(|s| s.parse().ok())
        .with_context(|| format!("{} has no readable {CLEAN_STREAK_KEY}", path.display()))?;

    let read_matrix = |key: &str, expected: [usize; 2]| -> Result<Tensor<AB::InnerBackend, 2>> {
        let view = st
            .tensor(key)
            .with_context(|| format!("optimizer state {} is missing {key}", path.display()))?;
        if view.dtype() != Dtype::F32 {
            bail!("optimizer tensor {key} is {:?}, expected F32", view.dtype());
        }
        let shape: Vec<usize> = view.shape().to_vec();
        if shape != expected {
            bail!(
                "optimizer state shape mismatch at {key}: file {shape:?} vs \
                 configured {expected:?} — did lora.rank or lora.targets change?"
            );
        }
        let vals: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(Tensor::from_data(TensorData::new(vals, expected), device))
    };

    // Seeded from the optimizer's own (empty) record so the map is burn's
    // `hashbrown::HashMap`, not `std`'s — naming the type directly is an E0308
    // whose message reads as if the two were the same type.
    let mut record = optim.to_record();
    let mut restored = 0usize;
    for (delta, target) in set.deltas.iter().zip(&set.targets) {
        for (suffix, is_down) in FACTOR_KEYS {
            let (id, dims): (ParamId, [usize; 2]) = if is_down {
                (delta.lora_a.weight.id, delta.lora_a.weight.dims())
            } else {
                (delta.lora_b.weight.id, delta.lora_b.weight.dims())
            };
            let m1 = read_matrix(&moment_key(target, suffix, "exp_avg"), dims)?;
            let m2 = read_matrix(&moment_key(target, suffix, "exp_avg_sq"), dims)?;
            record.insert(
                id,
                AdaptorRecord::<AdamW, AB>::from_state::<2>(AdamWState::new(
                    AdaptiveMomentumState::new(time, m1, m2),
                )),
            );
            restored += 1;
        }
    }

    Ok((
        optim.load_record(record),
        OptimRestored {
            params: restored,
            time,
            loss_scale,
            clean_streak,
        },
    ))
}

/// The one operator-visible sentence a resumed run emits, naming what **was**
/// and what was **not** restored.
///
/// One message rather than several: a resume advisory that arrives in three
/// pieces is three things to scroll past. The "not restored" half is not
/// optional — without it a reader reasonably assumes a resumed run is a
/// bit-identical continuation, and it is not (see the module docs).
pub fn resume_message(
    plan: &ResumePlan,
    deltas: usize,
    optim: Option<&OptimRestored>,
    total: u64,
) -> String {
    let trigger = if plan.explicit {
        "resume.from"
    } else {
        "an existing final artifact (resume.auto)"
    };
    let mut msg = format!(
        "resuming from {} via {trigger}: {deltas} deltas loaded",
        plan.source.display()
    );

    match &plan.provenance {
        ResumeProvenance::Finished { steps_done } => {
            msg.push_str(&format!(
                ", {steps_done} steps already done (a completed run)"
            ));
        }
        ResumeProvenance::Unfinished {
            steps_done,
            planned,
        } => {
            let planned = planned
                .map(|p| p.to_string())
                .unwrap_or_else(|| "an unrecorded total".to_string());
            msg.push_str(&format!(
                ", {steps_done} of {planned} steps already done (the run that \
                 wrote it did not finish)"
            ));
        }
        ResumeProvenance::Unrecorded => {
            msg.push_str(
                ", but its __metadata__ records no ss_steps (metadata.embed was \
                 off, or another tool wrote it), so the step count was NOT \
                 restored and this run numbers its steps from 1",
            );
        }
    }

    // The step counter follows the provenance. On the `Unrecorded` path the
    // arm above has just said the step count was NOT restored, so claiming it
    // here contradicts that sentence — and tells a `metadata.embed: false`
    // operator the batch cursor is somewhere it is not.
    let counter = match plan.provenance {
        ResumeProvenance::Unrecorded => "",
        _ => " and the step counter — so the batch cursor follows it",
    };
    match optim {
        Some(o) => msg.push_str(&format!(
            ". Restored: adapter weights, AdamW moments for {} parameters at \
             step {}, the loss scale ({}) and clean streak ({}){counter}.",
            o.params, o.time, o.loss_scale, o.clean_streak
        )),
        None => match &plan.stale_optim {
            Some(stale) => msg.push_str(&format!(
                ". Restored: adapter weights{counter}. NOT restored: AdamW's \
                 moments — the sidecar beside the source ({}) records {} steps \
                 while the source records {}, so it belongs to a different run \
                 and was skipped rather than paired with these weights; the \
                 optimizer re-warms from zero and the loss scale restarts at \
                 its initial value.",
                stale.path.display(),
                stale.steps_done,
                stale.source_steps_done,
            )),
            None => msg.push_str(&format!(
                ". Restored: adapter weights{counter}. NOT restored: AdamW's \
                 moments — no optimizer-state sidecar sits beside the source, \
                 so the optimizer re-warms from zero and the loss scale \
                 restarts at its initial value.",
            )),
        },
    }

    msg.push_str(&format!(
        " NOT restored: the RNG stream (timestep and noise draws), so this is a \
         continuation and not a bit-identical replay. `steps` is a total, not a \
         remainder: this run executes steps {}..={total}.",
        plan.start_step()
    ));
    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, &str)]) -> LoraMetadata {
        let mut m = LoraMetadata::new();
        for (k, v) in pairs {
            m.set(*k, *v);
        }
        m
    }

    #[test]
    fn provenance_reads_the_three_states_the_header_can_express() {
        assert_eq!(
            provenance_of(&meta(&[
                ("ss_steps", "12"),
                ("ss_max_train_steps", "12"),
                ("ss_training_finished_at", "1700000000"),
            ])),
            ResumeProvenance::Finished { steps_done: 12 }
        );
        // A mid-run checkpoint: steps recorded, no finish timestamp.
        assert_eq!(
            provenance_of(&meta(&[("ss_steps", "4"), ("ss_max_train_steps", "12")])),
            ResumeProvenance::Unfinished {
                steps_done: 4,
                planned: Some(12),
            }
        );
        // `metadata.embed: false`, or a third-party file: NOT "unfinished".
        assert_eq!(provenance_of(&meta(&[])), ResumeProvenance::Unrecorded);
        assert_eq!(provenance_of(&meta(&[])).steps_done(), 0);
    }

    #[test]
    fn the_sidecar_sits_beside_its_adapter() {
        assert_eq!(
            optim_sidecar_path(Path::new("/out/lora.safetensors")),
            PathBuf::from("/out/lora.optim.safetensors")
        );
        assert_eq!(
            optim_sidecar_path(Path::new("/out/checkpoint-25.safetensors")),
            PathBuf::from("/out/checkpoint-25.optim.safetensors")
        );
    }

    /// The sidecar must never carry a key an ecosystem LoRA loader would match
    /// (#137's failure shape: an unmatched key loads without error and does
    /// nothing — so a *matched* wrong key is worse still).
    ///
    /// Built through [`moment_key`], the same function both the writer and the
    /// reader call: composing the key from literals here would compare the
    /// test against itself, and rewriting the production suffix to `.weight`
    /// would leave it green (it did — the only teeth were in the 130 s e2e
    /// suites' on-disk key enumeration).
    #[test]
    fn moment_keys_cannot_collide_with_an_adapter_export() {
        for (suffix, _) in FACTOR_KEYS {
            for kind in ["exp_avg", "exp_avg_sq"] {
                let key = moment_key("transformer_blocks.0.attn.to_q", suffix, kind);
                assert!(!key.ends_with(".weight"), "{key} looks like a LoRA factor");
                assert!(!key.ends_with(".alpha"), "{key} looks like a LoRA alpha");
            }
        }
    }

    #[test]
    fn a_named_but_missing_resume_source_is_an_error() {
        let cfg = ResumeConfig {
            from: Some(PathBuf::from("/nonexistent/does-not-exist.safetensors")),
            ..Default::default()
        };
        let err = plan_resume(&cfg, &OutputConfig::default())
            .expect_err("a typo'd resume.from must not silently start fresh");
        let msg = format!("{err}");
        assert!(msg.contains("does-not-exist.safetensors"), "{msg}");
    }

    #[test]
    fn auto_off_or_no_artifact_means_no_plan() {
        // The workspace has no `tempfile` dev-dependency (see
        // `tests/qwen3vl_template_length.rs`); a pid+nanos dir is what the rest
        // of the suite uses.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("loractl-plan-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let output = OutputConfig {
            dir: dir.clone(),
            ..Default::default()
        };
        // Nothing on disk.
        assert!(
            plan_resume(&ResumeConfig::default(), &output)
                .unwrap()
                .is_none()
        );
        // Artifact present but auto disabled: the file is never even opened.
        std::fs::write(dir.join("lora.safetensors"), b"").unwrap();
        let off = ResumeConfig {
            auto: false,
            ..Default::default()
        };
        assert!(plan_resume(&off, &output).unwrap().is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A header-only safetensors file — enough for `read_metadata`, which
    /// reads the 8-byte length + JSON and never touches the data region.
    fn header_only_file(path: &Path, pairs: &[(&str, &str)]) {
        let map: std::collections::BTreeMap<&str, &str> = pairs.iter().copied().collect();
        let header = serde_json::json!({ "__metadata__": map }).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        std::fs::write(path, bytes).unwrap();
    }

    /// The auto path refuses an unfinished artifact, `allow_unfinished` gets
    /// through, and a FINISHED artifact needs no flag — all three, because the
    /// refusal is worthless if it fires on everything and the escape hatch is
    /// worthless if it is never consulted. (It was not, once: the flag was
    /// declared and the branch did not read it.)
    #[test]
    fn the_auto_paths_finished_check_reads_the_escape_hatch_and_the_header() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("loractl-prov-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let output = OutputConfig {
            dir: dir.clone(),
            ..Default::default()
        };
        let artifact = dir.join("lora.safetensors");

        header_only_file(&artifact, &[("ss_steps", "2"), ("ss_max_train_steps", "4")]);
        let err = plan_resume(&ResumeConfig::default(), &output).expect_err("unfinished");
        let msg = format!("{err}");
        assert!(msg.contains("records 2 of 4 steps"), "{msg}");
        assert!(msg.contains("resume.allow_unfinished"), "{msg}");

        let allowed = ResumeConfig {
            allow_unfinished: true,
            ..Default::default()
        };
        let plan = plan_resume(&allowed, &output).unwrap().expect("a plan");
        assert_eq!(plan.start_step(), 3);

        // Finished: no flag needed, so the refusal above is keyed on the
        // header and not merely on "auto resume happened".
        header_only_file(
            &artifact,
            &[
                ("ss_steps", "4"),
                ("ss_max_train_steps", "4"),
                ("ss_training_finished_at", "1700000000"),
            ],
        );
        let plan = plan_resume(&ResumeConfig::default(), &output)
            .unwrap()
            .expect("a plan");
        assert_eq!(
            plan.provenance,
            ResumeProvenance::Finished { steps_done: 4 }
        );
        assert_eq!(plan.start_step(), 5);
        assert!(!plan.explicit);

        // An explicit target is never subjected to the finished check.
        header_only_file(&artifact, &[("ss_steps", "2"), ("ss_max_train_steps", "4")]);
        let explicit = ResumeConfig {
            from: Some(artifact.clone()),
            ..Default::default()
        };
        let plan = plan_resume(&explicit, &output).unwrap().expect("a plan");
        assert!(plan.explicit);
        assert_eq!(plan.start_step(), 3);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A sidecar left behind by a *different* run must not be paired with
    /// these weights.
    ///
    /// The recovery workflow the issue documents produces exactly this: copy
    /// `checkpoint-2.safetensors` over the final export and resume, and the
    /// finished run's `krea2-lora.optim.safetensors` (step 4) is still sitting
    /// beside it. Shapes match, so `load_optimizer_state`'s guard cannot see
    /// it — only the two recorded step counts can.
    #[test]
    fn a_sidecar_from_a_different_run_is_skipped_and_named() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("loractl-stale-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let output = OutputConfig {
            dir: dir.clone(),
            ..Default::default()
        };
        let artifact = dir.join("lora.safetensors");
        let sidecar = dir.join("lora.optim.safetensors");

        // Weights at step 2, moments at step 4: skipped, and the plan keeps
        // enough to say why. (Finished, so the auto path's unfinished refusal
        // is not what this test is measuring.)
        header_only_file(
            &artifact,
            &[
                ("ss_steps", "2"),
                ("ss_max_train_steps", "4"),
                ("ss_training_finished_at", "1700000000"),
            ],
        );
        header_only_file(&sidecar, &[(STEPS_DONE_KEY, "4")]);
        let plan = plan_resume(&ResumeConfig::default(), &output)
            .unwrap()
            .expect("a plan");
        assert!(plan.optim_state.is_none(), "{:?}", plan.optim_state);
        assert_eq!(
            plan.stale_optim,
            Some(StaleSidecar {
                path: sidecar.clone(),
                steps_done: 4,
                source_steps_done: 2,
            })
        );
        let msg = resume_message(&plan, 42, None, 6);
        assert!(msg.contains("NOT restored: AdamW's moments"), "{msg}");
        assert!(msg.contains("lora.optim.safetensors"), "{msg}");
        assert!(msg.contains("records 4 steps"), "{msg}");
        assert!(msg.contains("the source records 2"), "{msg}");

        // The matching pair is adopted — otherwise the check above would be
        // passing by refusing every sidecar.
        header_only_file(&sidecar, &[(STEPS_DONE_KEY, "2")]);
        let plan = plan_resume(&ResumeConfig::default(), &output)
            .unwrap()
            .expect("a plan");
        assert_eq!(plan.optim_state, Some(sidecar.clone()));
        assert!(plan.stale_optim.is_none());

        // Nothing to compare against: a `metadata.embed: false` source has no
        // ss_steps, so its own sidecar must still be adopted.
        header_only_file(&artifact, &[]);
        header_only_file(&sidecar, &[(STEPS_DONE_KEY, "4")]);
        let plan = plan_resume(&ResumeConfig::default(), &output)
            .unwrap()
            .expect("a plan");
        assert_eq!(plan.provenance, ResumeProvenance::Unrecorded);
        assert_eq!(plan.optim_state, Some(sidecar));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_resume_message_never_implies_bit_exactness() {
        let plan = ResumePlan {
            source: PathBuf::from("/out/lora.safetensors"),
            explicit: false,
            provenance: ResumeProvenance::Finished { steps_done: 6 },
            optim_state: None,
            stale_optim: None,
        };
        let without = resume_message(&plan, 42, None, 12);
        assert!(
            without.contains("NOT restored: AdamW's moments"),
            "{without}"
        );
        assert!(without.contains("RNG stream"), "{without}");
        assert!(without.contains("steps 7..=12"), "{without}");

        let with = resume_message(
            &plan,
            42,
            Some(&OptimRestored {
                params: 84,
                time: 6,
                loss_scale: 1024.0,
                clean_streak: 3,
            }),
            12,
        );
        assert!(with.contains("AdamW moments for 84 parameters"), "{with}");
        assert!(with.contains("RNG stream"), "{with}");

        // `Unrecorded` says the step count was NOT restored — so the same
        // sentence must not then claim the step counter was. (It did: the
        // clause was appended unconditionally, and the `--no-metadata` test's
        // `contains("NOT")` was satisfied by the RNG clause, so nothing saw
        // the contradiction.)
        let unrecorded = ResumePlan {
            provenance: ResumeProvenance::Unrecorded,
            ..plan.clone()
        };
        let note = resume_message(&unrecorded, 42, None, 12);
        assert!(note.contains("was NOT restored"), "{note}");
        assert!(!note.contains("the step counter"), "{note}");
        assert!(note.contains("steps 1..=12"), "{note}");
    }
}
