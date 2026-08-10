//! The image dataset pipeline — aspect-ratio bucketing + latent/conditioning
//! caching (M12, #23).
//!
//! Implements the kohya/ai-toolkit folder convention the roadmap targets: a
//! directory of images with same-named `.txt` caption files. Images are
//! grouped into **aspect-ratio buckets** (each dimension a multiple of
//! [`BUCKET_ALIGN`] = 16 — the Krea 2 constraint `ae.compression · patch`),
//! resized cover-style + center-cropped to their bucket, VAE-encoded once,
//! and the latents cached to disk; captions are conditioning-encoded once and
//! cached the same way. Caching is what makes per-step cost tractable at 12B:
//! after the first pass, training never touches the image decoder, the VAE,
//! or the text encoder again.
//!
//! ## Which buckets, and how images land in them
//!
//! [`bucket_set`] generates the set from
//! [`DatasetConfig::bucketing`](crate::config::DatasetConfig::bucketing):
//! [`generate_buckets`] (the default seven fixed ratios) or
//! [`generate_grid_buckets`] (#148, an opt-in kohya-style symmetric 2-D grid
//! that trades batch density for much tighter aspect coverage). Either way,
//! [`assign_bucket`] then places each image in the nearest bucket **in
//! log-aspect space** — the metric that minimizes the crop, since the
//! discarded fraction is `1 − min(r, r_b)/max(r, r_b)`.
//!
//! [`DatasetConfig::no_upscale`](crate::config::DatasetConfig::no_upscale)
//! (#147) then shrinks a bucket per image rather than stretching the image:
//! see [`fit_bucket`] for why that is the whole feature, and why it costs no
//! cache-fingerprint bump.
//!
//! ## Decoupled from the concrete models
//!
//! [`prepare_dataset`] takes the two encoders as **closures** rather than
//! depending on [`QwenVae`](crate::QwenVae)/[`Qwen3VlConditioner`](crate::Qwen3VlConditioner)
//! directly: the pipeline's job is files → buckets → cache → batches, and the
//! encode step is whatever the trainer wires in (M14 passes the real frozen
//! models; the offline tests pass mocks). This keeps the pipeline fully
//! testable without checkpoints and keeps model choices out of the data
//! layer.
//!
//! ## Cache layout
//!
//! Under `<dataset>/.loractl-cache/`, keyed by the image file name (latents)
//! or stem (conditioning — captions are stem-keyed by convention, so images
//! sharing a stem share a caption), the bucket shape, and a caller-supplied
//! **fingerprint** (encoder identity — e.g. `"qwen-vae-f8x16+krea2-4b-ml512"`);
//! change the fingerprint — by *any* character — and the cache misses rather
//! than serving stale tensors from a different encoder setup (the filename
//! carries a sanitized prefix plus an FNV-1a hash of the raw string, so
//! sanitization cannot alias two fingerprints):
//!
//! ```text
//! {file_name}.{w}x{h}.{fingerprint}.latent.safetensors  "latent"        [1, z, h', w']
//! {stem}.{fingerprint}.cond.safetensors                 "conditioning"  [1, s, n, d]
//!                                                       "mask"          [1, s] (f32 0/1)
//! ```
//!
//! Cache keys deliberately do **not** hash file contents: an image or
//! caption edited *in place* under the same name serves the stale cache
//! until `.loractl-cache/` is deleted. Content-hash invalidation can come
//! later if it earns its cost; delete the cache dir after editing a dataset
//! in place.
//!
//! ## Residency: O(batch), never O(dataset) (#175)
//!
//! [`PreparedDataset`] is a **plan over the on-disk cache** — a list of file
//! paths and shapes. It holds no tensor, and it takes no `B: Backend`
//! parameter, which is the point: a struct with no backend parameter
//! *cannot* hold a `Tensor<B, D>`, so dataset-scale residency cannot be
//! reintroduced without re-adding a type parameter and breaking every call
//! site at once.
//!
//! It used to hold every example's latent + conditioning as live device
//! tensors, so peak VRAM scaled with **dataset size** rather than batch
//! size — Krea 2 conditioning is `[1, 512, 12, 2560]` f32 = 60 MiB *per
//! example*, resident for the whole run, against the ~4 GB of headroom
//! ADR-0005 Addendum 3 measured at 512px int4. Worse, the trainer also
//! materialized every concatenated batch up front, so the data was resident
//! **twice**. Now [`PreparedDataset::batches`] returns [`BatchPlan`]s and the
//! step loop calls [`PreparedDataset::load_batch`], which reads, uploads,
//! and drops one batch per step.
//!
//! **What the step now pays, stated in full.** Per batch: a `read` of each
//! item's two cache files, a scalar `u8 → f32` decode of their payloads
//! ([`read_cache`] collects `chunks_exact(4)`, so ~2× the payload is
//! allocated transiently), and the upload. Only the *first* of those three is
//! served by the OS page cache — batches are visited round-robin over files
//! that were just written, so the hot working set stays cached for free while
//! host RAM allows and is evicted under pressure, which a resident `Vec`
//! could not do. The decode and the upload are genuinely new per-step work,
//! and they are worst in the common small-dataset shape (`plans.len() == 1`
//! with `steps: 1000` re-reads the same file 1000 times). **The ms/step delta
//! is unmeasured** — it needs a `gh workflow run gpu.yml` bench dispatch, and
//! nothing here estimates it. The trade was made deliberately: residency was
//! a *fit* problem (the run did not start), throughput is not. If it does
//! show up, the cheap wins are reading from the mmap rather than into a
//! `Vec<u8>` first, and `f32::from_le_bytes` over an aligned slice instead of
//! the scalar loop; an in-process cache is the one fix that would undo the
//! property this module now has.
//!
//! One consequence to know: [`plan_dataset`] reads only file **headers**, so
//! a cache file with an intact header and a garbled payload surfaces at
//! [`PreparedDataset::load_batch`] time — possibly mid-run — rather than at
//! plan time. A *truncated* file is still caught up front (safetensors'
//! `read_metadata` validates the total buffer length), and every loader
//! error names the file.
//!
//! ## Parallelism: host-side pixels only (#178)
//!
//! The encode phase's CPU floor was [`decode_image_for_bucket`] — decode,
//! Lanczos3 cover-resize, HWC→CHW transpose — run one image at a time.
//! [`prepare_dataset`] now runs it across examples on rayon, one bounded
//! window ahead of an otherwise unchanged serial loop ([`decode_window`]).
//!
//! The split is where it is because of what *cannot* move: the encode
//! closures are `FnMut`, they are GPU-bound, and no `Tensor<B, D>` in this
//! tree carries a `Send` bound, so nothing about `encode_image` is safe or
//! useful to run concurrently. Decode is a pure function of
//! `(file, bucket)`, so it is the entire parallelizable surface.
//!
//! Two invariants make that safe to have done at all, and both are pinned by
//! tests: the decode is **bit-identical** to the serial path it replaced
//! (cache keys are name/bucket/fingerprint and never content, so a changed
//! value would invalidate nothing and corrupt silently), and everything
//! observable — encoder call order, progress order, error order, cached
//! bytes — is still scan order, independent of the machine's core count.
//!
//! Neither loader draws RNG or touches `B::seed`. That is load-bearing:
//! a draw here would shift where every lazily-initialized `Param`
//! materializes in the seed stream (see `.claude/rules/burn-lazy-param-init.md`)
//! and silently change frozen-base values across the suite.
//!
//! Like the rest of `loractl-core`, this module emits no output and imports
//! no CLI.

use crate::config::{BucketMode, DatasetConfig};
use crate::export::{OwnedF32Tensor, to_owned_f32};
use anyhow::{Context, Result, bail};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Every bucket dimension is a multiple of this: Krea 2's
/// `ae.compression (8) · patch (2)` — the resolution granularity the latent
/// patch grid supports.
pub const BUCKET_ALIGN: u32 = 16;

/// The aspect ratios buckets are generated for (width : height) under
/// [`BucketMode::Aspects`].
const ASPECTS: [(u32, u32); 7] = [(1, 1), (4, 3), (3, 4), (3, 2), (2, 3), (16, 9), (9, 16)];

/// The largest [`BUCKET_ALIGN`] multiple not exceeding `v`, floored at one
/// full alignment step — a zero-sided bucket is not a bucket.
fn align_down(v: f64) -> u32 {
    ((v.max(0.0) as u32) / BUCKET_ALIGN * BUCKET_ALIGN).max(BUCKET_ALIGN)
}

/// One resolution bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bucket {
    /// Pixel width (multiple of [`BUCKET_ALIGN`]).
    pub width: u32,
    /// Pixel height (multiple of [`BUCKET_ALIGN`]).
    pub height: u32,
}

impl Bucket {
    fn aspect(&self) -> f64 {
        self.width as f64 / self.height as f64
    }
}

/// Generate the bucket set for a target `resolution`: for each aspect ratio
/// in [`ASPECTS`], the [`BUCKET_ALIGN`]-aligned box with roughly
/// `resolution²` pixels. Deduplicated; the square `resolution × resolution`
/// bucket is always present. Errors on an unaligned `resolution` —
/// this value arrives straight from user YAML, so misconfiguration must
/// surface as an error, not a panic.
pub fn generate_buckets(resolution: u32) -> Result<Vec<Bucket>> {
    if !resolution.is_multiple_of(BUCKET_ALIGN) {
        bail!(
            "dataset.resolution = {resolution} must be a multiple of {BUCKET_ALIGN} \
             (Krea 2's compression × patch grid)"
        );
    }
    let align = |v: f64| -> u32 {
        let stepped = (v / BUCKET_ALIGN as f64).round().max(1.0) as u32;
        stepped * BUCKET_ALIGN
    };
    let mut buckets = Vec::new();
    for (aw, ah) in ASPECTS {
        let aspect = aw as f64 / ah as f64;
        let w = align(resolution as f64 * aspect.sqrt());
        let h = align(resolution as f64 / aspect.sqrt());
        let bucket = Bucket {
            width: w,
            height: h,
        };
        if !buckets.contains(&bucket) {
            buckets.push(bucket);
        }
    }
    Ok(buckets)
}

/// Generate the [`BucketMode::Grid`] bucket set (#148): every
/// [`BUCKET_ALIGN`]-aligned box whose area fits inside `resolution²` and
/// whose **shorter** side is at least `min_side`, plus each box's transpose.
///
/// The transpose pass is kohya's own symmetrization (`make_bucket_resolutions`
/// emits both orientations), and it is what makes portrait and landscape
/// sources provably equally served — the fixed [`ASPECTS`] list is symmetric
/// only because it was written that way by hand.
///
/// The *longest* side is derived rather than configured: `h ≥ min_side`
/// already implies `w ≤ resolution²/min_side`, so a `max_bucket_resolution`
/// knob would only add a second value to validate against the same budget.
///
/// Every argument arrives from user YAML, so every violation is an `Err`
/// naming the field, never a panic:
///
/// - `resolution` must be a multiple of [`BUCKET_ALIGN`] (as in
///   [`generate_buckets`]).
/// - `min_side` must be a multiple of [`BUCKET_ALIGN`], so the grid's own
///   step lands on it.
/// - `min_side < resolution`. Above it the square `resolution × resolution`
///   bucket falls outside the grid and the "the square bucket is always
///   present" contract dies silently; *at* it the grid collapses to **only**
///   that square, so every image is center-cropped square — strictly worse
///   than the [`BucketMode::Aspects`] set the user opted out of. Both are the
///   same silent-surprise shape the other rules exist to prevent.
/// - `min_side ≥ resolution/4`, which caps the extreme aspect at
///   `(resolution/min_side)² = 16:1`. Note this bounds the *aspect*, not the
///   bucket count: at that floor the loop runs `(4·r − r/4)/16 + 1` times, so
///   the count is `≈ 0.23 · resolution` and scales with it — 193 buckets at
///   512px, 385 at 1024px, 769 at 2048px (pinned by
///   `dataset_pipeline.rs::grid_buckets_are_aligned_symmetric_and_area_bounded`).
///   Beyond the floor the set is dominated by boxes no real image lands in,
///   and the partial-batch cost (batches never mix buckets) stops being worth
///   it — a cost that grows with resolution for exactly the same reason.
pub fn generate_grid_buckets(resolution: u32, min_side: u32) -> Result<Vec<Bucket>> {
    if !resolution.is_multiple_of(BUCKET_ALIGN) {
        bail!(
            "dataset.resolution = {resolution} must be a multiple of {BUCKET_ALIGN} \
             (Krea 2's compression × patch grid)"
        );
    }
    if min_side == 0 || !min_side.is_multiple_of(BUCKET_ALIGN) {
        bail!(
            "dataset.min_bucket_resolution = {min_side} must be a non-zero multiple of \
             {BUCKET_ALIGN} (Krea 2's compression × patch grid)"
        );
    }
    if min_side > resolution {
        bail!(
            "dataset.min_bucket_resolution = {min_side} exceeds dataset.resolution = \
             {resolution}; the square {resolution}×{resolution} bucket would fall outside \
             the grid"
        );
    }
    if min_side == resolution {
        bail!(
            "dataset.min_bucket_resolution = {min_side} equals dataset.resolution = \
             {resolution}, so the grid collapses to the single {resolution}×{resolution} \
             square bucket and every image is center-cropped square — strictly worse than \
             the `bucketing: aspects` set this mode was chosen over. Lower it (the default \
             is resolution / 2) or drop `bucketing: grid`"
        );
    }
    // u64 throughout: this function's whole contract is "report, never
    // panic", and a debug-build overflow on an absurd resolution would be a
    // panic on exactly the misconfiguration path it exists to describe.
    if min_side as u64 * 4 < resolution as u64 {
        bail!(
            "dataset.min_bucket_resolution = {min_side} is below dataset.resolution / 4 = \
             {}; the grid's extreme aspect is (resolution / min)² = {}:1, which is past \
             the 16:1 cap",
            resolution / 4,
            (resolution as u64 / min_side as u64).pow(2)
        );
    }

    let area = resolution as f64 * resolution as f64;
    let max_side = align_down(area / min_side as f64);
    let mut buckets = Vec::new();
    let mut width = min_side;
    while width <= max_side {
        let height = align_down(area / width as f64);
        if height >= min_side {
            for bucket in [
                Bucket { width, height },
                Bucket {
                    width: height,
                    height: width,
                },
            ] {
                if !buckets.contains(&bucket) {
                    buckets.push(bucket);
                }
            }
        }
        width += BUCKET_ALIGN;
    }
    Ok(buckets)
}

/// `resolution / 2`, aligned down — the [`BucketMode::Grid`] default shortest
/// side when [`DatasetConfig::min_bucket_resolution`] is unset. Yields
/// aspects from 1:4 to 4:1, which covers ordinary photography plus the
/// panoramas the fixed list handles worst, without the bucket count a 16:1
/// grid carries.
fn default_min_side(resolution: u32) -> u32 {
    align_down(resolution as f64 / 2.0)
}

/// The bucket set a run uses: dispatches on
/// [`DatasetConfig::bucketing`] (#148). This is the single entry point
/// [`prepare_dataset`] calls; [`generate_buckets`] and
/// [`generate_grid_buckets`] stay callable directly so each generator can be
/// tested for what it is.
pub fn bucket_set(config: &DatasetConfig) -> Result<Vec<Bucket>> {
    match config.bucketing {
        BucketMode::Aspects => {
            // `config.rs` carries no `deny_unknown_fields`, so a knob that is
            // merely ignored is this schema's quietest failure — a user sets
            // `min_bucket_resolution`, sees the seven fixed buckets, and has
            // nothing to read that says why. Refuse it instead.
            if let Some(min_side) = config.min_bucket_resolution {
                bail!(
                    "dataset.min_bucket_resolution = {min_side} is only meaningful with \
                     `bucketing: grid`; the aspects set is generated from a fixed ratio \
                     list and has no minimum side"
                );
            }
            generate_buckets(config.resolution)
        }
        BucketMode::Grid => generate_grid_buckets(
            config.resolution,
            config
                .min_bucket_resolution
                .unwrap_or_else(|| default_min_side(config.resolution)),
        ),
    }
}

/// The largest [`BUCKET_ALIGN`]-aligned box with `bucket`'s aspect ratio that
/// fits inside a `width × height` source — [`DatasetConfig::no_upscale`]'s
/// whole implementation (#147). Returns `bucket` unchanged when the source
/// already covers it (the common case: a downscale, which was never the
/// problem). Floors each side at [`BUCKET_ALIGN`].
///
/// Expressing "don't upscale" as a *smaller bucket* rather than as a scale cap
/// inside [`load_image_for_bucket`] is deliberate, and load-bearing three ways:
///
/// - A capped scale leaves the crop rectangle larger than the unscaled source,
///   so it would have to **pad** — feeding synthetic borders into the VAE,
///   which is strictly worse than the upscaling it replaces.
/// - The pixel pipeline stays a pure function of `(image, bucket box)`, so
///   [`load_image_for_bucket`] needs no edit at all: the cover scale it
///   computes for a fitted box is `≤ 1` by construction.
/// - The latent cache key already carries `{w}x{h}`, so flipping the knob
///   needs **no fingerprint bump**. An image whose treatment changes gets a
///   strictly smaller box, hence a different filename; one whose treatment
///   does not keeps its box and correctly reuses its latent — and the
///   conditioning cache (stem-keyed) is untouched either way, so the 4B text
///   encoder never re-runs.
pub fn fit_bucket(bucket: Bucket, width: u32, height: u32) -> Bucket {
    let scale = f64::max(
        bucket.width as f64 / width as f64,
        bucket.height as f64 / height as f64,
    );
    if scale <= 1.0 {
        return bucket;
    }
    Bucket {
        width: align_down(bucket.width as f64 / scale),
        height: align_down(bucket.height as f64 / scale),
    }
}

/// The bucket whose aspect ratio is nearest (in log space) to
/// `width × height`'s.
pub fn assign_bucket(buckets: &[Bucket], width: u32, height: u32) -> usize {
    let aspect = (width as f64 / height as f64).ln();
    buckets
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (a.aspect().ln() - aspect).abs();
            let db = (b.aspect().ln() - aspect).abs();
            da.partial_cmp(&db).expect("finite aspects")
        })
        .expect("bucket set is non-empty")
        .0
}

/// One dataset entry: an image, its caption, and its assigned bucket.
#[derive(Debug, Clone)]
pub struct DatasetEntry {
    /// The image file.
    pub image_path: PathBuf,
    /// The caption (contents of the same-stem `.txt`, trimmed; empty when no
    /// caption file exists — an unconditional example).
    pub caption: String,
    /// Index into the bucket set.
    pub bucket: usize,
}

/// Scan a kohya-style dataset folder: every `.png`/`.jpg`/`.jpeg` image (with
/// an optional same-stem `.txt` caption), each assigned to its nearest
/// bucket. Sorted by filename for determinism. Errors when the folder holds
/// no images (fail fast — an empty dataset is a configuration mistake).
///
/// `buckets` is the generated set on entry and is taken by `&mut` because
/// `no_upscale` can **grow** it: an image smaller than its nearest bucket
/// gets a [`fit_bucket`]-derived box appended (deduplicated), which is what
/// keeps [`load_image_for_bucket`] from ever scaling up.
///
/// **The two passes are not a style choice.** Sorting happens *before* any
/// assignment because the moment the bucket set can grow during assignment,
/// bucket *indices* would depend on `read_dir` order — and
/// [`PreparedDataset::batches`] iterates buckets by index, so batch order
/// (which every pinned loss trajectory depends on) would become
/// filesystem-dependent. Assignment is a pure function, so this is
/// behavior-preserving with `no_upscale` off.
pub(crate) fn scan_dataset(
    dir: &Path,
    buckets: &mut Vec<Bucket>,
    no_upscale: bool,
) -> Result<Vec<DatasetEntry>> {
    // Pass 1: enumerate. No bucket is assigned here — `read_dir` order is
    // unspecified and must not reach any output.
    let mut scanned: Vec<(PathBuf, String, u32, u32)> = Vec::new();
    let read = std::fs::read_dir(dir)
        .with_context(|| format!("reading dataset directory {}", dir.display()))?;
    for entry in read {
        let path = entry?.path();
        let is_image = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"));
        if !is_image {
            continue;
        }
        let (width, height) = image::image_dimensions(&path)
            .with_context(|| format!("reading dimensions of {}", path.display()))?;
        let caption_path = path.with_extension("txt");
        let caption = match std::fs::read_to_string(&caption_path) {
            Ok(text) => text.trim().to_string(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => {
                return Err(e)
                    .with_context(|| format!("reading caption {}", caption_path.display()));
            }
        };
        scanned.push((path, caption, width, height));
    }
    if scanned.is_empty() {
        bail!("no .png/.jpg/.jpeg images found in {}", dir.display());
    }
    scanned.sort_by(|a, b| a.0.cmp(&b.0));

    // Pass 2: assign, in the sorted order. Nearest-aspect is always resolved
    // against the *generated* set — derived boxes are per-image consequences,
    // never candidates another image could be attracted to.
    let generated = buckets.len();
    let mut entries = Vec::with_capacity(scanned.len());
    for (image_path, caption, width, height) in scanned {
        let nearest = assign_bucket(&buckets[..generated], width, height);
        let bucket = if no_upscale {
            let fitted = fit_bucket(buckets[nearest], width, height);
            match buckets.iter().position(|b| *b == fitted) {
                Some(existing) => existing,
                None => {
                    buckets.push(fitted);
                    buckets.len() - 1
                }
            }
        } else {
            nearest
        };
        entries.push(DatasetEntry {
            image_path,
            caption,
            bucket,
        });
    }
    Ok(entries)
}

/// Decode an image, resize it cover-style to its bucket (preserving aspect,
/// so the shorter relative side fits exactly), center-crop the overflow, and
/// return it as **host-side CHW f32** in `[-1, 1]` (the VAE's input range),
/// laid out `data[c · height · width + y · width + x]` — exactly the buffer
/// [`load_image_for_bucket`] hands to `Tensor::from_data`.
///
/// Split out of the tensor constructor for #178 because this is the whole CPU
/// cost of the encode phase (decode + Lanczos3 resize + transpose), it touches
/// no backend, and it is a **pure function of `(file, bucket)`** — which is
/// what lets [`prepare_dataset`] run it across examples on rayon while the
/// GPU-bound encode stays strictly serial. Nothing here draws RNG (see the
/// module docs' residency section for why that matters).
///
/// ## Bit-identical to the pre-#178 serial path, by construction
///
/// Same decode, same filter at the same target size, same center-crop origin,
/// same `p / 127.5 − 1.0`. Two things changed, and neither touches a value:
///
/// - The crop is **read** by indexing the resized buffer at the crop origin,
///   rather than materializing `crop_imm(…).to_image()` — a second full-frame
///   allocation and copy of pixels that are then read exactly once.
/// - The transpose is **written** as three contiguous channel planes, filled
///   row by row, rather than recomputing `c · bh · bw + y · bw + x` per pixel
///   per channel — the least efficient possible form of a transpose, and the
///   one shape that defeats both the prefetcher and autovectorization.
///
/// That this stays exact is not a nicety. Cache keys are
/// name/bucket/fingerprint and **never content**, so a resize or transpose
/// change invalidates nothing on disk: a shifted value would silently mix
/// old-algorithm and new-algorithm latents inside one adapter, with no error
/// anywhere. Pinned by `dataset_pipeline.rs`'s
/// `decode_is_bit_identical_to_the_serial_reference`, which keeps the old
/// loader verbatim as its oracle.
pub fn decode_image_for_bucket(path: &Path, bucket: Bucket) -> Result<Vec<f32>> {
    let img = image::open(path)
        .with_context(|| format!("decoding {}", path.display()))?
        .to_rgb8();
    let (w, h) = (img.width(), img.height());
    let (bw, bh) = (bucket.width, bucket.height);

    // Cover: scale so both dimensions reach the bucket, then center-crop.
    // `rw ≥ bw` and `rh ≥ bh` hold by construction (the scale is the max of
    // the two ratios and the resize rounds up), so the crop window below is
    // always inside the resized frame.
    let scale = f64::max(bw as f64 / w as f64, bh as f64 / h as f64);
    let rw = (w as f64 * scale).ceil() as u32;
    let rh = (h as f64 * scale).ceil() as u32;
    let resized = image::imageops::resize(&img, rw, rh, image::imageops::FilterType::Lanczos3);

    // HWC u8 → CHW f32 in [-1, 1], one contiguous plane per channel. The
    // crop is applied as an offset into the resized frame's raw samples
    // (`ImageBuffer` is row-major RGB with no padding, so the row for output
    // `y` starts at `((oy + y) · rw + ox) · 3`).
    let (ox, oy) = (((rw - bw) / 2) as usize, ((rh - bh) / 2) as usize);
    let (bw, bh, rw) = (bw as usize, bh as usize, rw as usize);
    let raw = resized.as_raw();
    let mut data = vec![0.0f32; 3 * bh * bw];
    let (red, rest) = data.split_at_mut(bh * bw);
    let (green, blue) = rest.split_at_mut(bh * bw);
    for y in 0..bh {
        let start = ((oy + y) * rw + ox) * 3;
        let src = &raw[start..start + bw * 3];
        let row = y * bw..(y + 1) * bw;
        let red = &mut red[row.clone()];
        let green = &mut green[row.clone()];
        let blue = &mut blue[row];
        for (((px, r), g), b) in src
            .chunks_exact(3)
            .zip(red.iter_mut())
            .zip(green.iter_mut())
            .zip(blue.iter_mut())
        {
            *r = px[0] as f32 / 127.5 - 1.0;
            *g = px[1] as f32 / 127.5 - 1.0;
            *b = px[2] as f32 / 127.5 - 1.0;
        }
    }
    Ok(data)
}

/// Wrap a [`decode_image_for_bucket`] buffer as the `[1, 3, h, w]` tensor the
/// image encoder takes. The upload is the only backend-touching part of the
/// image path, which is why it is separated from the decode.
fn image_tensor<B: Backend>(data: Vec<f32>, bucket: Bucket, device: &B::Device) -> Tensor<B, 4> {
    let (bw, bh) = (bucket.width as usize, bucket.height as usize);
    Tensor::from_data(TensorData::new(data, [1, 3, bh, bw]), device)
}

/// [`decode_image_for_bucket`] uploaded to `device` as a `[1, 3, height,
/// width]` tensor — the whole image path in one call, for callers with no
/// reason to hold the host buffer.
pub fn load_image_for_bucket<B: Backend>(
    path: &Path,
    bucket: Bucket,
    device: &B::Device,
) -> Result<Tensor<B, 4>> {
    Ok(image_tensor(
        decode_image_for_bucket(path, bucket)?,
        bucket,
        device,
    ))
}

/// A cache-file tensor as raw values + shape.
type CachedTensor = (Vec<f32>, Vec<usize>);

/// FNV-1a (64-bit) over a string — the injective-enough suffix that keeps
/// distinct fingerprints from colliding after filename sanitization.
fn fnv1a64(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Read named f32 tensors from a cache file; `None` on a miss (and *only* on
/// `NotFound` — a present-but-corrupt file is an `Err`, never a silent
/// re-encode).
fn read_cache(path: &Path, names: &[&str]) -> Result<Option<Vec<CachedTensor>>> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parsing cache file {}", path.display()))?;
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let view = st
            .tensor(name)
            .with_context(|| format!("cache file {} lacks '{name}'", path.display()))?;
        if view.dtype() != safetensors::Dtype::F32 {
            bail!("cache tensor '{name}' in {} is not F32", path.display());
        }
        let values: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        out.push((values, view.shape().to_vec()));
    }
    Ok(Some(out))
}

/// Like [`read_cache`], but a **missing file is an error**, not `None`.
///
/// This is the bail-on-miss contract at *step* granularity: a per-step read
/// has far more chances to miss than a single up-front pass, and every one of
/// them must be loud. Threading it through the type — `Result<Vec<_>>`, not
/// `Result<Option<Vec<_>>>` — is what keeps a future edit from quietly
/// treating a miss as "nothing to do".
fn read_cache_required(path: &Path, names: &[&str]) -> Result<Vec<CachedTensor>> {
    match read_cache(path, names)? {
        Some(tensors) => Ok(tensors),
        None => bail!(
            "cache file {} disappeared mid-run — the dataset must not change while training",
            path.display()
        ),
    }
}

/// Shapes for named f32 tensors **without reading their data**; `None` on a
/// miss, with the same `NotFound`-only rule as [`read_cache`].
///
/// The file is mmapped and only its header is parsed, so planning a 5000-image
/// dataset faults in a few KiB per entry rather than 5000 × 60 MiB of
/// conditioning payload (#175). Two properties come along for free:
/// `SafeTensors::read_metadata` validates `header + data == file length`, so a
/// **truncated** cache file is caught here, before the model loads; and the
/// dtype check stays exactly where [`read_cache`]'s is, so a non-F32 cache
/// file is rejected at plan time too.
///
/// `unsafe`: mmap's standard caveat — a file **shrinking** while it is mapped
/// raises SIGBUS on the next page touch, which `std::fs::read` (what this
/// replaced) cannot do. Two facts bound it, and the second is why
/// [`DatasetCache::write`] publishes by `rename`:
///
/// - **Within a process**, no cache file is ever mapped and written at the
///   same time. [`prepare_dataset`] calls this from its decode pre-pass, in
///   the same loop that writes — but each entry's paths are unique to that
///   entry and each is mapped strictly *before* it is written, never
///   concurrently. (`plan_dataset` writes nothing at all.)
/// - **Across processes**, two runs over one dataset directory share cache
///   filenames byte-for-byte (the fingerprint is encoder identity, not run
///   identity), so a sweep started with a cold cache used to be able to
///   truncate a file another run had mapped. `write` therefore serializes to
///   a temp file and renames into place: a cache file is only ever created
///   whole and replaced atomically, so it never shrinks under a mapping.
///
/// (A header-only `read` is not an alternative: `read_metadata` rejects a
/// prefix, by the very length check that makes it useful.)
fn read_cache_shapes(path: &Path, names: &[&str]) -> Result<Option<Vec<Vec<usize>>>> {
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .with_context(|| format!("mmapping {}", path.display()))?;
    let (_, meta) = safetensors::SafeTensors::read_metadata(&mmap)
        .with_context(|| format!("parsing cache file {}", path.display()))?;
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let info = meta
            .info(name)
            .with_context(|| format!("cache file {} lacks '{name}'", path.display()))?;
        if info.dtype != safetensors::Dtype::F32 {
            bail!("cache tensor '{name}' in {} is not F32", path.display());
        }
        out.push(info.shape.clone());
    }
    Ok(Some(out))
}

/// A cached tensor's shape as a fixed-rank array. Rank is checked at **plan**
/// time (it used to be checked per load), so a wrong-rank cache file fails
/// before the MMDiT is loaded rather than at the first step.
fn ranked<const N: usize>(shape: &[usize], name: &str, path: &Path) -> Result<[usize; N]> {
    shape.try_into().map_err(|_| {
        anyhow::anyhow!(
            "cached '{name}' in {} is rank-{}, expected rank-{N}",
            path.display(),
            shape.len()
        )
    })
}

/// The planned-vs-found shape check every load runs.
///
/// The plan records what each cache file contained when the run started; a
/// file rewritten *since* then is a different training example wearing the
/// same name. The previous design could not detect that at all — it read
/// whatever was on disk and trained on it. Cheap (two slice compares per
/// tensor) and the only guard against a silently swapped example.
fn check_shape(path: &Path, name: &str, planned: &[usize], found: &[usize]) -> Result<()> {
    if planned != found {
        bail!(
            "cache file {} changed since the run was planned: '{name}' was {planned:?}, \
             is now {found:?} — the dataset must not change while training",
            path.display()
        );
    }
    Ok(())
}

/// The on-disk latent/conditioning cache (see the module docs for layout).
///
/// Owns key derivation and writing only; reads are the free functions above,
/// because [`PreparedDataset`] holds resolved paths and must be able to load
/// from them without carrying a cache handle (and therefore a fingerprint it
/// could disagree with).
struct DatasetCache {
    dir: PathBuf,
    fingerprint: String,
}

impl DatasetCache {
    fn new(dataset_dir: &Path, fingerprint: &str) -> Result<Self> {
        let dir = dataset_dir.join(".loractl-cache");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating cache dir {}", dir.display()))?;
        // Filename-safe AND injective: a readable sanitized prefix plus an
        // FNV-1a hash of the RAW fingerprint, so fingerprints differing only
        // in sanitized-away characters ("qwen_vae.f8" vs "qwen-vae+f8", both
        // sanitizing to "qwen-vae-f8") cannot serve each other's cache.
        let sanitized: String = fingerprint
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' {
                    c
                } else {
                    '-'
                }
            })
            .collect();
        let fingerprint = format!("{sanitized}-{:016x}", fnv1a64(fingerprint));
        Ok(Self { dir, fingerprint })
    }

    fn latent_path(&self, file_name: &str, bucket: Bucket) -> PathBuf {
        self.dir.join(format!(
            "{file_name}.{}x{}.{}.latent.safetensors",
            bucket.width, bucket.height, self.fingerprint
        ))
    }

    fn cond_path(&self, stem: &str) -> PathBuf {
        self.dir
            .join(format!("{stem}.{}.cond.safetensors", self.fingerprint))
    }

    /// Write one cache file **atomically**: serialize to a per-process temp
    /// name in the same directory, then `rename` into place.
    ///
    /// `safetensors::serialize_to_file` is `File::create` + incremental
    /// `write_all`, so writing straight to `path` truncates it in place and
    /// grows it back over many syscalls. That leaves two windows this cache
    /// has no other defence against: an interrupted run leaves a
    /// short-but-present file that the next run reads as a hit (caught as a
    /// parse error, but only after the header check that `read_cache_shapes`
    /// does — and only because that check exists), and a *concurrent* run
    /// over the same dataset directory can truncate a file this process has
    /// mmapped, which is a SIGBUS rather than an error. `rename` within one
    /// directory is atomic, so a cache file is only ever absent or complete.
    fn write(&self, path: &Path, tensors: Vec<(&str, OwnedF32Tensor)>) -> Result<()> {
        let views: Vec<(&str, &OwnedF32Tensor)> = tensors.iter().map(|(k, t)| (*k, t)).collect();
        // Per-process temp name: two concurrent runs must not share the
        // staging file either, or the rename would publish a torn one.
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        safetensors::serialize_to_file(views, None, &tmp)
            .with_context(|| format!("writing cache file {}", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("publishing cache file {}", path.display()))
    }
}

/// One prepared example, **loaded**: cached tensors plus its bucket,
/// batchable with any other example from the same bucket.
///
/// Produced on demand by [`PreparedDataset::load_item`]; nothing retains one
/// for longer than a step. The plan-side counterpart is [`CachedItem`].
pub struct PreparedItem<B: Backend> {
    /// Normalized VAE latent `[1, z, h', w']`.
    pub latent: Tensor<B, 4>,
    /// Conditioning stack `[1, s, n, d]`.
    pub conditioning: Tensor<B, 4>,
    /// The conditioning key mask `[1, s]` (0/1).
    pub mask: Tensor<B, 2, Int>,
    /// Index into [`PreparedDataset::buckets`].
    pub bucket: usize,
}

/// One prepared example, **planned**: *where* its cached tensors live and
/// what shape they are. Deliberately holds no tensor — see the module docs'
/// residency section.
///
/// The recorded shapes are not redundant with the files: they are the plan's
/// record of what each file contained when the run started, which is what
/// makes a mid-run rewrite detectable ([`check_shape`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedItem {
    /// The `*.latent.safetensors` file holding `"latent"`.
    pub latent_path: PathBuf,
    /// The `*.cond.safetensors` file holding `"conditioning"` + `"mask"`.
    pub cond_path: PathBuf,
    /// `[1, z, h', w']`.
    pub latent_shape: [usize; 4],
    /// `[1, s, n, d]`.
    pub cond_shape: [usize; 4],
    /// `[1, s]`.
    pub mask_shape: [usize; 2],
    /// Index into [`PreparedDataset::buckets`].
    pub bucket: usize,
}

/// The prepared dataset: a **plan** over the on-disk cache.
///
/// This type has no `B: Backend` parameter on purpose (#175) — see the
/// module docs. If it ever needs one again, the O(dataset) residency
/// regression is back.
#[derive(Debug, Clone)]
pub struct PreparedDataset {
    /// One entry per training image, in scan order.
    pub items: Vec<CachedItem>,
    /// The bucket set the items reference.
    pub buckets: Vec<Bucket>,
    /// The scan that produced [`Self::items`], index-for-index — captions and
    /// bucket assignments kept alongside the cache plan so the adapter's
    /// `ss_tag_frequency`/`ss_bucket_info` metadata
    /// ([`crate::metadata`]) describes the data this run actually consumed,
    /// rather than a second scan of a folder that may have changed since.
    pub entries: Vec<DatasetEntry>,
}

/// Which items make up one batch, and from which bucket.
///
/// A plan rather than tensors: the step loop turns exactly one of these into
/// device memory at a time via [`PreparedDataset::load_batch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchPlan {
    /// Index into [`PreparedDataset::buckets`] — every item shares it.
    pub bucket: usize,
    /// Indices into [`PreparedDataset::items`], in scan order.
    pub items: Vec<usize>,
}

/// One training batch: same-bucket examples concatenated on the batch dim.
pub struct DatasetBatch<B: Backend> {
    /// Latents `[b, z, h', w']`.
    pub latents: Tensor<B, 4>,
    /// Conditioning `[b, s, n, d]`.
    pub conditioning: Tensor<B, 4>,
    /// Conditioning mask `[b, s]`.
    pub mask: Tensor<B, 2, Int>,
}

impl PreparedDataset {
    /// Group items per bucket and chunk into batches of at most
    /// `batch_size`. Batches never mix buckets (shapes differ across
    /// buckets); the final chunk of a bucket may be smaller.
    ///
    /// Returns **plans**, not tensors — but `batches().len()` still means
    /// exactly what it did, including as the exported
    /// `ss_num_batches_per_epoch`.
    ///
    /// `batch_size == 0` asserts rather than errors: it is a caller bug, not
    /// user input (the trainer pre-clamps with `.max(1)`).
    pub fn batches(&self, batch_size: usize) -> Vec<BatchPlan> {
        assert!(batch_size > 0, "batch_size must be positive");
        let mut batches = Vec::new();
        for bucket in 0..self.buckets.len() {
            let indices: Vec<usize> = (0..self.items.len())
                .filter(|&i| self.items[i].bucket == bucket)
                .collect();
            for chunk in indices.chunks(batch_size) {
                batches.push(BatchPlan {
                    bucket,
                    items: chunk.to_vec(),
                });
            }
        }
        batches
    }

    /// Read one item's cached tensors and upload them to `device`.
    ///
    /// Every file must be present and must still hold the shape the plan
    /// recorded; both failures are hard errors naming the file (see
    /// [`read_cache_required`] and [`check_shape`]).
    ///
    /// **Draws no RNG and never calls `B::seed`** — `Tensor::from_data` and
    /// `.int()` draw none, which is what keeps every pinned loss trajectory
    /// bit-identical now that loading happens inside the step loop. Do not
    /// add a shuffle, a random crop, or a jitter here.
    pub fn load_item<B: Backend>(
        &self,
        index: usize,
        device: &B::Device,
    ) -> Result<PreparedItem<B>> {
        let item = self.items.get(index).with_context(|| {
            format!(
                "dataset item {index} is out of range ({} items)",
                self.items.len()
            )
        })?;

        let mut cached = read_cache_required(&item.latent_path, &["latent"])?;
        let (values, shape) = cached.remove(0);
        check_shape(&item.latent_path, "latent", &item.latent_shape, &shape)?;
        let latent = Tensor::from_data(TensorData::new(values, shape), device);

        // The mask rides inside the conditioning file (the cache is
        // single-dtype, so it is stored as f32 0/1 and converted back here).
        let mut cached = read_cache_required(&item.cond_path, &["conditioning", "mask"])?;
        let (mask_values, mask_shape) = cached.remove(1);
        let (cond_values, cond_shape) = cached.remove(0);
        check_shape(
            &item.cond_path,
            "conditioning",
            &item.cond_shape,
            &cond_shape,
        )?;
        check_shape(&item.cond_path, "mask", &item.mask_shape, &mask_shape)?;
        let conditioning = Tensor::from_data(TensorData::new(cond_values, cond_shape), device);
        let mask =
            Tensor::<B, 2>::from_data(TensorData::new(mask_values, mask_shape), device).int();

        Ok(PreparedItem {
            latent,
            conditioning,
            mask,
            bucket: item.bucket,
        })
    }

    /// Read one batch's items and concatenate them on the batch dim. Rows
    /// stay in the plan's order, latent paired with conditioning paired with
    /// mask. Draws no RNG — see [`Self::load_item`].
    pub fn load_batch<B: Backend>(
        &self,
        plan: &BatchPlan,
        device: &B::Device,
    ) -> Result<DatasetBatch<B>> {
        if plan.items.is_empty() {
            // `batches()` never emits one, but this is a `pub` entry point and
            // `Tensor::cat` panics on an empty list.
            bail!("empty batch plan for bucket {}", plan.bucket);
        }
        let mut latents = Vec::with_capacity(plan.items.len());
        let mut conditioning = Vec::with_capacity(plan.items.len());
        let mut masks = Vec::with_capacity(plan.items.len());
        for &index in &plan.items {
            let item = self.load_item::<B>(index, device)?;
            latents.push(item.latent);
            conditioning.push(item.conditioning);
            masks.push(item.mask);
        }
        Ok(DatasetBatch {
            latents: Tensor::cat(latents, 0),
            conditioning: Tensor::cat(conditioning, 0),
            mask: Tensor::cat(masks, 0),
        })
    }
}

/// One progress report from [`prepare_dataset`], emitted **before** each
/// entry is processed.
///
/// Deliberately not a [`TrainEvent`](crate::TrainEvent): this module's job is
/// files → buckets → cache → batches, and it stays decoupled from the event
/// stream (see the module docs). The trainer maps these to
/// [`TrainEvent::Phase`](crate::TrainEvent::Phase).
///
/// It is emitted *before* the work because the work is the slow part: on the
/// real 4B text encoder a single cache miss costs minutes, so a report that
/// arrived afterwards would name the entry the operator already waited for.
#[derive(Debug, Clone, Copy)]
pub struct DatasetProgress<'a> {
    /// Entries fully processed before this one (0-based index of this entry).
    pub done: usize,
    /// Total entries in the dataset scan.
    pub total: usize,
    /// File name of the entry about to be processed.
    pub name: &'a str,
    /// `true` when both this entry's latent and conditioning are already
    /// cached, so processing it is a pair of disk reads rather than an encode.
    pub cached: bool,
}

/// One entry's cache coordinates — derived identically by [`prepare_dataset`]
/// and [`plan_dataset`], so the two can never disagree about where a file is.
struct EntryPaths<'a> {
    file_name: &'a str,
    stem: &'a str,
    latent: PathBuf,
    cond: PathBuf,
}

/// Resolve one entry's cache paths.
///
/// The latent keys on the **full file name**: `a.png` and `a.jpg` share a stem
/// (and thus a caption — the kohya convention keys captions by stem, so the
/// conditioning file sharing below is correct) but are different pixels, so
/// their latents must not collide.
fn entry_paths<'a>(
    entry: &'a DatasetEntry,
    bucket: Bucket,
    cache: &DatasetCache,
) -> Result<EntryPaths<'a>> {
    let file_name = entry
        .image_path
        .file_name()
        .and_then(|s| s.to_str())
        .with_context(|| format!("non-UTF-8 image name {}", entry.image_path.display()))?;
    let stem = entry
        .image_path
        .file_stem()
        .and_then(|s| s.to_str())
        .with_context(|| format!("non-UTF-8 image name {}", entry.image_path.display()))?;
    Ok(EntryPaths {
        file_name,
        stem,
        latent: cache.latent_path(file_name, bucket),
        cond: cache.cond_path(stem),
    })
}

/// Where one entry's latent comes from — the **single** hit/miss decision, made
/// once by [`prepare_dataset`]'s parallel pre-pass and then acted on by its
/// serial loop.
///
/// Deciding once is what keeps the two halves from disagreeing. An earlier
/// draft let the pre-pass guess with a cheap `is_file` probe and re-decided in
/// the loop; the two can only ever agree, so the second decision was dead
/// weight — and it made a pre-pass that decoded more than it should
/// *unobservable*, because the surplus buffers (errors included) were dropped
/// unread. Here every decode the pre-pass performs is one the loop encodes.
enum LatentSource {
    /// Already on disk; the recorded shape came from the file header.
    Cached([usize; 4]),
    /// A cache miss, decoded to host-side CHW f32 by
    /// [`decode_image_for_bucket`] and waiting for the encoder.
    Decoded(Vec<f32>),
}

/// How many entries [`prepare_dataset`]'s pre-pass holds in flight.
///
/// Bounded on purpose. A whole-dataset pre-pass would be the #175 residency
/// bug reintroduced in *host* memory: the CHW buffer the pre-pass **retains**
/// per entry is `3 · bucket.width · bucket.height · 4` bytes (~3 MiB at
/// 512px, ~12.6 MiB at 1024px), so 5000 of them is 15–63 GB. One window per
/// `rayon::current_num_threads()` (capped at 16) keeps every core fed while
/// the serial encode drains it. Pinned by
/// `dataset_residency.rs::the_cold_encode_pre_pass_is_bounded_by_the_decode_window`,
/// because a comment is not a bound.
///
/// **The retained buffer is not the peak, though.** Each *in-flight* decode
/// also holds its decoded RGB8 source and the full-frame Lanczos3 result,
/// `rw · rh · 3` bytes — and `rw · rh` is `w · h · scale²`, which the bucket
/// area does **not** bound for an aspect-mismatched source: a 4000×250 banner
/// into a 688×384 bucket resizes to 6144×384 = 7.1 MB, and a degenerate
/// 8000×20 strip to 177 MB. With `window` decodes in flight that term is
/// multiplied by the machine's core count, which is the one quantity nothing
/// else here is allowed to depend on. Two levers if it bites:
/// `RAYON_NUM_THREADS=1`, and
/// [`no_upscale`](crate::config::DatasetConfig::no_upscale) (a fitted box
/// forces `scale ≤ 1`, so the resized frame cannot exceed the source).
///
/// `RAYON_NUM_THREADS=1` collapses the window to one, which is exactly the
/// pre-#178 decode/encode interleaving — a free escape hatch if a decode ever
/// needs bisecting, and the reason nothing observable may depend on the
/// window size.
fn decode_window() -> usize {
    rayon::current_num_threads().clamp(1, 16)
}

/// Scan, bucket, and encode-into-the-cache every example of the dataset at
/// `config.path` (see the module docs). **The encoding entry point** — the
/// training path uses [`plan_dataset`] instead.
///
/// - `fingerprint` names the encoder setup for cache keying.
/// - `encode_image` maps a `[1, 3, h, w]` image in `[-1, 1]` to its latent
///   `[1, z, h', w']` (M14 wires [`QwenVae::encode`](crate::QwenVae::encode)).
/// - `encode_caption` maps a caption to its conditioning stack + mask (M14
///   wires [`Qwen3VlConditioner::encode_captions`](crate::Qwen3VlConditioner::encode_captions)).
/// - `progress` is called once per entry, before that entry is processed —
///   see [`DatasetProgress`]. Pass `|_| {}` when nothing surfaces it.
///
/// Both closures run **once per example on a cache miss and never on a
/// hit** — after the first pass, epochs re-read pure tensor files.
///
/// The returned plan holds **no tensor** (#175): on a hit only the file
/// header is read, and on a miss the freshly encoded tensor is written and
/// dropped at the end of the iteration. So this pass is O(1)-resident too —
/// it used to materialize every 60 MiB conditioning into host memory and keep
/// all of them.
///
/// ## Parallel decode, serial encode (#178)
///
/// The CPU half of a cache miss — decode, Lanczos3 cover-resize, HWC→CHW
/// transpose ([`decode_image_for_bucket`]) — is a pure function of
/// `(file, bucket)` and was the encode phase's single-threaded floor. It now
/// runs on rayon, one bounded window ahead of the loop below
/// ([`decode_window`]), together with the cache-header read that decides
/// whether a decode is needed at all ([`LatentSource`]).
///
/// The **encode stays serial, and must**: both closures are `FnMut`, they are
/// GPU-bound, and burn device tensors carry no `Send` bound anywhere in this
/// tree — nothing about `encode_image` is safe or useful to run concurrently.
/// So the parallelism is confined to the host-side pixel work, which feeds an
/// unchanged serial loop. Three properties follow, and are tested:
///
/// - **Scan order is preserved.** The window is collected into a
///   position-indexed `Vec` and consumed by `zip`, so the encoders (and the
///   progress sink, and therefore the cache writes) see exactly the order they
///   saw before — including for *errors*, which surface at the lowest failing
///   index rather than from whichever thread lost the race.
/// - **A hit still decodes nothing**, so a warm epoch does no pixel work at
///   all. That is a property of [`LatentSource`], not of an eager decode whose
///   result is discarded: the pre-pass reads the header first and decodes only
///   on a miss, and every buffer it produces is one the loop then encodes.
/// - **Nothing observable depends on the window size**, so a machine's core
///   count cannot change a single cached byte.
pub fn prepare_dataset<B: Backend>(
    config: &DatasetConfig,
    fingerprint: &str,
    device: &B::Device,
    mut encode_image: impl FnMut(Tensor<B, 4>) -> Result<Tensor<B, 4>>,
    mut encode_caption: impl FnMut(&str) -> Result<(Tensor<B, 4>, Tensor<B, 2, Int>)>,
    mut progress: impl FnMut(DatasetProgress<'_>),
) -> Result<PreparedDataset> {
    let mut buckets = bucket_set(config)?;
    let entries = scan_dataset(&config.path, &mut buckets, config.no_upscale)?;
    let cache = DatasetCache::new(&config.path, fingerprint)?;

    let total = entries.len();
    // Resolved up front so the decode pre-pass can consult a latent path
    // without re-deriving it, and so the two derivations cannot disagree.
    let cache_paths = entries
        .iter()
        .map(|entry| entry_paths(entry, buckets[entry.bucket], &cache))
        .collect::<Result<Vec<_>>>()?;

    let mut items = Vec::with_capacity(entries.len());
    let window = decode_window();
    for start in (0..total).step_by(window) {
        let end = (start + window).min(total);

        // The parallel half (#178): for every entry in this window, read the
        // latent cache header and — only on a miss — decode/resize/transpose
        // the image into a host CHW buffer. The `Result` is *carried* rather
        // than propagated, so a failure surfaces from the serial loop below at
        // the lowest failing index, not from whichever thread lost the race.
        let sources: Vec<Result<LatentSource>> = (start..end)
            .into_par_iter()
            .map(|i| {
                let entry = &entries[i];
                let latent = &cache_paths[i].latent;
                match read_cache_shapes(latent, &["latent"])? {
                    Some(shapes) => Ok(LatentSource::Cached(ranked::<4>(
                        &shapes[0], "latent", latent,
                    )?)),
                    None => Ok(LatentSource::Decoded(decode_image_for_bucket(
                        &entry.image_path,
                        buckets[entry.bucket],
                    )?)),
                }
            })
            .collect();

        // The serial half: one entry at a time, in scan order. `zip` over the
        // window's indices consumes `sources` positionally, which is what
        // makes that order structural rather than a convention.
        for (done, source) in (start..end).zip(sources) {
            let entry = &entries[done];
            let bucket = buckets[entry.bucket];
            let paths = &cache_paths[done];
            // Reported BEFORE the work, so a consumer names the entry it is
            // waiting on. Two `stat`s summarize hit-vs-miss for display; the
            // authority is the header read (a truncated file is an error, not
            // a hit) — done for the latent in the pre-pass above and for the
            // conditioning below — so this flag is advisory display detail,
            // never control flow.
            progress(DatasetProgress {
                done,
                total,
                name: paths.file_name,
                cached: paths.latent.is_file() && paths.cond.is_file(),
            });

            // Latent: cache hit → the header the pre-pass already read; miss →
            // encode the buffer it already decoded, store it, and let the
            // tensor drop with this iteration.
            let latent_shape = match source? {
                LatentSource::Cached(shape) => shape,
                LatentSource::Decoded(data) => {
                    let image = image_tensor::<B>(data, bucket, device);
                    let latent = encode_image(image)
                        .with_context(|| format!("encoding {}", entry.image_path.display()))?;
                    let shape = latent.dims();
                    cache.write(&paths.latent, vec![("latent", to_owned_f32(latent))])?;
                    shape
                }
            };

            // Conditioning: same shape of hit/miss, two tensors per file. The
            // mask is stored as f32 0/1 (the cache is a single-dtype format
            // here) and converted back to Int on load.
            let (cond_shape, mask_shape) =
                match read_cache_shapes(&paths.cond, &["conditioning", "mask"])? {
                    Some(shapes) => (
                        ranked::<4>(&shapes[0], "conditioning", &paths.cond)?,
                        ranked::<2>(&shapes[1], "mask", &paths.cond)?,
                    ),
                    None => {
                        let (conditioning, mask) = encode_caption(&entry.caption)
                            .with_context(|| format!("encoding caption for {}", paths.stem))?;
                        let shapes = (conditioning.dims(), mask.dims());
                        cache.write(
                            &paths.cond,
                            vec![
                                ("conditioning", to_owned_f32(conditioning)),
                                ("mask", to_owned_f32(mask.float())),
                            ],
                        )?;
                        shapes
                    }
                };

            items.push(CachedItem {
                latent_path: paths.latent.clone(),
                cond_path: paths.cond.clone(),
                latent_shape,
                cond_shape,
                mask_shape,
                bucket: entry.bucket,
            });
        }
    }

    Ok(PreparedDataset {
        items,
        buckets,
        entries,
    })
}

/// Plan a run over an **already-encoded** dataset: scan, bucket, and record
/// where every example's cached tensors live.
///
/// Takes no encoder closures and no device, which is the whole point. On a
/// warm cache there is nothing to call, so "epochs never re-encode" becomes a
/// property of this **signature** rather than of closures that happen not to
/// fire — the training path structurally cannot re-encode at the training
/// precision (f16 encoders are exactly the bug the encode/train split exists
/// to prevent).
///
/// Every entry whose cache file is missing, truncated, non-F32, or the wrong
/// rank is a hard error naming the file. A miss here means the dataset
/// changed between the encode phase and now.
pub fn plan_dataset(
    config: &DatasetConfig,
    fingerprint: &str,
    mut progress: impl FnMut(DatasetProgress<'_>),
) -> Result<PreparedDataset> {
    let mut buckets = bucket_set(config)?;
    let entries = scan_dataset(&config.path, &mut buckets, config.no_upscale)?;
    let cache = DatasetCache::new(&config.path, fingerprint)?;

    let total = entries.len();
    let mut items = Vec::with_capacity(entries.len());
    for (done, entry) in entries.iter().enumerate() {
        let bucket = buckets[entry.bucket];
        let paths = entry_paths(entry, bucket, &cache)?;
        progress(DatasetProgress {
            done,
            total,
            name: paths.file_name,
            cached: paths.latent.is_file() && paths.cond.is_file(),
        });

        // The bail-on-miss contract. Wording kept recognizable from when it
        // lived in the trainer's closures — it moved from convention into
        // this signature, not away.
        let latent_shape = match read_cache_shapes(&paths.latent, &["latent"])? {
            Some(shapes) => ranked::<4>(&shapes[0], "latent", &paths.latent)?,
            None => bail!(
                "latent cache miss after the encode phase — did the dataset change mid-run? \
                 (missing {})",
                paths.latent.display()
            ),
        };
        let (cond_shape, mask_shape) =
            match read_cache_shapes(&paths.cond, &["conditioning", "mask"])? {
                Some(shapes) => (
                    ranked::<4>(&shapes[0], "conditioning", &paths.cond)?,
                    ranked::<2>(&shapes[1], "mask", &paths.cond)?,
                ),
                None => bail!(
                    "conditioning cache miss after the encode phase — did the dataset change \
                     mid-run? (missing {})",
                    paths.cond.display()
                ),
            };

        items.push(CachedItem {
            latent_path: paths.latent,
            cond_path: paths.cond,
            latent_shape,
            cond_shape,
            mask_shape,
            bucket: entry.bucket,
        });
    }

    Ok(PreparedDataset {
        items,
        buckets,
        entries,
    })
}
