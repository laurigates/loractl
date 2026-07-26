//! Throughput measurement for a real training run (#110) — the burn-side
//! adapter over the backend-agnostic [`loractl_bench`] harness.
//!
//! `loractl-bench` carries the primitives ported from CAEF: the
//! `RESULT`/`SANITY`/`MODEL` line schema, [`time_wall_sync`], the 2×-iters
//! dead-graph ratio, and [`plausible`]. This module is what makes them
//! measure *loractl* — two adapters and a work model:
//!
//! - [`device_fence`] / [`BurnOpBench`] — drive the harness from burn
//!   `Tensor`/`Autodiff` ops. The fence is [`Backend::sync`], which goes
//!   through the backend's **own** compute client; nothing here constructs a
//!   second cubecl runtime, so a measurement observes the same queues, pools
//!   and autotune cache the training run actually uses.
//! - [`StepBench`] — times a real [`Trainer`](crate::Trainer) run by observing
//!   its [`TrainEvent`] stream, and turns the run into `RESULT`/`SANITY`/
//!   `MODEL` lines.
//! - [`StepWork`] — the analytic per-step work model (tokens, FLOPs) that
//!   `tok_s=` and `tflops=` are quotients of, emitted alongside them so the
//!   denominator is auditable rather than asserted.
//!
//! ## Why observing the event stream is a device-resident measurement
//!
//! [`time_wall_sync`] fences a batch of work between two full device syncs
//! because cubecl's own profiled window only spans the first pass
//! (cubecl#1421). A training step cannot be batched that way — steps are not
//! interchangeable, each one mutates the adapters — so [`StepBench`] uses the
//! fence the training loop **already contains**, once per step, structurally:
//!
//! > A [`TrainEvent::Step`] carries `loss: f32`, a **host** scalar. Producing
//! > it requires reading a device tensor back to the host, which drains the
//! > device queue for everything submitted before it. A `Step` event therefore
//! > cannot be emitted without a device fence having just happened.
//!
//! So the window between two consecutive `Step` events is bounded by a drain
//! at each end, and holds exactly one step's worth of compute — phase-shifted
//! by where in the step the loss readback sits. In [`DiffusionTrainer`](crate::DiffusionTrainer)'s
//! monolithic arm the readback is at the end of the forward, so a window is
//! `backward(n) + optimizer(n) + forward(n+1)`; in the block-checkpointed arm
//! (#134) `checkpointed_step` returns a host `f32` after the backward sweep,
//! so a window is `optimizer(n) + step(n+1)`. Different phases, same content:
//! one whole step, fenced at both ends.
//!
//! Three consequences are handled rather than hidden. The first `Step` has no
//! predecessor, so the model-load/dataset-encode window is never a sample. A
//! window containing a checkpoint or sample export also contains disk I/O;
//! those samples are marked `ckpt=1` and excluded from the aggregate. And the
//! observer's own cost — chiefly the `nvidia-smi` subprocess behind
//! [`resident_vram_mib`] — is kept out of both adjacent windows by re-stamping
//! the window start *after* the sample is built, so it cannot bias the very
//! number it annotates.
//!
//! That last one bounds the bias but does not remove the pause: a default
//! [`StepBench`] still spawns `nvidia-smi` at every step boundary, so a long
//! run is not quite back-to-back training and the GPU idles briefly between
//! steps. It is outside the reported windows, so it cannot inflate `ms=`, but
//! if a real dispatch ever shows step times drifting across a run (clock or
//! thermal behaviour rather than compute), the knob is
//! [`with_vram_source`](StepBench::with_vram_source) — sample every N steps, or
//! poll from a watcher thread the way `examples/step_probe.rs` does.
//!
//! ## What this module does NOT do
//!
//! It does not render. [`StepBench`] accumulates [`StepSample`]s and hands back
//! `RESULT`/`SANITY`/`MODEL` values; the caller prints them
//! (`examples/bench_step.rs` is the reference driver). That is core's
//! load-bearing invariant, and it is what lets the same measurement feed a
//! terminal, a CI log, or an HTTP surface unchanged.

use crate::event::TrainEvent;
use crate::mmdit::MmditConfig;
use burn::tensor::backend::{Backend, ExecutionError};
use loractl_bench::{BenchResult, ModelLine, Sanity, plausible, time_wall_sync};
use std::time::{Duration, Instant};

/// Drain the backend's device queue — the fence every timed region needs.
///
/// This is [`Backend::sync`], which dispatches to the backend's existing
/// compute client (for the cubecl backends, the very one the training run's
/// tensors live on). Deliberately *not* a freshly initialized runtime: a
/// second client would have its own queues, memory pool and autotune cache,
/// and fencing it would prove nothing about the work being timed.
///
/// On the `ndarray` backend this is burn's default no-op — CPU execution is
/// already synchronous, so the fence is trivially satisfied.
pub fn device_fence<B: Backend>(device: &B::Device) -> Result<(), ExecutionError> {
    B::sync(device)
}

/// Times a repeatable burn `Tensor`/`Autodiff` op through the harness'
/// wall-sync timer, with the 2×-iters dead-graph check.
///
/// For work that *can* be replayed identically (a forward pass, a matmul, a
/// dequant) — unlike a training step, which mutates state. `run` measures
/// `iters` fenced iterations, then `2·iters`, and reports both the per-iter
/// time and the [`Sanity`] ratio between the two totals: work that got elided
/// or cached does not cost twice as much when asked for twice.
///
/// `iters` must be non-zero: [`run`](Self::run) panics on zero rather than
/// returning `Err`, since dividing by no iterations is a caller bug, not a
/// device failure.
#[derive(Debug, Clone, Copy)]
pub struct BurnOpBench {
    /// Timed iterations in the first (reported) measurement. Must be > 0.
    pub iters: u32,
    /// Unmeasured iterations first, absorbing shader compilation and autotune.
    pub warmup: u32,
}

impl Default for BurnOpBench {
    fn default() -> Self {
        Self {
            iters: 10,
            warmup: 3,
        }
    }
}

impl BurnOpBench {
    /// Measure `work` on `device`, returning a `RESULT`-shaped
    /// [`BenchResult`] carrying the per-iteration time and the 2×-iters
    /// verdict.
    ///
    /// `work` must submit the op and nothing else — no host readback, which
    /// would fence inside the timed region and serialize the batch.
    pub fn run<B: Backend>(
        &self,
        label: impl Into<String>,
        device: &B::Device,
        mut work: impl FnMut(),
    ) -> Result<BenchResult, ExecutionError> {
        // Surface a fence failure instead of swallowing it: a device that
        // failed to sync has not finished the work we are about to divide by,
        // so the measurement is void rather than merely noisy. The first error
        // wins — later fences on a broken device say nothing new.
        let mut failure: Option<ExecutionError> = None;

        let avg_n = time_wall_sync(self.iters, self.warmup, &mut work, || {
            if failure.is_none()
                && let Err(e) = device_fence::<B>(device)
            {
                failure = Some(e);
            }
        });
        if let Some(e) = failure {
            return Err(e);
        }
        let avg_2n = time_wall_sync(2 * self.iters, 0, &mut work, || {
            if failure.is_none()
                && let Err(e) = device_fence::<B>(device)
            {
                failure = Some(e);
            }
        });
        if let Some(e) = failure {
            return Err(e);
        }

        Ok(BenchResult::new(label, avg_n).with_sanity(Sanity::from_avgs(avg_n, avg_2n)))
    }
}

/// The analytic work behind one training step — the denominator of `tok_s=`
/// and `tflops=`.
///
/// Both figures are quotients of a *model*, never of anything the run
/// reports, so the model travels with them: [`StepBench::model_line`] prints
/// its terms and the `excludes=` list. Time and VRAM are measured; these are
/// derived.
#[derive(Debug, Clone, Default)]
pub struct StepWork {
    /// Tokens the trunk processes per step (`batch × seq_len`). `None`
    /// suppresses `tok_s=`.
    pub tokens: Option<u64>,
    /// Modelled FLOPs per step, forward **and** backward. `None` suppresses
    /// `tflops=`.
    pub flops: Option<f64>,
    /// Terms of the model, for the `MODEL` line.
    pub terms: Vec<(String, String)>,
}

impl StepWork {
    /// A model that claims nothing: `RESULT` lines carry time and VRAM only.
    ///
    /// The honest default for a run whose token geometry is not known to the
    /// caller — `ms=` and `vram_mib=` are measured and need no model at all.
    pub fn unmodelled() -> Self {
        Self::default()
    }

    /// Declare the per-step token count (`batch × seq_len`), enabling `tok_s=`.
    pub fn with_tokens(mut self, tokens: u64) -> Self {
        self.tokens = Some(tokens);
        self.terms
            .push(("tokens_per_step".into(), tokens.to_string()));
        self
    }

    /// Declare modelled per-step FLOPs (forward + backward), enabling
    /// `tflops=`.
    pub fn with_flops(mut self, flops: f64) -> Self {
        self.flops = Some(flops);
        self
    }

    /// Attach a term to the `MODEL` line.
    pub fn with_term(mut self, key: impl Into<String>, value: impl std::fmt::Display) -> Self {
        self.terms.push((key.into(), value.to_string()));
        self
    }

    /// The per-step work model for an MMDiT LoRA training step: `batch`
    /// sequences of `seq_len` tokens through `cfg.layers` trunk blocks.
    ///
    /// **Counted** — the trunk blocks' dense projections, at `2·d_in·d_out`
    /// FLOPs per token each (`attn.{wq,wk,wv,gate,wo}`, `mlp.{gate,up,down}`)
    /// — and the two attention matmuls, `4·seq_len²·features` per block per
    /// sequence (`QKᵀ` and `P·V`; GQA repeats the KV heads, so the score
    /// matmuls are query-head wide either way).
    ///
    /// **Excluded** — the 2+2 text-fusion blocks (text tokens only, ~2% of
    /// trunk work at Krea 2's widths), the per-sample modulation projection,
    /// norms, RoPE, softmax, the patch embed/unembed, and the LoRA deltas
    /// themselves (rank ≪ `features`). Everything excluded makes the model an
    /// **under**-count, so the reported `tflops=` is a floor on achieved
    /// throughput, not a ceiling.
    ///
    /// **Backward multiplier** — a LoRA step backpropagates through a frozen
    /// base: burn computes an input gradient for each frozen projection but no
    /// weight gradient, so the backward costs about one forward, not two.
    /// `grad_checkpointing` (#134) replays the trunk forward once more in the
    /// backward sweep, adding another. Hence `×2`, or `×3` when checkpointing.
    ///
    /// The VAE and text encoder are absent by construction, not by omission:
    /// the M12 dataset pipeline caches latents and conditioning, so a step
    /// runs the denoiser alone.
    pub fn mmdit_step(
        cfg: &MmditConfig,
        batch: usize,
        seq_len: usize,
        grad_checkpointing: bool,
    ) -> Self {
        let f = cfg.features;
        let kv_out = cfg.head_dim() * cfg.kvheads;
        let inner = MmditConfig::swiglu_dim(f, cfg.multiplier);

        // Per token, per block: 2·d_in·d_out over the dense projections.
        let proj_macs_per_token = (f * f) * 3 + (f * kv_out) * 2 + (f * inner) * 3;
        let proj = 2.0 * proj_macs_per_token as f64 * (seq_len * batch * cfg.layers) as f64;
        // Per sequence, per block: QKᵀ then P·V, both [seq, seq] × width f.
        let attn = 4.0 * (seq_len * seq_len) as f64 * f as f64 * (batch * cfg.layers) as f64;

        let fwd = proj + attn;
        let passes = if grad_checkpointing { 3.0 } else { 2.0 };

        Self::unmodelled()
            .with_tokens((batch * seq_len) as u64)
            .with_flops(fwd * passes)
            .with_term("batch", batch)
            .with_term("seq_len", seq_len)
            .with_term("layers", cfg.layers)
            .with_term("features", f)
            .with_term("fwd_flops", fwd)
            .with_term("passes", passes)
            .with_term("grad_ckpt", grad_checkpointing)
            .with_term(
                "excludes",
                "text_fusion,modulation,norms,rope,softmax,patch_embed,lora_delta",
            )
    }
}

/// One timed step: the fenced window between two consecutive
/// [`TrainEvent::Step`]s.
#[derive(Debug, Clone, Copy)]
pub struct StepSample {
    /// The step whose `Step` event closed this window.
    pub step: u64,
    /// Wall time of the window — one whole step's compute (see the module
    /// docs for the phase shift).
    pub window: Duration,
    /// Loss reported at the closing `Step` event.
    pub loss: f32,
    /// Resident VRAM at the closing `Step` event, when telemetry is available.
    pub vram_mib: Option<u64>,
    /// Whether a checkpoint or sample export landed inside this window — its
    /// disk I/O is not step compute, so the sample is excluded from the
    /// aggregate.
    pub contaminated: bool,
}

/// Times a real training run by observing its [`TrainEvent`] stream.
///
/// Feed every event to [`record`](Self::record) from inside the caller's sink;
/// afterwards read [`samples`](Self::samples) or the ready-made
/// [`result_lines`](Self::result_lines). Nothing is printed — the caller
/// renders.
pub struct StepBench {
    work: StepWork,
    label: String,
    vram: Box<dyn FnMut() -> Option<u64> + Send>,
    samples: Vec<StepSample>,
    last: Option<Instant>,
    contaminated: bool,
    warmup_steps: usize,
}

impl StepBench {
    /// A bench that labels its lines `label` and derives throughputs from
    /// `work`, reading VRAM via [`resident_vram_mib`].
    pub fn new(label: impl Into<String>, work: StepWork) -> Self {
        Self {
            work,
            label: label.into(),
            vram: Box::new(resident_vram_mib),
            samples: Vec::new(),
            last: None,
            contaminated: false,
            // The first timed window still pays one-time costs the steady
            // state does not — a cold autotune cache, lazily materialized
            // params, the allocator growing into its working set.
            warmup_steps: 1,
        }
    }

    /// Replace the VRAM reader (a test injects a deterministic one; a non-CUDA
    /// host can supply its own telemetry).
    pub fn with_vram_source(
        mut self,
        source: impl FnMut() -> Option<u64> + Send + 'static,
    ) -> Self {
        self.vram = Box::new(source);
        self
    }

    /// How many leading timed windows to treat as warm-up and exclude from
    /// the aggregate (default 1).
    pub fn with_warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Observe one event. Call this for every event the trainer emits.
    pub fn record(&mut self, event: &TrainEvent) {
        match event {
            TrainEvent::Started { .. } => {
                self.samples.clear();
                self.last = None;
                self.contaminated = false;
            }
            TrainEvent::Step { step, loss, .. } => {
                let now = Instant::now();
                // No sample for the first Step: the window before it is model
                // load and dataset encode, not a training step.
                if let Some(prev) = self.last {
                    self.samples.push(StepSample {
                        step: *step,
                        window: now.duration_since(prev),
                        loss: *loss,
                        vram_mib: (self.vram)(),
                        contaminated: self.contaminated,
                    });
                }
                // Re-stamp AFTER the sample is built, not from `now`. `record`
                // runs synchronously inside the trainer's sink, and the VRAM
                // read above shells out to `nvidia-smi` — tens of ms. Starting
                // the next window at `now` would bury that subprocess inside
                // it, biasing every counted window upward by the same amount:
                // invisible to the 2×-steps ratio (both halves shift equally),
                // absent on a host without `nvidia-smi` (where it is tested),
                // and present on the GPU host (where it is quoted). The cost of
                // observing is now outside both windows, so a window slightly
                // *under*-reports the step-to-step interval rather than
                // over-reporting the step.
                self.last = Some(Instant::now());
                self.contaminated = false;
            }
            // Disk I/O inside the window — the export is real work, but it is
            // not the step time anyone is asking about.
            TrainEvent::Checkpoint { .. } | TrainEvent::Sample { .. } => self.contaminated = true,
            TrainEvent::Warning { .. } | TrainEvent::Finished { .. } => {}
        }
    }

    /// Every timed window, in order.
    pub fn samples(&self) -> &[StepSample] {
        &self.samples
    }

    /// Whether the sample at `index` feeds the aggregate.
    ///
    /// The single spelling of the rule: [`counted`](Self::counted) selects with
    /// it and [`step_lines`](Self::step_lines) annotates with it, so a
    /// `counted=1` can never appear on a line the aggregate actually dropped —
    /// which is the exact discrepancy the annotation exists to expose.
    fn is_counted(&self, index: usize, sample: &StepSample) -> bool {
        index >= self.warmup_steps && !sample.contaminated
    }

    /// The samples the aggregate is computed from: warm-up dropped, windows
    /// contaminated by a checkpoint export dropped.
    pub fn counted(&self) -> Vec<&StepSample> {
        self.samples
            .iter()
            .enumerate()
            .filter(|(index, sample)| self.is_counted(*index, sample))
            .map(|(_, sample)| sample)
            .collect()
    }

    /// The dead-graph guard over the run's losses.
    ///
    /// [`plausible`] is CAEF's check that a kernel produced *something*; the
    /// training analogue is that the run produced a real loss — finite (no
    /// device-thread panic) and not identically zero (no elided graph). A
    /// `false` here voids the timings above it: a step that computed nothing
    /// is fast for reasons nobody wants.
    pub fn losses_plausible(&self) -> bool {
        let losses: Vec<f32> = self.samples.iter().map(|s| s.loss).collect();
        plausible(&losses)
    }

    /// The 2×-steps scaling verdict.
    ///
    /// The same dead-graph question [`Sanity`] asks of a replayed kernel, put
    /// to a run that cannot replay anything: the first half of the counted
    /// steps and all of them are compared as if they were an `N` and a `2·N`
    /// measurement. Stable per-step cost ⇒ ratio ≈ 2. A run whose later steps
    /// got cheap — work elided, a graph gone dead, thermal-free caching —
    /// falls out of the band and reads `SUSPECT`.
    ///
    /// `None` with fewer than two counted steps, where the question is empty.
    pub fn sanity(&self) -> Option<Sanity> {
        let counted = self.counted();
        if counted.len() < 2 {
            return None;
        }
        let half = counted.len() / 2;
        let mean = |slice: &[&StepSample]| {
            slice.iter().map(|s| s.window).sum::<Duration>() / slice.len() as u32
        };
        Some(Sanity::from_avgs(mean(&counted[..half]), mean(&counted)))
    }

    /// Median per-step window over [`counted`](Self::counted).
    ///
    /// Median, not mean: a training run's outliers are one-sided (an OS
    /// scheduling hiccup, a page fault, an allocator growth) and a mean would
    /// carry them into the steady-state number.
    ///
    /// The **upper** median on an even count — element `len/2` of the sorted
    /// windows, not the average of the two middle ones. Deliberate: it keeps
    /// the reported figure an actually-observed step time rather than a
    /// synthesized one, and it errs slow.
    pub fn median_step(&self) -> Option<Duration> {
        let mut windows: Vec<Duration> = self.counted().iter().map(|s| s.window).collect();
        if windows.is_empty() {
            return None;
        }
        windows.sort_unstable();
        Some(windows[windows.len() / 2])
    }

    /// Peak resident VRAM seen at any step boundary.
    pub fn peak_vram_mib(&self) -> Option<u64> {
        self.samples.iter().filter_map(|s| s.vram_mib).max()
    }

    /// A per-step `RESULT` line for every timed window.
    ///
    /// Each line says whether it fed the aggregate (`counted=`) and, if not,
    /// why — `ckpt=1` for a checkpoint export, otherwise warm-up. Without that
    /// a reader who greps the per-step lines and averages them gets a
    /// different number from the `_median` line and no way to see the
    /// discrepancy.
    pub fn step_lines(&self) -> Vec<BenchResult> {
        self.samples
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                let counted = self.is_counted(index, sample);
                let line = self
                    .throughputs(
                        BenchResult::new(self.label.clone(), sample.window),
                        sample.window,
                    )
                    .with("step", sample.step)
                    .with("loss", format!("{:.6}", sample.loss))
                    .with("ckpt", u8::from(sample.contaminated))
                    .with("counted", u8::from(counted));
                match sample.vram_mib {
                    Some(mib) => line.with("vram_mib", mib),
                    None => line,
                }
            })
            .collect()
    }

    /// The run's aggregate `RESULT` line — the one a reader should quote.
    ///
    /// `None` when no window survived warm-up and contamination filtering.
    pub fn summary_line(&self) -> Option<BenchResult> {
        let median = self.median_step()?;
        let counted = self.counted().len();
        let line = self
            .throughputs(
                BenchResult::new(format!("{}_median", self.label), median),
                median,
            )
            .with("steps_counted", counted)
            .with("steps_timed", self.samples.len())
            .with("plausible", self.losses_plausible());
        let line = match self.peak_vram_mib() {
            Some(mib) => line.with("vram_peak_mib", mib),
            None => line,
        };
        Some(match self.sanity() {
            Some(sanity) => line.with_sanity(sanity),
            None => line,
        })
    }

    /// The `MODEL` line for the work model behind `tok_s=` / `tflops=`.
    ///
    /// `None` when the model claims nothing ([`StepWork::unmodelled`]) — there
    /// is no denominator to audit.
    pub fn model_line(&self) -> Option<ModelLine> {
        if self.work.terms.is_empty() {
            return None;
        }
        let mut line = ModelLine::new(self.label.clone());
        for (key, value) in &self.work.terms {
            line = line.with(key, value);
        }
        Some(line)
    }

    /// Every line the run produced, rendered in emission order: the `MODEL`
    /// line, one `RESULT` per timed step, the aggregate `RESULT`, and the
    /// `SANITY` verdict.
    pub fn result_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if let Some(model) = self.model_line() {
            lines.push(model.to_string());
        }
        lines.extend(self.step_lines().iter().map(ToString::to_string));
        if let Some(summary) = self.summary_line() {
            lines.push(summary.to_string());
        }
        if let Some(sanity) = self.sanity() {
            lines.push(sanity.to_string());
        }
        lines
    }

    /// Attach the derived `tok_s=` / `tflops=` figures for a window.
    fn throughputs(&self, line: BenchResult, window: Duration) -> BenchResult {
        let secs = window.as_secs_f64();
        if secs <= 0.0 {
            return line;
        }
        let line = match self.work.tokens {
            Some(tokens) => line.with_throughput(tokens as f64 / secs, "tok_s"),
            None => line,
        };
        match self.work.flops {
            Some(flops) => line.with("tflops", format!("{:.4}", flops / secs / 1e12)),
            None => line,
        }
    }
}

impl std::fmt::Debug for StepBench {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepBench")
            .field("label", &self.label)
            .field("work", &self.work)
            .field("samples", &self.samples.len())
            .finish_non_exhaustive()
    }
}

/// Resident VRAM (MiB) on CUDA device `0`, via `nvidia-smi`.
///
/// Best-effort telemetry: a host without the tool degrades to `None` (the
/// `vram_mib=` term is simply absent), never to a failure. Deliberately the
/// driver's own accounting rather than the allocator's — an allocator reports
/// what it believes it holds, while the question a 24 GB card poses is what
/// the *driver* has committed.
pub fn resident_vram_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Tensor;
    use std::path::PathBuf;

    type B = NdArray<f32>;

    /// Build a bench over a scripted event stream, sleeping `gap` between
    /// steps so the windows are real (and ordered) without being slow.
    fn run_scripted(events: &[TrainEvent], gap: Duration) -> StepBench {
        let mut bench = StepBench::new("step", StepWork::unmodelled())
            .with_vram_source(|| Some(1234))
            .with_warmup_steps(0);
        for event in events {
            if matches!(event, TrainEvent::Step { .. }) {
                std::thread::sleep(gap);
            }
            bench.record(event);
        }
        bench
    }

    fn step(step: u64, loss: f32) -> TrainEvent {
        TrainEvent::Step {
            step,
            loss,
            lr: 1e-4,
        }
    }

    #[test]
    fn first_step_opens_a_window_but_is_not_a_sample() {
        let bench = run_scripted(
            &[
                TrainEvent::Started { total_steps: 3 },
                step(1, 0.9),
                step(2, 0.8),
                step(3, 0.7),
            ],
            Duration::from_millis(2),
        );
        // Three Step events, two inter-step windows — the load/encode window
        // before step 1 is never a measurement.
        assert_eq!(bench.samples().len(), 2);
        assert_eq!(bench.samples()[0].step, 2);
        assert_eq!(bench.samples()[1].step, 3);
        assert!(bench.samples().iter().all(|s| s.window > Duration::ZERO));
    }

    #[test]
    fn the_vram_read_does_not_land_inside_a_timed_window() {
        // The regression guard for the observer-overhead bias: on the GPU host
        // `resident_vram_mib` shells out to `nvidia-smi` for tens of ms, and
        // `record` runs synchronously inside the trainer's sink. If the next
        // window started before that read, every counted window would carry it
        // — uniformly, so the 2×-steps ratio could not see it, and only on the
        // machine whose numbers get quoted.
        //
        // A deliberately slow VRAM source stands in for `nvidia-smi`. The
        // windows here contain nothing else, so absorbing the read would be
        // unmistakable: they would each be ~the read's cost instead of ~0.
        const READ: Duration = Duration::from_millis(60);
        let mut bench = StepBench::new("step", StepWork::unmodelled())
            .with_vram_source(|| {
                std::thread::sleep(READ);
                Some(4096)
            })
            .with_warmup_steps(0);
        for event in [
            TrainEvent::Started { total_steps: 4 },
            step(1, 0.9),
            step(2, 0.8),
            step(3, 0.7),
            step(4, 0.6),
        ] {
            bench.record(&event);
        }

        assert_eq!(bench.samples().len(), 3);
        for sample in bench.samples() {
            assert!(
                sample.window < READ / 2,
                "step {}'s window ({:?}) absorbed the {READ:?} VRAM read",
                sample.step,
                sample.window,
            );
        }
        // The read still happened — this fixes where its cost is accounted,
        // not whether the telemetry is collected.
        assert_eq!(bench.peak_vram_mib(), Some(4096));
    }

    #[test]
    fn checkpoint_windows_are_marked_and_excluded() {
        let bench = run_scripted(
            &[
                TrainEvent::Started { total_steps: 4 },
                step(1, 0.9),
                step(2, 0.8),
                TrainEvent::Checkpoint {
                    step: 2,
                    path: PathBuf::from("ckpt.safetensors"),
                },
                step(3, 0.7),
                step(4, 0.6),
            ],
            Duration::from_millis(2),
        );
        let contaminated: Vec<u64> = bench
            .samples()
            .iter()
            .filter(|s| s.contaminated)
            .map(|s| s.step)
            .collect();
        // The export sat between step 2 and step 3, so step 3's window carries
        // its disk I/O — and only that one.
        assert_eq!(contaminated, vec![3]);
        assert_eq!(bench.counted().len(), 2);
        assert!(bench.counted().iter().all(|s| !s.contaminated));

        // The other half of the annotation: a contaminated window is dropped
        // for a different reason than warm-up, and must still read counted=0.
        assert_counted_terms_agree(&bench);
        let contaminated_line = bench
            .step_lines()
            .iter()
            .map(ToString::to_string)
            .find(|l| l.contains("step=3 "))
            .expect("step 3's line");
        assert!(
            contaminated_line.contains("ckpt=1") && contaminated_line.contains("counted=0"),
            "{contaminated_line}"
        );
    }

    #[test]
    fn warmup_windows_are_dropped_from_the_aggregate() {
        let events = [
            TrainEvent::Started { total_steps: 4 },
            step(1, 0.9),
            step(2, 0.8),
            step(3, 0.7),
            step(4, 0.6),
        ];
        let mut bench = StepBench::new("step", StepWork::unmodelled()).with_warmup_steps(1);
        for event in &events {
            std::thread::sleep(Duration::from_millis(1));
            bench.record(event);
        }
        assert_eq!(bench.samples().len(), 3);
        assert_eq!(bench.counted().len(), 2, "the first window is warm-up");
        assert_eq!(bench.counted()[0].step, 3);

        // Tie the printed annotation to the selection it claims to describe.
        // `counted=` exists so a reader can reconcile the per-step lines with
        // the `_median` aggregate; without this the term could be inverted, or
        // lose its warm-up clause, and the whole suite would still pass —
        // leaving exactly the discrepancy it was added to expose.
        assert_counted_terms_agree(&bench);
        assert!(
            bench.step_lines()[0].to_string().contains("counted=0"),
            "the warm-up window must be marked as not counted"
        );
    }

    /// Assert every `counted=` term matches [`StepBench::counted`]'s selection.
    fn assert_counted_terms_agree(bench: &StepBench) {
        let lines: Vec<String> = bench.step_lines().iter().map(ToString::to_string).collect();
        let marked: Vec<&String> = lines.iter().filter(|l| l.contains("counted=1")).collect();
        assert_eq!(
            marked.len(),
            bench.counted().len(),
            "counted=1 lines must equal the aggregate's sample count\n{lines:#?}"
        );
        // Every line carries the term, so `counted=1` + `counted=0` accounts
        // for all of them — an omitted term cannot hide as a `counted=0`.
        assert_eq!(
            lines.iter().filter(|l| l.contains("counted=0")).count(),
            lines.len() - marked.len(),
            "every line must carry a counted= term\n{lines:#?}"
        );
    }

    #[test]
    fn started_resets_a_reused_bench() {
        let mut bench = run_scripted(
            &[
                TrainEvent::Started { total_steps: 2 },
                step(1, 0.5),
                step(2, 0.4),
            ],
            Duration::from_millis(1),
        );
        assert_eq!(bench.samples().len(), 1);
        bench.record(&TrainEvent::Started { total_steps: 2 });
        assert!(bench.samples().is_empty());
        assert!(bench.summary_line().is_none(), "nothing measured yet");
    }

    #[test]
    fn dead_graph_losses_are_rejected() {
        let dead = run_scripted(
            &[
                TrainEvent::Started { total_steps: 3 },
                step(1, 0.0),
                step(2, 0.0),
                step(3, 0.0),
            ],
            Duration::from_millis(1),
        );
        assert!(
            !dead.losses_plausible(),
            "an all-zero loss stream is an elided graph, not a fast run"
        );
        let nan = run_scripted(
            &[
                TrainEvent::Started { total_steps: 2 },
                step(1, 0.5),
                step(2, f32::NAN),
            ],
            Duration::from_millis(1),
        );
        assert!(!nan.losses_plausible(), "NaN is a device panic");
        let live = run_scripted(
            &[
                TrainEvent::Started { total_steps: 2 },
                step(1, 0.5),
                step(2, 0.4),
            ],
            Duration::from_millis(1),
        );
        assert!(live.losses_plausible());
    }

    #[test]
    fn summary_reports_median_peak_vram_and_sanity() {
        let mut vram = [4096u64, 8192, 6144].into_iter();
        let mut bench = StepBench::new("train_step", StepWork::unmodelled())
            .with_vram_source(move || vram.next())
            .with_warmup_steps(0);
        for event in [
            TrainEvent::Started { total_steps: 4 },
            step(1, 0.9),
            step(2, 0.8),
            step(3, 0.7),
            step(4, 0.6),
        ] {
            std::thread::sleep(Duration::from_millis(2));
            bench.record(&event);
        }
        assert_eq!(bench.peak_vram_mib(), Some(8192));
        let line = bench
            .summary_line()
            .expect("three windows were timed")
            .to_string();
        assert!(
            line.starts_with("RESULT label=train_step_median ms="),
            "{line}"
        );
        assert!(line.contains("steps_counted=3"), "{line}");
        assert!(line.contains("plausible=true"), "{line}");
        assert!(line.contains("vram_peak_mib=8192"), "{line}");
        // That a verdict is *present* is structural and safe to assert here.
        // Which verdict is a wall-clock claim about `sleep(2ms)` windows on
        // whatever machine runs this — jitter on a loaded runner is the same
        // order as the signal, so asserting `ok` here would be a flake, not a
        // check. The `ok` direction is pinned deterministically instead, in
        // `sanity_ok_on_stable_synthetic_windows`.
        assert!(line.contains("sanity="), "{line}");
        // No work model was declared, so no throughput is claimed.
        assert!(!line.contains("tok_s="), "{line}");
        assert!(!line.contains("tflops="), "{line}");
    }

    /// Build a bench holding hand-written windows — pure arithmetic, no
    /// wall clock, so the verdict assertions below cannot flake.
    fn bench_with_windows(windows: &[Duration]) -> StepBench {
        let mut bench = StepBench::new("step", StepWork::unmodelled()).with_warmup_steps(0);
        bench.samples = windows
            .iter()
            .enumerate()
            .map(|(i, window)| StepSample {
                step: i as u64 + 2,
                window: *window,
                loss: 0.5,
                vram_mib: None,
                contaminated: false,
            })
            .collect();
        bench
    }

    #[test]
    fn sanity_ok_on_stable_synthetic_windows() {
        // The `ok` direction, pinned without a clock: equal per-step cost means
        // the first-half mean equals the overall mean, so the ratio is exactly
        // 2. Its wall-clock counterpart lives in
        // `summary_reports_median_peak_vram_and_sanity`, which deliberately
        // asserts only that a verdict is present.
        let bench = bench_with_windows(&[Duration::from_millis(10); 4]);
        let sanity = bench.sanity().expect("four counted windows");
        assert!((sanity.ratio - 2.0).abs() < 1e-9, "ratio {}", sanity.ratio);
        assert!(sanity.ok);
        assert!(
            bench
                .summary_line()
                .unwrap()
                .to_string()
                .contains("sanity=ok")
        );

        // And still `ok` when the two halves genuinely differ, so the band is
        // shown to have room rather than only being hit dead centre. The
        // sequence is deliberately unbalanced — first half means 9.667 ms
        // against 10 ms overall, ratio ≈ 2.069 — because a *balanced* jitter
        // sequence averages back to exactly 2.0 and would re-assert the
        // arithmetic above instead of the tolerance.
        let jittered = [9, 11, 9, 11, 10, 10].map(Duration::from_millis);
        let sanity = bench_with_windows(&jittered).sanity().unwrap();
        assert!(
            (sanity.ratio - 2.0).abs() > 0.05,
            "the jitter case must actually leave 2.0 (got {})",
            sanity.ratio
        );
        assert!(
            sanity.ok,
            "±10% jitter must not trip the band: {}",
            sanity.ratio
        );
    }

    #[test]
    fn sanity_flags_a_run_whose_later_steps_went_free() {
        // A hand-built sample set: the first half costs 10 ms/step, the second
        // half ~0 — the shape an elided graph produces.
        let mut bench = StepBench::new("step", StepWork::unmodelled()).with_warmup_steps(0);
        bench.samples = (1..=4)
            .map(|step| StepSample {
                step,
                window: if step <= 2 {
                    Duration::from_millis(10)
                } else {
                    Duration::from_micros(10)
                },
                loss: 0.5,
                vram_mib: None,
                contaminated: false,
            })
            .collect();
        let sanity = bench.sanity().expect("four counted windows");
        assert!(
            !sanity.ok,
            "half the steps costing nothing must not read ok (ratio {})",
            sanity.ratio
        );
        assert_eq!(sanity.verdict(), "SUSPECT");
    }

    #[test]
    fn throughputs_appear_only_with_a_work_model() {
        let work = StepWork::unmodelled().with_tokens(2048).with_flops(1e12);
        let mut bench = StepBench::new("step", work).with_warmup_steps(0);
        bench.samples = vec![StepSample {
            step: 2,
            // 1 s exactly, so the quotients are readable by inspection.
            window: Duration::from_secs(1),
            loss: 0.5,
            vram_mib: None,
            contaminated: false,
        }];
        let line = bench.step_lines()[0].to_string();
        assert!(line.contains("tok_s=2048.0000"), "{line}");
        assert!(line.contains("tflops=1.0000"), "{line}");
        assert!(line.contains("step=2 loss=0.500000 ckpt=0"), "{line}");
        let model = bench
            .model_line()
            .expect("tokens declared a term")
            .to_string();
        assert!(
            model.contains("MODEL label=step tokens_per_step=2048"),
            "{model}"
        );
    }

    /// Re-derives the count term by term, naming each projection explicitly
    /// rather than reusing the implementation's grouping.
    ///
    /// What this pins is the *arithmetic*: a term dropped, double-counted, or
    /// regrouped wrongly fails here. What it does **not** pin is the
    /// architectural premise — it calls the same `swiglu_dim`/`head_dim` the
    /// implementation does, so a wrong width would pass both sides. That half
    /// is pinned by the mmdit parity suite, which compares against the official
    /// `mmdit.py`; this test deliberately does not restate those widths, since
    /// a hand-copied constant here would be the thing most likely to go stale.
    #[test]
    fn mmdit_work_model_matches_a_hand_derivation() {
        let cfg = MmditConfig::tiny();
        let (batch, seq) = (2usize, 8usize);
        let work = StepWork::mmdit_step(&cfg, batch, seq, false);
        assert_eq!(work.tokens, Some((batch * seq) as u64));

        // Re-derive independently of the implementation's grouping.
        let f = cfg.features;
        let kv_out = cfg.head_dim() * cfg.kvheads;
        let inner = MmditConfig::swiglu_dim(f, cfg.multiplier);
        let per_token: usize = [
            f * f,      // attn.wq
            f * kv_out, // attn.wk
            f * kv_out, // attn.wv
            f * f,      // attn.gate
            f * f,      // attn.wo
            f * inner,  // mlp.gate
            f * inner,  // mlp.up
            inner * f,  // mlp.down
        ]
        .iter()
        .sum();
        let fwd = 2.0 * per_token as f64 * (seq * batch * cfg.layers) as f64
            + 4.0 * (seq * seq * f * batch * cfg.layers) as f64;
        let flops = work.flops.expect("mmdit_step always models flops");
        assert!(
            (flops - fwd * 2.0).abs() < 1.0,
            "frozen-base backward costs one forward: {flops} vs {}",
            fwd * 2.0
        );

        // Checkpointing replays the trunk forward once more per step.
        let ckpt = StepWork::mmdit_step(&cfg, batch, seq, true).flops.unwrap();
        assert!(
            (ckpt - fwd * 3.0).abs() < 1.0,
            "checkpointing adds a replayed forward: {ckpt} vs {}",
            fwd * 3.0
        );
    }

    #[test]
    fn mmdit_work_model_scales_quadratically_in_sequence() {
        // Attention is the only super-linear term, so doubling the sequence
        // must more than double the modelled work — the property that makes
        // `tflops=` comparable across resolutions at all.
        let cfg = MmditConfig::krea2();
        let short = StepWork::mmdit_step(&cfg, 1, 1024, false).flops.unwrap();
        let long = StepWork::mmdit_step(&cfg, 1, 2048, false).flops.unwrap();
        assert!(long > 2.0 * short, "{long} vs {short}");
        assert!(
            long < 4.0 * short,
            "projections stay linear: {long} vs {short}"
        );
    }

    #[test]
    fn burn_op_bench_times_a_real_tensor_op() {
        let device = Default::default();
        let a = Tensor::<B, 2>::ones([64, 64], &device);
        let bench = BurnOpBench {
            iters: 4,
            warmup: 1,
        };
        let result = bench
            .run::<B>("matmul", &device, || {
                let _ = a.clone().matmul(a.clone());
            })
            .expect("ndarray sync cannot fail");
        assert_eq!(result.label, "matmul");
        assert!(result.ms > 0.0, "a real matmul takes real time: {result}");
        assert!(result.sanity.is_some(), "the 2x-iters check always runs");
    }

    #[test]
    fn device_fence_is_available_on_the_cpu_backend() {
        // The ndarray backend takes burn's default no-op `sync` — CPU work is
        // already complete when the call returns. Pinned so the fence stays
        // callable on the always-compiled backend the offline suite uses.
        assert!(device_fence::<B>(&Default::default()).is_ok());
    }
}
