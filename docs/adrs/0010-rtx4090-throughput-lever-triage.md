---
id: ADR-0010
status: Accepted
date: 2026-07-31
---

# 0010 — RTX 4090 throughput: the premise is unmeasured, and three of the four proposed levers are already shipped or dead by construction

- **Status:** Accepted
- **Date:** 2026-07-31
- **Milestones:** post-M15 throughput follow-ups
  ([#110](https://github.com/laurigates/loractl/issues/110) the bench harness,
  [#162](https://github.com/laurigates/loractl/issues/162) the timing-mechanism
  ADR that waits on its first dispatch)
- **Deciders:** loractl maintainers
- **Builds on:** [ADR-0005](0005-int4-training-vram-bound.md) (the VRAM
  reclassification and its Addendum 3 measured 19.4 GB / ~4 GB-headroom fit) and
  [ADR-0008](0008-host-offload-mechanism-and-scope.md) (the precedent for
  triaging an external prior-art review claim-by-claim, and the rule that a
  lever spending throughput for VRAM cannot land before #110 can price it)
- **Numbering:** ADR-0009 is deliberately skipped — it is reserved by
  [#162](https://github.com/laurigates/loractl/issues/162) for the #110 bench
  timing mechanism, to be written *after* the first real dispatch.

## Context

An external analysis was supplied attributing slow loractl **model loading** and
**image encoding** on the RTX 4090 to four causes, with a fix for each:

1. host-to-device transfers run from **pageable** rather than pinned memory, so
   the driver cannot use `cudaMemcpyAsync` — fix: allocate with
   `cudaMallocHost`/`cudaHostAlloc`;
2. image decoding is **CPU-bound in the `image` crate** — fix: swap to
   `zune-jpeg`/`imageproc` and parallelize with rayon; and **no latent caching**
   — fix: cache VAE latents to disk up front;
3. **`autotune` is not enabled** for `burn-cuda`, so CubeCL falls back to
   suboptimal convolution layouts instead of engaging Tensor Cores;
4. accidental `.into_data()` / `.to_data()` **stream synchronization** stalls the
   asynchronous backend.

Plus an `nsys` recipe to isolate which of the four it is.

**The premise needs auditing before the fixes do.** loractl has no measured step
time, and no measured load time, on the 4090. `just bench` (#110, PR #161) — the
harness that can price a step — has only ever run against the offline two-block
toy at 32px, whose numbers the justfile itself calls meaningless; the roadmap's
open-questions paragraph says exactly this ("step **throughput** is unmeasured on
real hardware… the number needs a GPU dispatch"). So "loractl is slow" is not an
established fact in this repo, and a diagnosis of *why* sits upstream of a number
nobody has. The analysis arrived with no timings, no profile, and no host
description.

What source inspection *can* settle without hardware, it settles completely —
which of these levers exist, which are already taken, and which cannot exist at
loractl's layer. This ADR records that triage so a dispatch spends its time on
the live questions, and so the dead levers are not re-proposed. It follows
ADR-0008's shape deliberately: a full claim ledger, so the source does not need
re-reading.

## The claim ledger

Claims are labelled **VERIFIED** (a primary source — published crate source,
this repo, or `cargo`'s own resolution — was read for this ADR) or **DERIVED**
(arithmetic over verified shapes). Nothing here is measured on hardware; that is
the point of Decision 1.

| # | Claim | Disposition for loractl |
|---|---|---|
| 1 | H2D runs from pageable memory; pin it with `cudaMallocHost` | **Dead at our layer, and already done below us.** cubecl-cuda 0.10 stages through pinned memory itself for transfers ≤100 MB and copies with `memcpy_htod_async` (`src/compute/command.rs:127-152`, `:528` — VERIFIED). loractl allocates no host buffer for upload: every tensor goes through `Tensor::from_data` → the cubecl client, so it inherits that staging. burn 0.21 exposes no pinned-allocation API to call, and `loractl-core` does not link cudarc — reaching for one would invert the ADR-0005 layering rule (cubecl provides buffers, burn owns activations, loractl owns the model and the loop). **One live sub-question survives:** that 100 MB threshold is a cliff, and the fp8 → f32 load path materializes `[d_out, d_in]` weights as f32, so the 6144×6144 attention projections (`wq`/`gate`/`wo`, `mmdit.rs:561-565`) land at ~151 MB and the SwiGLU projections (6144×16384 via `MmditConfig::swiglu_dim`, `mmdit.rs:211-214`, `:675-683`) at ~403 MB — both above it, pinning skipped. Cheap to see in a profile; a cubecl-side constant, not a loractl bug |
| 2 | Dequantize on the GPU, not the CPU, before H2D | **Half true, and the half that is true has a different fix.** int4/int8 base quantization already runs *on device*: `load_quant_module` uploads one transient f32 weight and quantizes it there (`diffusion_trainer.rs:1095-1115` — VERIFIED). But the **scaled-fp8 dequant is on the CPU**, single-threaded, per tensor: `LUT[byte] · scale` into a `Vec<f32>` (`src/fp8.rs::dequant_snapshot` — VERIFIED). Moving *that* to a kernel is not a config change — burn 0.21 has no fp8 dtype at all (which is why `fp8.rs` exists), so it means uploading e4m3 as `u8` and writing a CubeCL dequant. Real, but priced only after Decision 4's cheaper fixes |
| 3 | `image` is CPU-bound; swap to `zune-jpeg` | **Already there, transitively.** `image` 0.25.10 decodes JPEG *through* zune-jpeg 0.5.15, which is in `Cargo.lock` via `image` (VERIFIED). (`imageproc` is not a decoder — it is a filter/geometry library over `image`.) What *is* single-threaded is the Lanczos3 resize and the HWC→CHW conversion loop in `dataset.rs::load_image_for_bucket:185-216`, plus the extra `to_image()` copy after the crop. rayon there is a genuine win — see Decision 5 — but it is a **one-time** cost, not a per-step one |
| 4 | Implement up-front latent caching to keep the VAE out of the loop | **Shipped in M12 (#23).** Latents *and* conditioning are cached as safetensors under `<dataset>/.loractl-cache/` keyed by name/bucket/encoder fingerprint; `DiffusionTrainer` runs a separate encode phase, **drops the encoders before the MMDiT loads**, and the training-phase closures `bail!` if a cache miss appears (`diffusion_trainer.rs:1292-1297`, the two `prepare_dataset` miss closures — VERIFIED). `tests/dataset_pipeline.rs` proves warm epochs never call the encoders by passing closures that *panic*. The VAE is not in the step |
| 5 | (unstated by the analysis) | **New, and the most decision-relevant finding here.** The cache *read* is eager and device-resident: `prepare_dataset` pushes every example's latent **and** conditioning into `PreparedDataset` as live device tensors (`dataset.rs:400-532` — VERIFIED). Conditioning is fixed-length by construction — captions are right-padded to `max_length` regardless of length (`qwen3vl.rs::tokenize:587-621`), `max_length = 512` for both Krea 2 variants (`diffusion_trainer.rs::variant_configs:115-126`) — so each example is `[1, 512, 12, 2560]` f32 = **60 MiB of VRAM, per example, held for the whole run** (DERIVED). Against ADR-0005 Addendum 3's ~4 GB headroom at 512px int4, ~65 examples exhausts the card on dataset residency alone. This is a **fit** question, not throughput, and it is the analysis's instinct ("load the cached tensors per batch") landing on a real gap loractl has: it caches to disk, then reads the whole thing back at once |
| 6 | Enable `autotune` for `burn-cuda` or CubeCL picks suboptimal conv layouts | **The named feature does not do that — but it is next door to the one real config gap.** cubecl 0.10 has **no `autotune` feature at all** (only `autotune-checks`), and neither does cubecl-runtime 0.10 (VERIFIED, both manifests): kernel-level autotune is unconditionally compiled in, so there is no "compiled-out autotune" to restore. burn's `autotune` feature forwards `burn-cuda?/autotune` → `burn-cubecl/autotune` → `burn-cubecl-fusion?/autotune` — it enables *fusion's* autotune only (VERIFIED). What **is** off is **`fusion`**: `burn` declares `[dependencies.burn-cuda] default-features = false` (burn 0.21.0 manifest `:292-295`) and burn-cuda's `fusion`/`autotune` live only inside its `default`. Confirmed against the actual resolved graph rather than inferred — `cargo tree -p loractl-core --features cuda -e features -i burn-cuda` reports `burn-cuda feature "std"` and nothing else (VERIFIED). Consequence: `burn::backend::Cuda` resolves to the raw `CubeBackend<CudaRuntime, f32, i32, u8>`, not `Fusion<CubeBackend<…>>` (the two cfg'd aliases in `burn-cuda/src/lib.rs`), so every elementwise chain in the MMDiT — zero-centered RMSNorm, sigmoid gating, 6-way modulation, residuals — runs as separate kernels with full global-memory round-trips. Promoted to a gated A/B by Decision 3 |
| 7 | The backend builds an execution graph asynchronously; stray readbacks stall it | **Describes a configuration loractl is not running, and the readback it names is load-bearing.** Without `fusion` there is no op queue to build, reorder or fuse — cubecl dispatches to a CUDA stream asynchronously, and that is all. The one per-step host readback is the `into_scalar()` on the training-loop loss (`diffusion_trainer.rs:1578`), whose next line says why it is deliberate: it is the fence the #110 timer measures *between*. Removing it would silently void every quoted `ms=`/`tok_s=` — #162 already names this as the invariant worth a `.claude/rules` entry. The encode-phase `to_owned_f32` readbacks are the disk-write path; a cache is a host-side artifact by definition |
| 8 | `nsys profile … cargo run --release -- --config config.toml` | **Right tool, wrong invocation.** loractl's config is a **positional YAML** path (`loractl train <config.yaml>`, `cli.rs:246-248`), never `--config`, and never TOML. For the load phase specifically the bench example is the better subject, since it drives the same `select_trainer` path and already reports `vram_mib=` per step. Corrected recipe in Consequences |

**Deliberately not adopted, recorded so it is a decision and not an oversight:**
the analysis's framing that PyTorch trainers win here via "CuDNN autotuning" and
"fast C++ dataloaders". Both are true of PyTorch and neither is reachable from
this stack; loractl's comparable levers are the ones in this ledger. Its
kohya-latent-caching observation is correct and was already the M12 design.

## Decision

1. **No throughput change lands unmeasured.** The gate is a real `just bench`
   dispatch on the self-hosted 4090 (`gh workflow run gpu.yml`), read under the
   existing rules: never quote `tok_s=`/`tflops=` without the `MODEL` line they
   are a quotient of, and discard any run reading `sanity=SUSPECT` /
   `plausible=false`. This is ADR-0008's rule applied to a lever that spends
   correctness risk rather than VRAM, and #162's argument applied to its own
   subject matter: an unmeasured performance document costs more to unwind than
   it saves.
2. **The pinned-memory lever is closed** at loractl's layer (ledger #1). The
   >100 MB staging cliff is the only part worth a profile line, and it is a
   cubecl-side observation to report upstream if it shows, not a loractl change.
3. **`fusion` (with its autotune) becomes a dispatchable A/B, not a default
   flip.** The change is one line — `cuda = ["burn/cuda", "burn/fusion",
   "burn/autotune"]` — and it must clear **both** gates before it is adopted,
   because it changes the concrete backend type underneath the whole quant and
   block-checkpointing stack:
   - `just bench` for the win, and
   - a re-run **zero-panic** `just step-probe` for the memory regression.
     Fusion's lazy op queue changes *when* allocations happen, and the 512px int4
     peak is 19.4 GB with ~4 GB headroom (ADR-0005 Addendum 3). A survived OOM
     storm is not a pass — that gate is unchanged.

   Compatibility is *plausible but not proven*: `impl QuantBackend for
   burn::backend::Cuda` would then apply to `Fusion<…>`, and burn-fusion 0.21
   does implement `quantize`/`dequantize`/`q_matmul`
   (`burn-fusion-0.21.0/src/ops/qtensor.rs` — VERIFIED), so int4 is not
   obviously excluded; but quantized `Transaction` reads are
   `todo!("Quantization not supported yet")` (`src/ops/transaction.rs:21`), a
   live panic surface. The A/B is also not compile-verifiable off-box: the
   `cuda` feature needs nvcc, so the first thing the dispatch learns may be that
   it does not build.
4. **If *loading* is the complaint, the mechanism is host-side serialization —
   not pinning.** `load_quant_module` walks the base-linear sites strictly
   sequentially, and per site does: re-parse the safetensors header → CPU
   dequant/convert → H2D → GPU quantize → drop. The loop's own progress-throttle
   comment describes the real-model cost: it "dequantizes, quantizes and stores
   261 multi-hundred-MiB tensors over minutes". Three structural costs, cheapest first:
   - **the header is re-deserialized per tensor** — every snapshot closure calls
     `SafeTensors::deserialize(&mmap)` again (`fp8.rs::dequant_snapshot`,
     `fp8.rs::plain_snapshot:341-353` — VERIFIED). Parse once per file;
   - **the CPU dequant is single-threaded** — there is no rayon anywhere in the
     workspace (VERIFIED);
   - **CPU and GPU never overlap** — the GPU idles through the dequant, the CPU
     idles through the quantize.

   The fix is bounded-depth prefetch (dequant tensor *N+1* on a worker while the
   GPU quantizes *N*), with the depth explicit, because each in-flight worker
   adds one transient f32 weight to host-RAM peak — the exact quantity the
   streamed loader was built to bound. **It cannot be a rayon wrapper around the
   existing closures:** burn-store 0.21's `TensorSnapshot::from_closure` takes an
   `Rc<dyn Fn() -> …>` (`burn-store-0.21.0/src/tensor_snapshot.rs:204-205` —
   VERIFIED), so a snapshot is not `Send`. A parallel path has to produce
   `TensorData` off the snapshot API.
5. **Parallelizing image decode/resize is real but lowest priority.** It is a
   one-time cache-fill cost amortized over every epoch (ledger #3/#4); the
   encode phase's own dominant cost is the VAE and text-encoder forwards, which
   are already on the GPU.
6. **Dataset-cache residency is classified as a fit question, not a throughput
   one, and is tracked as
   [#175](https://github.com/laurigates/loractl/issues/175)** (ledger #5):
   60 MiB of VRAM per example, linear in dataset size, against ~4 GB of
   headroom — a sibling of #147–#149 in the dataset pipeline, but a memory bug
   rather than an ergonomic one. The candidate fix — read a batch's
   latents/conditioning lazily per step instead of materializing the whole
   dataset on the device — trades a little PCIe traffic per step for O(batch)
   residency instead of O(dataset), and is measurable by the same two probes as
   Decision 3.
7. **The existing synchronization points stay.** The per-step loss readback is
   the bench fence; "removing accidental syncs" is a change that would break
   measurement while looking like an optimization.

## Consequences

This ADR's live levers now have issues —
[#174](https://github.com/laurigates/loractl/issues/174) (Decision 3, the
`fusion` A/B), [#175](https://github.com/laurigates/loractl/issues/175)
(Decision 6, dataset-cache residency),
[#176](https://github.com/laurigates/loractl/issues/176) (Decision 4,
host-serial load), [#177](https://github.com/laurigates/loractl/issues/177)
(ledger #2, GPU fp8 dequant),
[#178](https://github.com/laurigates/loractl/issues/178) (Decision 5, image
decode). The four perf issues stay ordered behind the #110 dispatch of
Consequence 1 below; #175 is a memory bug and is not gated on it.

Ordered, because each step's result decides whether the next is worth doing:

1. **Dispatch `just bench` on the 4090** (`gh workflow run gpu.yml`). Until this
   exists, every claim in the analysis — and every number in this ADR's
   Decision 3/4 cost estimates — is a prediction. This also unblocks #162 and
   #158, which are both waiting on the same dispatch.
2. **If the step is slower than the `MODEL` line's floor:** run the Decision 3
   fusion A/B, both gates.
3. **If loading is the complaint:** profile the load phase, then take Decision
   4's fixes cheapest-first. The corrected recipe, against the real binary:

   ```bash
   # the load + step path the bench harness already instruments
   nsys profile -t cuda,osrt -o loractl_load --stats=true \
     cargo run --release -p loractl-core --features cuda --example bench_step -- \
     config/examples/krea2-comfyui.yaml --steps 3

   # or a full run through the CLI (positional YAML, not --config)
   nsys profile -t cuda,osrt -o loractl_train --stats=true \
     cargo run --release -p loractl-cli -- train config/examples/krea2-dog.yaml
   ```

   Read it the way the analysis suggests — gaps between `cudaMemcpyHtoD` and
   kernel execution — with one correction: a sparse GPU timeline during
   *loading* is expected here and means Decision 4, not a starved dataloader.
   The dataloader cannot starve the step at all, because M12 took the decoder,
   the VAE and the text encoder out of the loop entirely.
4. **Only then** consider the fp8 dequant kernel (ledger #2) or rayon over image
   decode (Decision 5).

What this ADR does **not** claim: that loractl's step time is fine, that it is
slow, or that fusion will help. It claims that three of the four proposed fixes
are already shipped or unreachable, that the fourth is misidentified but sits
beside a real one, and that the ordering above is what a measurement should be
spent on.

## Alternatives considered

- **Act on the analysis as written.** Rejected: it would have added a
  `cudaMallocHost` path duplicating what cubecl already does (and breaking the
  ADR-0005 layering rule), swapped `image` for a decoder it already uses,
  re-implemented M12's latent cache, and enabled a feature that is a no-op for
  the effect claimed — while leaving the one real config gap (`fusion` off) and
  the one real residency bug (60 MiB/example) untouched.
- **Reply to the analysis in a comment and keep no record.** Rejected on the
  ADR-0008 precedent: an external review triaged in a thread gets re-mined by
  the next person who reads a similar post. The ledger exists so the source does
  not need re-reading.
- **A `.claude/rules/` entry instead of an ADR.** Rejected for the bulk of it —
  the durable content is a decision record with a citation trail, not a trap a
  session re-derives. The one rules-shaped fragment is "do not remove the
  per-step loss readback; it is the bench fence", and #162 has already scoped
  that as its own deliverable; duplicating it here would create two homes for one
  invariant.
- **Enabling `fusion` immediately, since it is one line and plainly an
  improvement.** Rejected per Decision 3: it changes the concrete backend type
  under the int4 + block-checkpointing stack whose fit was measured at ~4 GB of
  headroom, and it cannot even be compile-checked without nvcc. "One line" is a
  statement about the diff, not about the risk.

## References

External (VERIFIED — read for this ADR, at the versions this workspace resolves):

- `cubecl-cuda` 0.10.0 — `src/compute/command.rs:127-152` (pinned staging for
  ≤100 MB transfers, `reserve_pinned`), `:528` (`memcpy_htod_async`)
- `cubecl` 0.10.0 / `cubecl-runtime` 0.10.0 manifests — no `autotune` feature
  exists; only `autotune-checks`
- `burn` 0.21.0 manifest — `[dependencies.burn-cuda] default-features = false`
  (`:292-295`); `autotune = ["burn-wgpu?/autotune", "burn-cuda?/autotune", …]`
- `burn-cuda` 0.21.0 — `default = ["std", "fusion", "autotune", …]`;
  `src/lib.rs` (the two cfg'd `Cuda` aliases)
- `burn-cubecl` 0.21.0 — `autotune = ["burn-cubecl-fusion?/autotune"]`
- `burn-fusion` 0.21.0 — `src/ops/qtensor.rs` (quantize/dequantize/`q_matmul`
  implemented), `src/ops/transaction.rs:21` (`todo!("Quantization not supported
  yet")`)
- `burn-store` 0.21.0 — `src/tensor_snapshot.rs:204-205`
  (`from_closure(data_fn: Rc<dyn Fn() -> …>)`, hence not `Send`)
- `cargo tree -p loractl-core --features cuda -e features -i burn-cuda` — the
  resolved feature set for burn-cuda in this workspace: `std` only

Internal:

- [`docs/roadmap.md`](../roadmap.md) — "step **throughput** is unmeasured on real
  hardware"; M12's cache; M15's fp8 loader
- [ADR-0005](0005-int4-training-vram-bound.md) Addendum 3 — the 19.4 GB / ~4 GB
  headroom fit the fusion A/B must not regress
- [ADR-0008](0008-host-offload-mechanism-and-scope.md) — the claim-ledger
  precedent and the #110-gates-throughput-levers rule
- [#110](https://github.com/laurigates/loractl/issues/110) /
  [#162](https://github.com/laurigates/loractl/issues/162) — the harness and the
  reserved ADR-0009
- `crates/loractl-core/examples/bench_step.rs` — the output schema and the
  derived-figure policy any quoted number must travel with

External (UNVERIFIED — the source claim itself):

- The supplied RTX 4090 analysis. No timings, no profile, no host description,
  and no comparison run; its PyTorch-side attributions (CuDNN autotuning, C++
  dataloaders, kohya latent caching) are plausible as PyTorch statements and were
  not independently checked, since none of them is reachable from this stack.
