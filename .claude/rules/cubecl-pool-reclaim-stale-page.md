# cubecl 0.10 Pool Reclaim Corrupts Under Real Pressure — the #25 Real Run's Blocker Is cubecl#1401, Not VRAM

The int4/int8 quant work is done and correct (loractl #119, #96). The
real-model Krea-2 training run (#25) is blocked by a **cubecl-cuda memory-pool
bug** ([tracel-ai/cubecl#1401], root cause [#1384]) that panics a real training
step with `couldn't find resource for that handle: Memory page N doesn't exist`.
This is a **reclaim-correctness bug, not an out-of-memory** — the int4 live
working set fits comfortably under 24 GB. Sibling to
[`burn-wgpu-metal-numerics.md`](burn-wgpu-metal-numerics.md) (the *other* GPU
backend bug that blocks the pure-Rust run): that one is burn wgpu/Metal
autodiff; this one is cubecl-cuda's pool.

## The memory picture (measured on-box, RTX 4090, 2026-07-18)

int4 does exactly its job — the blocker is downstream of it:

| scheme | reclaimed resident base | free on 24 GB | dequant worst-max |
|---|---|---|---|
| int8 | ~17.1 GB | ~6.9 GB | 0.4% |
| **int4** | **~10.1 GB** (261/261 sites, Q4S/block-32) | **~14 GB** | 7% |

So the *live* set is ~10 GB base + one ~906 MB transient + activations, well
under 24 GB. The step nonetheless **peaks at ~24 GB** — that peak is the
**transient ratchet**, not the live set.

## The bug (cubecl#1401)

`SlicedPool`/`ExclusiveMemoryPool::cleanup` reclaim only on the **explicit**
path, and cubecl-cuda (unlike cubecl-wgpu, which cleans every submit) never
cleans on the healthy hot path. A QLoRA-style forward transiently dequantizes
~260 frozen base weights (upload → matmul → **drop**), so reserved memory
ratchets to ~24 GB even though the live set fits. When `reserve` then OOMs, its
reclaim-and-retry (`cleanup(true)`) tombstones fully-free pages — **but cubecl
resolves kernel bindings lazily at dispatch**, so:

1. transient `T` bound (dequant of a weight); matmul `(x, T)` **queued**,
2. `T`'s host handle drops → its page is `is_free`,
3. `reserve` OOMs → reclaim tombstones `T`'s page,
4. the queued matmul finally **dispatches** → `find(T)` → tombstoned/reused slot
   → `NotFound` → cubecl-cuda `.unwrap()`/`.expect()` **panic** at
   `server.rs:{124 (reserve), 701 (resource)}` and `stream.rs:101 (cursor)`.

## What was tried (2026-07-18) — and the result

Against the fork `laurigates/cubecl@fix/stable-page-indices-v0.10` (rev
`fe7c4f8`, tombstone pool + reserve reclaim-and-retry, already pinned in
loractl's `Cargo.toml`):

1. **Graceful `NotFound`** — `MemoryManagement::find` `assert_eq!` → `NotFound`
   + `handle_cursor` `.unwrap()` → `.unwrap_or(u64::MAX)`. **Cleared the
   survivable `stream.rs:101` cursor panic.** Memory-safe *because* the
   tombstone pool guarantees `find` never returns the wrong page (addresses
   nathanielsimard's #1384 "don't hide the error" objection). Landed as fork
   **PR laurigates/cubecl#1**. Did NOT fix the fatal `server.rs` sites.
2. **Sync-before-reclaim** — new `ComputeStorage::sync()` (no-op default; cuda =
   `perform_deallocations()` + `cuStreamSynchronize`) called in
   `MemoryManagement::cleanup(explicit)`. **Did NOT help** — same panic, same
   ~24 GB peak. **Key insight: a *stream* sync drains GPU work but NOT cubecl's
   *command queue*** (kernels queued but not yet dispatched), so the reclaim
   still tombstones a page a queued kernel will later look up. (Not committed;
   left in the box clone's working tree.)

## The fix directions (for whoever picks this up)

Full handoff + repro is in fork issue **laurigates/cubecl#2** (and taskwarrior
task 316). Two candidate root-cause fixes:

- **Command-queue drain before reclaim** — flush/dispatch all pending commands
  so every queued binding is resolved before `cleanup(true)` frees any page.
- **Cursor-aware cleanup** (nathanielsimard's preferred style) — pages already
  carry `slice.cursor`; only free an `is_free` page whose `cursor <=
  completed_cursor`. Least-invasive; needs the completed cursor exposed to
  `cleanup`.

The live set fits, so a *correct* reclaim WILL fit — this is reclaim
correctness, not a genuine OOM.

## Repro / dev setup (popos RTX 4090)

- Editable fork clone: `/mnt/sabrent/comfyui-workspace/cubecl-fix` (branch
  `fix/stable-page-indices-v0.10`; the sync experiment is in its working tree).
- Wire into loractl: a `path = ".../cubecl-fix/crates/<name>"`
  `[patch.crates-io]` block over the 15 cubecl crates.
- loractl side: `feat/quant-int4` + cherry-pick the `memory_cleanup` reclaim
  commit; `compute.{backend: cuda, quant: int4}`.
- Config: `config/examples/krea2-comfyui.yaml`, `model.base:
  /mnt/sabrent/comfyui-workspace/ComfyUI/models` (the Krea-2 components are
  already there scattered — denoiser `diffusion_models/krea2/…`, text_encoder
  `text_encoders/qwen/qwen3vl_4b_fp8_scaled.safetensors` [Krea-2-Raw uses the
  **4B**], vae `vae/qwen/qwen_image_vae.safetensors`). Tokenizer auto-fetches.
- **Encode phase runs on ndarray/CPU single-threaded (~11 min/sample)** — cache
  one sample, then trim the dataset to it so re-runs hit the GPU step in ~1 min.
- `cargo build --release -p loractl-cli --features cuda`; `./target/release/loractl
  train <config>`. `nvidia-smi`: base loads ~14.9 GB, step ratchets to ~24 GB,
  then panics.

## Watch / status

- Upstream: [tracel-ai/cubecl#1401] (open, deterministic repro; mechanism
  refinement posted 2026-07-18), [#1384] (closed graceful-hiding attempt;
  maintainer wants root cause).
- Fork: PR `laurigates/cubecl#1` (graceful-cursor half, fork-internal — its
  safety framing needs the tombstone pool, which isn't upstream), issue
  `laurigates/cubecl#2` (handoff for the deeper fix).
- Downstream: loractl #96 (memory + on-box findings), #119 (the int4 PR —
  correct & merge-ready), #25 (the real run — blocked on THIS bug).
- The full write-up (with box-local specifics) is in `CLAUDE.local.md`; this
  rule is the checked-in, shareable summary.

[tracel-ai/cubecl#1401]: https://github.com/tracel-ai/cubecl/issues/1401
[#1384]: https://github.com/tracel-ai/cubecl/pull/1384
