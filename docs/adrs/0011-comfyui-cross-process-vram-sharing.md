---
id: ADR-0011
status: Accepted
date: 2026-07-31
---

# 0011 — Cross-process VRAM sharing with ComfyUI is closed at the cubecl storage layer, and would not fit the card even if it were open

- **Status:** Accepted
- **Date:** 2026-07-31
- **Milestones:** post-M15; no issue — this ADR closes the question rather than
  opening work
- **Deciders:** loractl maintainers
- **Builds on:** [ADR-0005](0005-int4-training-vram-bound.md) (the measured
  19.4 GB / ~4 GB-headroom fit, and the co-tenancy hazard note),
  [ADR-0008](0008-host-offload-mechanism-and-scope.md) (the rejected
  base-weight-streaming alternative, and the rule that a lever spending
  throughput for VRAM cannot land before #110 can price it), and
  [ADR-0010](0010-rtx4090-throughput-lever-triage.md) (the precedent for
  triaging an external proposal claim-by-claim)
- **Supersedes nothing.** It *upgrades* two prior assertions from "verified" to
  "demonstrated, with the mechanism quoted" — see Decision 1.

## Context

A proposal was supplied to let loractl train a LoRA against base weights that
are **already resident in ComfyUI's VRAM**, so that training and generation can
share one copy of Krea 2 on a single 24 GB RTX 4090. The mechanism:

1. Run [`shared-tensor`](https://github.com/world-sim-dev/shared-tensor) inside
   ComfyUI as a "VRAM model server", exposing the loaded `torch` tensors over a
   Unix domain socket.
2. Have loractl read the exported `cudaIpcMemHandle_t` from that socket and call
   `cudaIpcOpenMemHandle` via `cudarc`.
3. Wrap the resulting device pointer in a burn/cubecl buffer and run the forward
   pass directly against ComfyUI's allocation.
4. For live preview, push in-progress LoRA weights back to ComfyUI over a
   reverse IPC channel.

The library is real: `shared-tensor` v0.2.16, published 2026-03-28, MIT,
**Development Status 3 – Alpha**. Its socket is
`/tmp/shared-tensor-<device>.sock` (`SHARED_TENSOR_BASE_PATH`), not the
`/tmp/shared-tensor.sock` the proposal assumed.

Step 3 is the load-bearing one, and it was the one nobody had tested. ADR-0008
and ADR-0010 both already asserted the negative — "cubecl exposes no
managed-memory allocation path (verified)" and "burn 0.21 exposes no
pinned-allocation API to call" — but neither quoted source. This ADR records a
spike that held those assertions to an empirical standard, deliberately ordered
to test the most-likely-to-fail rung first.

## Decision 1 — cubecl 0.10 cannot hold a pointer it did not allocate. Demonstrated, not asserted.

Read at the pinned versions (cubecl `0.10.0` from crates.io, the version loractl
resolves transitively through its burn 0.21 fork):

**The trait admits no foreign memory.** `ComputeStorage`
(`cubecl-runtime-0.10.0/src/storage/base.rs:74-95`) has exactly five methods —
`alignment`, `get`, `alloc`, `dealloc`, `flush`. The only entry point for memory
is `fn alloc(&mut self, size: u64)` (`:86`), which takes a **size**, not a
pointer. There is no `register`, `adopt`, `from_ptr`, or `external`.

**Nor does the CUDA implementation.** In
`cubecl-cuda-0.10.0/src/compute/storage/gpu.rs`:

- `GpuStorage` (`:18-23`) keeps `memory: HashMap<StorageId, (CUdeviceptr,
  AllocationKind)>` — and the field is **private** (`:19`).
- Its only public inherent constructor is `new(mem_alignment, stream)` (`:50`).
- The **sole** insertion into `memory` is inside `alloc` (`:169`), which
  unconditionally calls `malloc_async` (`:174`) or falls back to `malloc_sync`
  (`:178`) *before* inserting. A pointer that cubecl did not allocate can never
  enter the map.
- `AllocationKind` (`:9-12`) is a **private** enum with exactly two variants,
  `Async` and `Sync` — both meaning "cubecl owns this". There is no
  `External`/`Borrowed` variant.

**The back door is closed too.** `StorageHandle::new` and `StorageId::new` are
public, so a handle *can* be hand-constructed — but `get` resolves it via
`self.memory.get(&handle.id).expect("Storage handle not found")` (`:148-152`).
A hand-made handle panics. Both ends are shut.

A grep for every adoption-shaped name — `register|adopt|from_ptr|from_raw|
from_device_ptr|external|import|wrap|attach` — across the **entire** source of
`cubecl-runtime-0.10.0` and `cubecl-cuda-0.10.0` returns **zero** hits. One
layer up, `MemoryManagement::reserve`
(`cubecl-runtime-0.10.0/src/memory_management/memory_manage.rs:451`) is likewise
`(&mut self, size: u64)` — size-only, same story.

This is not a gap a loractl-side change can bridge. Per ADR-0010 Decision 2, a
cubecl-side gap is "an observation to report upstream, not a loractl change" —
and the minimum upstream shape is now concrete: a third `AllocationKind` variant
that `perform_deallocations` skips, plus a public registration path on
`GpuStorage`. That is an upstream cubecl feature, then a burn change, then
loractl.

## Decision 2 — if a foreign pointer *were* forced in, cubecl would free it. ComfyUI would be corrupted.

The spike's second rung was going to test this on hardware. It did not need to:
`perform_deallocations` (`gpu.rs:63-80`) drains every pending id, removes it from
`memory`, and frees it unconditionally — `free_async(ptr, self.stream)` for
`Async` (`:72`), `free_sync(ptr)` for `Sync` (`:75`). There is no ownership flag
to consult, because there is no variant that means "not mine".

So a fork that merely widened the map without also teaching `perform_deallocations`
to skip foreign entries would hand ComfyUI's live model weights to `cuMemFree` —
silent VRAM corruption in another process, which is the worst failure shape
available here and would not surface as an error in either process.

## Decision 3 — the reverse direction (loractl → ComfyUI) is separately unavailable

The proposal's step 4 asks loractl to export its in-progress LoRA tensors back
over IPC. cubecl's `alloc` prefers `malloc_async` (`gpu.rs:174`) — stream-ordered
memory-pool allocations. NVIDIA documents sharing those through a **different**
API family than the `cudaIpcGetMemHandle` the proposal names:
`cudaMemPoolExportToShareableHandle`/`ImportFromShareableHandle` plus
`cudaMemPoolExportPointer`/`ImportPointer`. IPC capability must also be requested
**at pool creation** — "setting `handleTypes` to a non zero value will make the
pool exportable (IPC capable)" — and `GpuStorage::new` (`:50`) sets no pool
properties at all, taking the device default.

This costs nothing, because the reverse channel is **already shipped over disk**:
`diffusion_trainer.rs:1650-1663` writes a kohya `checkpoint-{step}.safetensors`
every `output.checkpoint_every` steps (default 250) in `Krea2Diffusers` format,
whose keys are pinned against ComfyUI's real key map by
`tests/krea2_lora_keys.rs`. The adapter is tens of MB. No IPC, no `unsafe`, and
already correct.

## Decision 4 — the goal does not fit the card even on the most generous assumptions

Worth stating separately, because it holds *independently* of Decisions 1–3: even
granting a perfect zero-copy import, "train while ComfyUI generates" does not fit
24 GB at 512px.

| | VRAM |
|---|---|
| loractl training peak, measured (ADR-0005 Addendum 3) | **19.4 GB** |
| — of which int4 Q4S base | ~10.1 GB |
| — loractl's non-base working set (activations, block pins, optimizer) | ~9.3 GB |
| ComfyUI's Krea 2 base — the copy actually shareable (int8 repack, measured) | **~13.5 GB** |
| ComfyUI non-base (Qwen3-VL encoder + VAE + generation activations) | several GB |
| **Best case with a perfectly shared base** | **~23 GB + ComfyUI's non-base > 24 GB** |

The 13.5 GB is measured, not estimated, from the safetensors header of the
checkpoint actually in use — `Comfy-Org/Krea-2` @ `7b75ff3`,
`diffusion_models/krea2_turbo_int8_convrot.safetensors`: **13,492,686,496 bytes**,
878 tensors, comprising **224 `I8` weights (12.16 GB)** each paired with an F32
`weight_scale` and a U8 `comfy_quant` marker, plus 76 `BF16` tensors left
unquantized (1.33 GB) and 354 F32 norm scales (0.01 GB). The widely-mirrored
scaled-fp8 repack is ~13.1 GB, so the conclusion is insensitive to which of the
two ComfyUI holds.

The shared copy is also the *wrong* one. loractl trains against int4 Q4S —
block 32, symmetric per-block scales, packed `u32` (`quant.rs:52-162`,
`mmdit.rs:407-421`) — quantized **on device by loractl itself**. ComfyUI holds
either `float8_e4m3fn` or this `I8` + `weight_scale` + `comfy_quant` scheme, both
row-major. Aliasing either as packed-`u32` Q4S is not a cast; it is a different
number system. Even reading ComfyUI's *own* fp8 file today, loractl dequantizes
host-side (`fp8.rs:279-329`, `lut[byte] * scale` into a `Vec<f32>`) and
re-quantizes on device. In-place consumption would need a device-side dequant
kernel, and loractl writes no GPU kernels.

So the best achievable outcome is sharing a **~13.5 GB int8 base in place of the
~10.1 GB int4 base loractl already uses** — over 3 GB *worse* than today, for an
architecture that requires an upstream cubecl feature and a Rust reimplementation
of torch's IPC refcounting.

ADR-0008 already rejected the adjacent idea — "offloading the quantized base
weights instead of activations" — on the same ground: quantization banked that
saving (~49 GB f32 → ~10.1 GB int4). Nothing here reopens it.

## Decision 5 — the int8 ComfyUI repack is not a loadable loractl input today, over any transport

Found while measuring Decision 4, and worth recording on its own because it bites
the **disk** path that *is* supported, independently of everything above.

`krea2_turbo_int8_convrot.safetensors` contains no `F8_E4M3` tensor — its dtypes
are `I8`/`BF16`/`F32`/`U8`. So `is_fp8_checkpoint` (`fp8.rs:74-83`, which returns
true only when some tensor is `F8_E4M3`) is **false**, and the trainer routes it
to the plain `load_module` path (`diffusion_trainer.rs:777`) instead of
`load_fp8_module`. Nothing on that path pairs a weight with its scale, and three
separate layers decline to catch it:

1. burn-store's dtype map accepts `Dtype::I8 => DType::I8`
   (`burn-store/src/safetensors/store.rs:1055`) — no error.
2. `CastFloatsAdapter` (`diffusion_trainer.rs:744-774`) tests `is_float` against
   `F64|F32|Flex32|F16|BF16` and returns non-float snapshots **unchanged** — so
   `I8` passes straight through, untransformed.
3. The applier validates **shape but not dtype** (`burn-store/src/applier.rs:228-240`):
   the `I8` weight has the same shape as the float param it targets, so there is
   no `ShapeMismatch`, and it proceeds to
   `Tensor::from_data(data, (target_device, snapshot.dtype))`.

The 224 `weight_scale` tensors are keys no module param claims, so they are
reported as **"Unused Tensors"** — the same signature
[`burn-store-skip-enum-variants.md`](../../.claude/rules/burn-store-skip-enum-variants.md)
documents. Whether the final `from_data` coerces the raw integers into the float
param or rejects them is backend-dependent and untested here; what is certain
from the three layers above is that **the scale is never applied**, so the
checkpoint cannot load correctly by any route. Verified by reading the load path
at the pinned versions, not by running it — there is no GPU on this box.

Supported denoiser inputs today are therefore bf16/f32 or a ComfyUI **scaled-fp8**
repack, with loractl performing its own int4/int8 quantization on device. Adding
an int8-repack reader would mean pairing `<name>.weight_scale` in the same shape
`fp8.rs` already does for fp8 — tractable, and unrelated to IPC — but it does not
exist, and until it does this checkpoint must not be pointed at
`model.denoiser`.

Two further notes on this checkpoint: it is **Turbo**, and ADR-0004 makes
Krea-2-**Raw** the LoRA-training target — Turbo wants the ostris training
adapter, which is itself incompatible with `compute.quant` (#83). And it is a
distinct artifact from same-named community re-uploads: a mirror inspected first
(`AX1Y2JP/Krea-2-Turbo-INT8-ConvRot`) is 16.17 GB / 816 tensors / 192 quantized
sites with 4.16 GB left in BF16 — same scheme, materially different content. Pin
the source when quoting a size.

## Consequences

- **The architecture is closed.** No `foreign_ptr_probe.rs`, no `cudarc`
  dependency, no `gpu.yml` input, no GPU hours spent. The spike stopped at its
  first rung by design.
- **Two ADRs' "verified" is now demonstrated** with quoted source and line
  numbers, at pinned versions. Re-check on the burn 0.22 migration (#79), which
  moves cubecl — this ADR is version-scoped, not a permanent law.
- **The co-tenancy goal keeps its existing answer: time-division.** ADR-0005
  already records that "an idle ComfyUI can hold ~18 GB of cached models and
  re-grab the card mid-run"; ComfyUI's `free_memory()` unloads on demand. Free
  the card before a training run rather than aliasing during one.
- **Reading ComfyUI's model *directory* was never the blocked part** and remains
  shipped: `config/examples/krea2-comfyui.yaml` points at a scattered ComfyUI
  `models/` tree with "no restructuring into an HF snapshot dir, no duplicate
  files, no symlinks." Decision 5 narrows *which files* in that tree are valid
  denoiser inputs — the int8 repack is not one of them.
- **Decision 5 is the only actionable item here**, and it is unrelated to IPC: an
  int8-repack reader pairing `<name>.weight_scale` the way `fp8.rs` already pairs
  fp8 would close it. Not filed as an issue by this ADR; worth one if anyone is
  pointing `model.denoiser` at an int8 ComfyUI checkpoint.

## Alternatives considered

- **Fork cubecl to add an `External` allocation kind.** Rejected for now, not on
  feasibility — the shape is known (Decision 1) — but on payoff: Decision 4 shows
  the completed feature makes the fit *worse*, so the fork would buy a
  regression. Revisit only if the dtype gap closes first.
- **Have ComfyUI hold an int4 copy in loractl's layout.** Rejected: cubecl's
  packed-`u32` Q4S is an internal representation with no PyTorch producer, and
  ComfyUI could not run inference against it.
- **Reverse-engineer `shared-tensor`'s wire protocol for a Rust client.**
  Rejected as moot given Decision 1, and independently unattractive: the library
  documents only "control plane is a UDS RPC channel, data plane is native
  `torch` CUDA IPC serialization" — no schema, no format, no stability
  commitment, no non-Python client support, at alpha maturity. `reduce_tensor`
  emits a *pickled* tuple carrying a storage offset, a ref-counter shared-memory
  file and an IPC **event** handle, not a bare 64-byte handle; a consumer must
  join the `CudaIPCSentData`/limbo refcount protocol, and getting it wrong reads
  freed VRAM rather than erroring. A further hazard for loractl-as-receiver:
  PyTorch's `expandable_segments` routes IPC through `pidfd_open`/`pidfd_getfd`,
  and [pytorch#186213](https://github.com/pytorch/pytorch/issues/186213) reports
  the *receiver* reserving ~9/8 of total GPU VA per imported handle.
- **NVIDIA MPS to overlap the two processes' SM usage.** Out of scope here: MPS
  addresses scheduling, not memory, so Decision 4's budget is unchanged. It also
  falls under ADR-0008 Decision 3 / ADR-0010 Decision 1 — a lever trading
  throughput cannot land before #110 has a real GPU dispatch, and loractl has
  still never timed a training step on the 4090.
