# LoKr (Kronecker) Adapter Training Support

## Summary

Add **LoKr** (Kronecker-product low-rank adaptation, from LyCORIS) as a
selectable adapter parameterization alongside the existing LoRA path in
loractl. LoKr expresses a weight update as a Kronecker product `ΔW = W₁ ⊗ W₂`
(with the larger factor optionally further low-rank-decomposed), which delivers
a **higher effective rank per trained parameter** than a rank-`r` LoRA of the
same file size. This is a **quality / expressiveness / param-efficiency
feature only**. It is explicitly **not** a VRAM lever: per
[ADR-0005](../adrs/0005-int4-training-vram-bound.md), the #25 real-run blocker
is *activation retention* (67.9 GiB pinned per forward, topology-driven), and
no adapter parameterization moves that wall — a trainable factor early in the
trunk taints the entire downstream graph regardless of how the delta is
factored. LoKr rides on the already-landed fit mechanisms (block checkpointing
#134 + int4/int8 QLoRA base #96/#119); it neither helps nor is gated by them.

The delivery must include a **ComfyUI consumer-contract test** (LoKr keys load,
generated from pinned upstream source, kill-tested) and a **numerics golden vs
a LyCORIS/PyTorch reference**, because an export whose keys the consumer does
not parse loads *without error and does nothing* — the worst failure shape
(the #137 lesson, `.claude/rules/testing.md`).

## Motivation

- **Higher effective rank per parameter.** A LoRA delta on a `d_in × d_out`
  site costs `r·(d_in + d_out)` parameters and is capped at rank `r`. LoKr
  factors the site into `(a₁·a₂) × (b₁·b₂)` blocks and represents the update as
  `W₁ ⊗ W₂`; the Kronecker product of two rank-`r` factors can reach effective
  rank up to `r₁·r₂`, so for a comparable or *smaller* parameter budget LoKr
  spans a richer update subspace. This is why LyCORIS LoKr is a popular choice
  for diffusion-model character/style LoRAs where a plain LoRA of the same file
  size underfits.
- **Smallest shippable adapter file.** Among the ComfyUI-parseable formats
  (LoRA, LoHa, LoKr, OFT), LoKr produces the smallest on-disk adapter for a
  given expressiveness target — attractive for distribution.
- **Interop is already proven-out.** ComfyUI's `comfy/lora.py` has a live
  `lokr` handler (the `lokr_w1`, `lokr_w2`, `lokr_w1_a/_b`, `lokr_w2_a/_b`,
  `lokr_t2`, `.alpha` key family). We emit into an existing, enabled consumer
  path rather than an experimental one.

**What this is NOT (state bluntly, per ADR-0005):** LoKr does **not** reduce
the activation-VRAM peak. Retention is topology-driven, not parameter-count
driven; the 35.4 GiB attention-score trio and the SwiGLU/quant-site outputs are
base-trunk interior, untouched by how the adapter delta is expressed. If
anything, a *naive* materialize-ΔW LoKr forward would *add* to the pinned set
(see Design) — which the reshape-trick forward is specifically designed to
avoid. Adapter choice is a quality decision made *after* the fit is solved,
never a way to reach it.

## Goals / Non-Goals

### Goals

- A `LokrDelta` module in core that produces an **additive** `[.., d_out]`
  delta at each injection site, drop-in with the existing
  `LoraAdapters::apply` contract.
- Config selection of the adapter algorithm (`algo: lora | lokr`) with LoKr's
  `factor` and `decompose_both` knobs, globally and per-`TargetSpec`.
- A generalized `LoraAdapters` container that can hold a mix of delta types.
- kohya/LyCORIS-format LoKr export (`lokr_w1/w2[_a/_b]`, `lokr_t2`, `.alpha`)
  that **loads and visibly conditions generation in ComfyUI**, proven by a
  consumer-contract test generated from pinned upstream source, with a
  kill-test.
- A numerics golden pinning `LokrDelta`'s forward against a LyCORIS/PyTorch
  reference (RED → GREEN), with a kill-test.
- Correct behavior under block-level gradient checkpointing (#134): every LoKr
  factor re-marked for autodiff and grad-collected per block.

### Non-Goals

- **VRAM reduction of any kind.** LoKr is not a fit lever and will not be
  measured or pitched as one. The #25 fit remains int4 + #134, unchanged.
- **LoHa, DoRA, OFT/BOFT, VeRA** and other parameterizations — out of scope
  here; the container generalization this PRD lands makes them cheaper later,
  but each is its own decision (see the PEFT synthesis).
- **A new trainer or routing change.** `select_trainer` (`src/train.rs`) keys
  on `model.base` and is orthogonal to adapter parameterization — untouched.
- **The wgpu/Metal path.** Blocked upstream (burn#5162); LoKr trains on the
  same cuda-f32 / ndarray paths as LoRA.

## Background: how LoKr differs from LoRA

**LoRA** adapts a frozen linear site `W ∈ ℝ^{d_in × d_out}` with a low-rank
additive delta:

```
ΔW = (α/r) · B·A,   A ∈ ℝ^{d_in × r},  B ∈ ℝ^{r × d_out}
```

The forward at each site is `base(x) + scaling · B(A(x))` — this is exactly the
existing `LoraDelta` (`crates/loractl-core/src/lora.rs:155`).

**LoKr** replaces the low-rank product with a **Kronecker product**. It reshapes
the `d_in × d_out` update into a block grid and factors it as:

```
ΔW = W₁ ⊗ W₂
```

where the site dimensions are split by a `factor` `f`:
`d_out = a₁·a₂`, `d_in = b₁·b₂` (LyCORIS picks the split by `factor`, choosing
the largest divisor `≤ f`). `W₁` is the small `a₁ × b₁` block; `W₂` is the large
`a₂ × b₂` block. The Kronecker product of an `(a₁×b₁)` and an `(a₂×b₂)` matrix
is `(a₁a₂ × b₁b₂) = (d_out × d_in)`, i.e. the full update — but described by
`a₁b₁ + a₂b₂` parameters instead of `d_in·d_out`.

**Optional low-rank decomposition of the factors.** The large factor `W₂` (and,
with `decompose_both`, also `W₁`) may itself be written low-rank to shrink it
further:

- `W₂ = w₂_a · w₂_b` (rank `r`), giving the `lokr_w2_a` / `lokr_w2_b` keys.
- With a Tucker-style core, an extra `lokr_t2` tensor mediates the
  contraction (LyCORIS's `use_w2 = False` + `use_effective_conv2d` path);
  for the linear (non-conv) MMDiT sites we target, `lokr_t2` is the
  contraction core over the decomposed `w2_a`/`w2_b`.
- If a factor is kept full-rank it is stored directly as `lokr_w1` / `lokr_w2`
  (no `_a`/`_b` split).

So the LyCORIS key set for one site is a **subset** of
`{lokr_w1, lokr_w1_a, lokr_w1_b, lokr_w2, lokr_w2_a, lokr_w2_b, lokr_t2,
.alpha}`, and which keys are present depends on `decompose_both` and whether
each factor is stored full or low-rank. **`scaling = α/r`** applies exactly as
in LoRA, recovered from the `.alpha` scalar on load.

Consequence for injection: the LoKr site output is still just `base(x) + ΔW·x`
(scaled), an additive `[.., d_out]` tensor — identical injection contract to
LoRA. What changes is only how `ΔW` (or, better, `ΔW·x`) is computed and how the
factors are named on disk.

## Design

Concrete code changes, mapped to real files. The container-type generalization
and the export-trait generalization are the two structural changes; everything
else is additive.

### 1. `LokrDelta` module — `crates/loractl-core/src/lora.rs`

Add `LokrDelta<B>` mirroring `LoraDelta` (`lora.rs:155`): `#[derive(Module)]`,
**base-free** (the base is whatever `Linear` already lives at the site), a
`forward<const D>(input) -> Tensor` returning **only the scaled additive
delta** `[.., d_out]`, and a `scaling: f64` = `α/r`. Factors are burn `Linear`
(or bare `Param` tensors) sized from the `factor` split: `w1` (or `w1_a`/`w1_b`),
`w2` (or `w2_a`/`w2_b`), optional `t2`. Zero-initialize the factor that makes a
fresh delta an **exact no-op** until trained (the LoRA path zero-inits `B`,
`lora.rs:134`; the LoKr analogue zero-inits the outer/last factor so `ΔW = 0` at
attach — preserving the "free attach-integrity check", `mmdit.rs:1314`).

**Materialize-ΔW vs reshape-trick — decision: reshape trick REQUIRED.**

- The naive path (`make_kron` in LyCORIS) forms the full `d_in × d_out` `ΔW`
  in the forward, then `x · ΔWᵀ`. On the ~12.8B MMDiT trunk this materializes a
  full weight-sized tensor *inside the tracked forward graph*, adding a
  weight-class allocation per site to the already-pinned set — directly
  contributing to the 67.9 GiB retention wall ADR-0005 describes. **Rejected.**
- The **required** path computes `ΔW·x` **without ever forming `ΔW`**, using the
  Kronecker mixed-product / reshape identity: reshape the activation into the
  `(b₁, b₂)` block grid, contract against `W₂` then `W₁` (and `t2` when present)
  as small matmuls/reshapes, and reshape back to `[.., d_out]`. This keeps the
  live tensors at activation scale, never weight scale, so LoKr adds **nothing**
  to the activation wall beyond what an equivalent LoRA delta would. This is
  non-negotiable given ADR-0005; the module's doc comment must state it.

All ops are plain matmul/reshape — no custom autodiff op, no nested `backward()`
(which deadlocks on burn 0.21, `.claude/rules/burn-nested-backward-and-param-clone.md`).

`freeze` semantics and lazy-init care follow the existing LoRA pattern; only the
zero-init factor's random draw is discarded, so no `burn-lazy-param-init.md`
eager-materialization concern for the trained factors.

### 2. Polymorphic `LoraAdapters` — `crates/loractl-core/src/adapters.rs`

`LoraAdapters.deltas` is concretely `Vec<LoraDelta<B>>` (`adapters.rs:49`). A
`Vec<Box<dyn Trait>>` is **not** a burn `Module` (the container relies on
`#[derive(Module)]` visiting each element by `ParamId` — the reason it is a
`Vec` and not a `HashMap`, `adapters.rs:15-25`). The burn-compatible route is an
**enum that derives `Module`**:

```rust
#[derive(Module, Debug)]
pub enum AdapterDelta<B: Backend> {
    Lora(LoraDelta<B>),
    Lokr(LokrDelta<B>),
}
```

with `deltas: Vec<AdapterDelta<B>>` and a delegating `forward` /
per-variant param access. `apply` (`adapters.rs:75-85`) and `get`
(`adapters.rs:62`) keep their contract unchanged — the match is
`base_out + delta.forward(x)` regardless of variant, so **every injection site
in `mmdit.rs` (`site`, `mmdit.rs:519`; `forward_with_adapters`,
`mmdit.rs:1317`) is untouched.**

> **burn-store gotcha — `skip_enum_variants(true)`.** A `#[derive(Module)]`
> enum makes `burn-store`'s `SafetensorsStore` inject the **active variant name**
> as a path segment into every key beneath it — so a checkpoint key
> `...deltas.3.lora_a.weight` is looked up as `...deltas.3.Lora.lora_a.weight`
> and silently reported "Unused Tensors" with the tensor left at random init, no
> error (`.claude/rules/burn-store-skip-enum-variants.md`; this is exactly the
> `BaseLinear::Plain/Quant` trap). **Any load/save path that touches
> `AdapterDelta` must add `.skip_enum_variants(true)` to the store builder**, and
> the load must be verified against `tests/fixtures/tiny-krea2` to report **no
> "Unused Tensors"**. Note: adapter I/O today goes through the kohya-format
> `export.rs` path (per-tensor, not a whole-module `SafetensorsStore`), so the
> primary risk surface is any *module-level* snapshot of `LoraAdapters` — flag it
> loudly in the enum's doc comment.

### 3. Config — `crates/loractl-core/src/config.rs`

Extend `LoraConfig` (`config.rs:166`) and `TargetSpec` (`config.rs:198`), all
`#[serde(default)]` so existing YAML keeps parsing bit-identically:

- `algo: AdapterAlgo` — enum `{ Lora, Lokr }`, default `Lora`.
- `factor: i32` — LoKr Kronecker split factor (LyCORIS convention: `-1` =
  auto/largest-square split; positive = target block size). Ignored by LoRA.
- `decompose_both: bool` — when true, low-rank-decompose `W₁` as well as `W₂`.
- Per-`TargetSpec` overrides of `algo`/`factor`/`decompose_both`
  (`Option<...>`, falling back to the globals, matching the existing
  `rank`/`alpha` override pattern, `adapters.rs:135-136`).

`rank` / `alpha` / `dropout` keep their meaning (rank of the decomposed factors;
`α` for `scaling = α/r`).

### 4. `build_adapters` — `crates/loractl-core/src/adapters.rs:116`

The factory currently hard-codes `LoraDelta::new` (`adapters.rs`). Branch on the
resolved `algo` per matched site to construct either
`AdapterDelta::Lora(LoraDelta::new(...))` or
`AdapterDelta::Lokr(LokrDelta::new(d_in, d_out, rank, alpha, factor,
decompose_both, dropout, device))`. Registration order still follows `sites`, so
`deltas` / `targets` stay aligned. The `(d_in, d_out, rank, alpha, dropout,
device)` sizing contract is preserved and extended with the two LoKr knobs.

### 5. Export — `crates/loractl-core/src/export.rs`

`AdapterNameMapper` (`export.rs:45`) is a **three-key** trait (`down_key`,
`up_key`, `alpha_key`) — it assumes the LoRA down/up/alpha triple. LoKr emits a
**variable key set** (`lokr_w1` or `lokr_w1_a`+`lokr_w1_b`; `lokr_w2` or
`lokr_w2_a`+`lokr_w2_b`; optional `lokr_t2`; `.alpha`). Generalize the trait to
emit an **arbitrary `(suffix, tensor)` set per site** — e.g. replace the three
fixed methods with a `keys(path) -> Vec<(String /*suffix*/, ...)>` shape, or add
LoKr-specific methods (`lokr_w1_key`, `lokr_w2_a_key`, ... , `t2_key`) — while
keeping the `KohyaMapper` prefix rule (`lora_<path dots→underscores>`,
`export.rs:57`) and the `Krea2DiffusersMapper` path translation
(`export.rs:93-133`) intact for the shared `.alpha` and prefix construction.

`export_adapters` (`export.rs:224`) currently writes exactly down/up/alpha per
delta with the burn `[d_in, d_out]` → loader `[out, in]` transpose
(`export.rs:242-247`). Generalize to dispatch on the `AdapterDelta` variant:
LoRA writes its existing triple; LoKr writes its present factors with the
transpose/layout LyCORIS/ComfyUI expects, plus the `.alpha = scaling · r`
scalar. `import_adapters` (`export.rs:272`, the resume path) gains the inverse
for LoKr. The self-golden export test (`tests/adapter_export.rs`) pins our
convention; the consumer-contract test (below) pins ComfyUI's.

### 6. Block checkpointing — `crates/loractl-core/src/block_ckpt.rs`

`track_adapters` (`block_ckpt.rs:81`) re-marks `require_grad` on
`delta.lora_a.weight` and `delta.lora_b.weight`, preserving `ParamId` via
`Param::initialized(weight.id, ...)` — the burn 0.21 `Param::clone`-drops-
`require_grad` workaround (`.claude/rules/burn-nested-backward-and-param-clone.md`,
§2). This loop **hard-codes the two LoRA weights**. For `AdapterDelta::Lokr`,
**every LoKr factor** (`w1`/`w1_a`/`w1_b`, `w2`/`w2_a`/`w2_b`, `t2` — whichever
are present) must be re-marked the same way, preserving each factor's id.
Symmetrically, the per-block grad-collection loop (`block_ckpt.rs:192-200`) must
enumerate and `GradientsParams`-collect every LoKr factor, or block-checkpointed
LoKr training **silently trains nothing** (forward fine, zero grads, optimizer
skips them — the exact failure that `tests/block_ckpt.rs`'s completeness
assertions caught for LoRA). Add **completeness teeth** for LoKr: assert the
expected factor-count of grads present and **non-zero**, not a value-only
comparison (which passes vacuously on missing grads).

### 7. `mmdit.rs` — no change

The `site` helper (`mmdit.rs:519`) and `forward_with_adapters` /
`forward_capture` (`mmdit.rs:1317`, `1467`) consume `Option<&LoraAdapters<B>>`
through `apply`'s additive contract. Because `AdapterDelta::forward` returns the
same `[.., d_out]` additive delta, no forward code changes.

## Interop requirement

The exported LoKr adapter **must load in ComfyUI** (the `lokr` branch of
`comfy/lora.py`) and visibly condition generation — the same bar as the M6 LoRA
export and the M14 #25 real-run A/B. A self-golden (`tests/adapter_export.rs`)
pins *our* key convention and, by construction, cannot tell us whether ComfyUI
*accepts* those keys — and an unmatched-key adapter loads **without error and
does nothing** (`.claude/rules/testing.md`; the #137 misdiagnosis).

Therefore, mirroring `tests/krea2_lora_keys.rs`, add a **consumer-contract
test** for LoKr:

- **Generate the contract from pinned upstream source, never a hand-copy.** A
  new `reference/comfyui_lokr_keys_reference.py` downloads `comfy/lora.py` (and
  any helper it needs) at a **pinned commit**, extracts the real `lokr`-handler
  key set via `ast` (the literal `lokr_w1`, `lokr_w1_a`, `lokr_w1_b`,
  `lokr_w2`, `lokr_w2_a`, `lokr_w2_b`, `lokr_t2`, `alpha` names the handler
  reads), and **asserts the specific key lines the export depends on are still
  present** before emitting a golden. Add a `just comfyui-lokr-keys-reference`
  recipe; bump the commit deliberately. A transcribed map drifts silently; a
  pinned-source generator fails loud when the contract moves.
- **Run the real export path over the real site enumeration.** `build_adapters`
  is config-derived, so the full Krea 2 site set builds from
  `MmditConfig::krea2()` **without instantiating the ~12.8B model** — offline,
  seconds. Assert **every on-disk LoKr key** the export writes is one the
  ComfyUI `lokr` handler actually reads.
- **Give the contract teeth (kill-test).** Pin that a deliberately *wrong* key
  schema (e.g. LoRA `lora_up`/`lora_down` names emitted for a LoKr delta, or a
  mangled `lokr_w2` → `lokr_w9`) is **not** accepted, so the assertion cannot
  pass vacuously; and prove that sabotaging the LoKr key mapper to a wrong/
  identity form **fails** the test.

For a first landing, at minimum the ComfyUI-native LoKr key form must be proven
loadable; the Krea2/diffusers path-translation for LoKr should reuse
`Krea2DiffusersMapper`'s existing site→diffusers table (`export.rs:104-113`) for
the prefix and be covered by the same contract test over the Krea 2 sites.

## Testing

RED → GREEN, per `.claude/rules/testing.md` and `development.md`:

1. **Numerics golden (RED first).** Add `tests/lokr_reference.rs` pinning
   `LokrDelta::forward` against a **LyCORIS/PyTorch reference**: a new
   `reference/lokr_reference.py` (torch via `uv`, mirroring
   `reference/*_reference.py`) builds a tiny LoKr module with fixed factors,
   dumps the factors + input + reference `ΔW·x`, and the Rust test loads them,
   runs the reshape-trick forward, and asserts bit-parity (to the established
   golden tolerance). This proves the Kronecker/reshape math *and* the
   scaling = α/r convention. Add a `just lokr-reference` recipe.
   - **Kill-test:** the golden must be sensitive — swapping `w1`/`w2` order,
     dropping `t2`, or using materialize-ΔW-with-a-transpose-bug must make the
     test **fail**. A parity test that passes under a broken forward is worthless.
   - **No-op-at-attach test:** a freshly built `LokrDelta` (zero-init factor)
     makes `forward_with_adapters` bit-identical to the plain forward
     (`mmdit.rs:1314`'s attach-integrity property).
2. **Export round-trip.** `tests/adapter_export.rs` gains a LoKr case: export →
   `import_adapters` recovers the factors and `alpha`, shapes checked.
3. **Consumer-contract test + kill-test.** As in Interop above
   (`tests/comfyui_lokr_keys.rs`), generated from pinned upstream, with the
   wrong-schema-rejected and sabotage-fails teeth.
4. **Block-checkpointing completeness teeth.** `tests/block_ckpt.rs` gains a
   LoKr case asserting **all** LoKr factors receive non-zero grads under
   `checkpointed_step`, and that the monolithic and checkpointed grads are
   bit-identical on the tiny fixture (the #134 property). Value-only comparison
   is insufficient — assert grad **presence and count** (the missing-grad
   failure mode of `Param::clone`).
5. **Config layering.** `algo`/`factor`/`decompose_both` obey the YAML → env →
   flag precedence and per-`TargetSpec` override, including a flag beating an env
   var beating the file (`load_config`). Assert a non-default `algo` **changes**
   which delta variant `build_adapters` constructs (kill-test the wiring, not its
   presence — `.claude/rules/burn-optimizer-and-dropout.md`'s dead-config
   lesson).
6. **Lint/format gate.** `just fmt-check && just lint` (clippy warnings-as-
   errors) green; the enum + generalized trait must not regress the offline
   default features. If a new opt-in reference path lands, mirror it in a
   feature-lint recipe as CI expects.

## Rollout / effort estimate

Land behind the default-`Lora` `algo` so every existing config and golden is
byte-for-byte unchanged (LoKr is purely additive surface). Suggested sequence:

1. **Container generalization (`AdapterDelta` enum) + `skip_enum_variants`
   audit** — the riskiest structural change; land it first with LoRA-only
   variants so all existing tests stay green (proves the enum is transparent).
   ~1–2 days incl. the burn-store verification against `tiny-krea2`.
2. **`LokrDelta` + numerics golden (RED → GREEN)** — the reshape-trick forward
   and its LyCORIS parity. ~2–3 days (the reshape/factor-split math is the
   subtle part; the reference script and kill-tests included).
3. **Config + `build_adapters` branch** — ~0.5 day.
4. **Export generalization + consumer-contract test (pinned upstream +
   kill-test)** — ~1–2 days (the `AdapterNameMapper` trait reshape plus the new
   `reference/comfyui_lokr_keys_reference.py` and `just` recipe).
5. **`block_ckpt` factor enumeration + completeness teeth** — ~0.5–1 day.
6. **Real ComfyUI A/B** — export a LoKr trained through `DiffusionTrainer`,
   confirm it loads and visibly conditions generation (rides on the #25 fit,
   which must be a confirmed zero-panic `just step-probe` run *first*).

**Total: ~1–1.5 weeks** of core work plus the real-run A/B, which is gated on
the M14 fit being confirmed (not on LoKr). No trainer, routing, or front-end
changes (the event abstraction holds; `select_trainer` untouched).

## Open questions

1. **Krea 2 diffusers key form for LoKr.** ComfyUI accepts both the bare
   diffusers key we emit and the native `diffusion_model.blocks.N.*` form for
   LoRA (M6). Does the same dual-acceptance hold for the `lokr_*` family on the
   Krea 2 MMDiT sites, or does the `lokr` handler key-match differently? The
   consumer-contract test must settle this from upstream source, not inference.
2. **`factor` / split policy for the MMDiT projection dims.** The attention and
   SwiGLU projection sizes (`attn.wq/wk/wv/wo`, `mlp.gate/up/down`) determine the
   `d_out = a₁·a₂`, `d_in = b₁·b₂` splits. Do we adopt LyCORIS's exact
   largest-divisor-≤-`factor` rule verbatim (safest for interop), and what
   default `factor` do the `config/examples/krea2-*.yaml` presets ship?
3. **`t2` / Tucker path scope.** For linear (non-conv) sites, is `lokr_t2` ever
   emitted by real LyCORIS LoKr configs we want to interop with, or can the first
   landing target only the `w1 (⊗) w2` / `w2 = w2_a·w2_b` subset and reject
   `t2`-bearing configs loudly? (Scope-narrowing is fine if the contract test
   pins the rejection.)
4. **`decompose_both` default.** LyCORIS defaults `W₁` full-rank; do we mirror
   that (`decompose_both: false`) and is the resulting file still what community
   ComfyUI LoKr tooling expects?
5. **Adapter-quality vs int4 base.** Independent of format: does LoKr's higher
   effective rank interact with the int4-quantized frozen base's ~7% worst-case
   dequant error differently than LoRA (the open #25 quality question)? A/B is
   the only way to know; out of scope to answer here, in scope to note.
6. **burn 0.22 migration.** The `AdapterDelta` enum + `track_adapters`
   `require_grad` workaround are both burn-0.21-specific; PR #5045 fixes the
   `Param::clone` drop on `main`. Confirm the enum + `skip_enum_variants`
   approach survives the backend-erased-`Tensor` rework (#79) or plan the port.
