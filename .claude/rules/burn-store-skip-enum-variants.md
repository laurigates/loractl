# Loading a Module Tree With a `#[derive(Module)]` Enum Field Needs `skip_enum_variants(true)`

When a burn module contains an **enum field** that itself derives `Module`,
`burn-store`'s `SafetensorsStore` injects the *active variant name* as a path
segment into every key under that field. So a site the checkpoint stores as
`blocks.0.attn.wq.weight` is looked up by the module as
`blocks.0.attn.wq.Plain.weight` (or `.Quant.weight`) — the extra `.Plain.` /
`.Quant.` segment means the checkpoint key **silently does not match**, the
load reports those tensors as **"Unused Tensors"**, and the base is left at its
random init. No error is raised; the forward just runs on garbage.

This is the `BaseLinear` enum (`Plain(Linear)` / `Quant(QuantLinear)`, the #96
int8 work) in `src/mmdit.rs`. Any `Mmdit` load path must bridge the segment.

## The fix

Add `.skip_enum_variants(true)` to the store builder, right after `.remap(...)`:

```rust
let mut store = SafetensorsStore::from_file(path)
    .remap(remapper)
    // Bridge BaseLinear's Plain/Quant enum-variant path segment
    // (blocks.0.attn.wq.Plain.weight) so the checkpoint's
    // blocks.0.attn.wq.weight matches.
    .skip_enum_variants(true)
    .with_from_adapter(/* PyTorchToBurnAdapter.chain(CastFloatsAdapter { .. }) */);
```

`diffusion_trainer::load_module` is the canonical site (it always had this).
The trap is anything that hand-rolls its own `SafetensorsStore` builder instead
of going through `load_module`.

## Why it hides

The break is **runtime-only and load-time**: the code compiles, so
`cargo clippy --all-targets` (and CI's `feature-lints`) stay green. It only
surfaces when the example/binary is actually *run* against a real checkpoint —
and the GPU diagnostics that hit it (`grad_compare`, `metal_bisect`,
`trace_f16_forward`) are run **by hand on the GPU box**, never in CI. So the
regression rode silently from whenever `BaseLinear` became an enum until a run
turned up "Unused Tensors". These are the exact examples
`burn-wgpu-metal-numerics.md` says to reach for *first* on any wgpu anomaly, so
a silent load break there is especially costly.

Fixed 2026-07-18 (#116): `grad_compare` / `metal_bisect` / `trace_f16_forward`
each gained the one line; `quant_grad_compare` (#111) shipped with it.

## Check

After adding any new `SafetensorsStore`-based `Mmdit` load path, run it against
`tests/fixtures/tiny-krea2` and confirm the load reports **no "Unused
Tensors"** and 92/92 tensors present (`metal_bisect verify-load` prints
`tensors compared: 92  corrupted: 0`). A green compile proves nothing here.

## Rationale

`skip_enum_variants` exists precisely because a `Module`-deriving enum is a
container the checkpoint's flat key namespace has no notion of. Forgetting it
is a *silent* correctness bug — random-init base, plausible-looking run — not a
loud failure, which is why it needs a rule and a load-time check rather than
trust in the compiler. Same family as `burn-lazy-param-init.md`: a burn-store
mechanism whose omission is invisible until the values are wrong.
