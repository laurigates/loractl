---
summary: The outward-facing LoRA contract — which sites get adapters, and the exact on-disk layout consumers read.
anchors:
  - claim: >
      Adapter placement is config-derived, not hard-coded: `build_adapters` walks a model's
      site enumeration, attaches a LoraDelta to each site whose path matches a `lora.targets`
      regex (per-target rank/alpha overriding the global), and leaves unmatched sites bare.
      Registration order follows the site enumeration, so `deltas` and `targets` stay aligned
      with it — export keys are derived from that same order.
    at:
      - crates/loractl-core/src/adapters.rs > build_adapters
      - crates/loractl-core/src/adapters.rs > LoraAdapters
    hash: 2:2999cf06317f
    id: c_18c63b4af10032680001
    verified_at: 2026-07-27T19:11:35Z
    verified_commit: 069d768374f09c4a8bfe9a8bcbb75ee863132a88
  - claim: >
      The export is the interop boundary and its layout is load-bearing: burn's `A` is
      [d_in, rank] and `B` is [rank, d_out], both transposed on the way out to kohya's
      [rank, d_in] / [d_out, rank]; each site also writes an `.alpha` scalar [1] tensor
      reconstructed as `scaling * rank`. `import_adapters` reverses exactly this, so the
      two must change together or a round-trip silently degrades.
    at:
      - crates/loractl-core/src/export.rs > export_adapters
      - crates/loractl-core/src/export.rs > import_adapters
    hash: 2:7569189f1def
    id: c_18c63b4af1e172780002
    verified_at: 2026-08-26T11:54:29Z
    verified_commit: 649acfffdace4b49b9eaf50839bba3cf22bf909e
refs: []
---

# LoRA injection and interop

The LoRA math is `base(x) + (alpha/rank) · B(A(x))`: the base is frozen, only the
low-rank factors train. Which *sites* get that treatment is a config decision resolved
by `build_adapters`, which is why the Krea 2 196-site set can be enumerated offline
without instantiating the ~12.8B model.

The export exists so a trained adapter loads in ComfyUI/Krea with no key conversion.
Two independent things must hold, and they fail differently:

- **The keys must be ones the consumer actually looks up.** A LoRA with unmatched keys
  loads *without error* and does nothing — the worst failure shape. That is pinned
  separately by `tests/krea2_lora_keys.rs` against a golden generated from ComfyUI's own
  key map at a pinned commit, not against our own convention.
- **The tensor layout must match.** That is what the anchor above guards: a dropped or
  flipped transpose is a logic change in `export_adapters`, and this gate fires.

**Boundary.** Surface guards the *code* these claims point at. It cannot tell you the
consumer still accepts the keys — that is the consumer-contract test's job, and the two
are complementary, not redundant.
