# loractl brand assets

The `loractl` mark is a **low-rank bottleneck**: a wide signal projected *down*
to a thin rank-*r* spine (matrix **A**, ember) and back *up* to wide (matrix
**B**, teal) — the LoRA update `base(x) + (α/r)·B(A(x))` drawn as one glyph. The
rank spine is the negative space between the two halves, so the mark works on
any ground and in a single colour. The silhouette also reads as a control/flow
glyph, fitting a `*ctl` tool.

## Files

| File | Use |
|---|---|
| `mark.svg` | The two-tone mark. Vector source of truth — scale it anywhere. |
| `mark-mono.svg` | Single-colour mark (`currentColor`). For badges, embroidery, one-ink print. |
| `header-light.png` / `header-dark.png` | Wordmark lockup, transparent. Light = dark text (light backgrounds), dark = light text (dark backgrounds). Swap with `<picture>`. |
| `social-card.png` | 1280×640 repo social preview / link unfurl card. |
| `avatar-512.png` | Square org/repo avatar (mark on a dark tile). |
| `favicon-16/32/48.png` | Transparent favicons. |
| `apple-touch-icon-180.png`, `icon-192.png`, `icon-512.png` | Filled app icons, ready for an `apple-touch-icon` link or a web manifest. Nothing references them yet — wiring a manifest is a follow-up. |

## Palette

| Token | Hex | Role |
|---|---|---|
| ember | `#E4572E` | down-projection **A**; primary accent (Rust heat) |
| ember-deep | `#C1440E` | ember on light grounds / gradients |
| teal | `#0EA5A4` | up-projection **B**; secondary accent |
| teal-deep | `#0B7C7B` | `ctl` on light grounds |
| teal-lift | `#34C7C0` | `ctl` on dark grounds |
| ink (dark) | `#17140F` | warm near-black ground |
| ink (light) | `#F1EEE9` | warm off-white on dark grounds |

Only `ember` and `teal` appear in the `.svg` sources; the `-deep` / `-lift`
variants are used solely by the raster lockup and social-card layouts (gradients
and light/dark `ctl` tinting), which is why they aren't in the vector files.

**Type:** the wordmark is set in a monospace (system stack: SF Mono / Cascadia
Code / JetBrains Mono / Menlo) at weight 600, `-0.02em` tracking, with `ctl`
tinted teal to echo the up-projection and flag the `*ctl` lineage.

## Usage

- Keep clearspace around the mark equal to the width of one triangle.
- Don't recolour the two halves, rotate the mark, or add effects.
- On busy photography, use `mark-mono.svg` knocked out of a solid ember or dark tile.
- The **social preview** is not a repo file — upload `social-card.png` under
  **Settings → General → Social preview** on GitHub.

## Regenerating

The `.svg` files are the sources of truth; the `.png` rasters are rendered from
the mark + HTML lockup/card layouts via headless Chromium at the sizes in the
table above. Re-cut them whenever the mark or wordmark changes.
