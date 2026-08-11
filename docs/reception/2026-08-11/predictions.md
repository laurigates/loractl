# Reception panel — pre-registration (2026-08-11)

**This file is the control. It is committed before any persona is dispatched
and is not edited afterwards.** The rollup partitions every finding the panel
returns into KNOWN (predicted here), NEW (not predicted), or WRONG (predicted
here and contradicted by the panel).

**The stopping rule: if NEW ≈ 0, the harness added nothing — stop running it.**
The run has earned its cost iff **≥3 verified findings are absent from this
file**.

Venue under test: **r/StableDiffusion** (primary), r/LocalLLaMA (secondary).
Pitch line under test: *"A terminal-native LoRA trainer, in Rust."*

---

## Predictions

Each is stated so the panel can contradict it. `evidence` is what was checked
in the repo before dispatch — these are verified facts about the artifact, not
guesses; the prediction is about **how a stranger reacts to them**.

### P1 — "Krea 2 only" is the single biggest unstated fact
Reviewers will discover that the diffusion path supports **only** Krea 2 — no
SDXL, no Flux, no SD1.5 — and will say it should be stated in the first
paragraph. At least one persona bounces on it.
`evidence:` zero occurrences of sdxl/flux/sd1.5/"stable diffusion" in
README.md; first "Krea 2" mention is line 23, inside the Status blockquote.

### P2 — no visible path to "train on my own images"
The quickstart trains a **synthetic LoRA-MLP demo**. A practitioner will ask
how to point it at a folder of their own images and will not find the answer
where they look.
`evidence:` README.md:61-62 quickstart = `config/examples/lora.yaml`
(synthetic); the kohya-style folder layout is first described at README.md:230.

### P3 — one evidence image, and they will ask for a grid
Exactly one output image exists. The r/SD audience asks for a sample grid
(same seed, with/without LoRA, several prompts) within two comments.
`evidence:` `docs/evidence/` contains only `m14-krea2-dog-interop.jpg`.

### P4 — no competitor comparison anywhere
kohya-ss appears only as a file format and folder convention, never as a tool
being compared. Reviewers ask "why would I use this over kohya / ai-toolkit?"
and the README does not answer it.
`evidence:` all 5 kohya mentions in README.md (lines 44, 155, 230, 277, 456)
are format/layout references.

### P5 — the arithmetic will be run, and LLM authorship inferred
Someone computes ~38k lines of Rust in ~5 weeks by one author and concludes it
is LLM-written. **They are right**, and the repo already says so in its commit
trailers — but not in the README, so it lands as an accusation rather than a
disclosure.
`evidence:` 38,077 tracked Rust LOC, 178 commits over 37 days,
164/178 (92.13%) carry `Co-authored-by: Claude`; README states this nowhere.

### P6 — the README is too long and too internally-numbered
480 lines, dense with milestone and issue identifiers that mean nothing to an
outsider. At least one persona names length or the `M6–M15` / `#110` vocabulary
as its bounce reason.
`evidence:` README.md is 480 lines with 27 `M<n>`/`#<n>` references.

### P7 — cosmetic leaks read as unfinished
The `managed-by-opentofu` topic (internal tooling leaking onto a public repo),
no homepage URL, a gap at ADR-0009, and an empty `docs/prps/`.
`evidence:` `gh repo view` topics include `managed-by-opentofu`,
`homepageUrl` empty; `docs/adrs/` jumps 0008 → 0010; `docs/prps/` holds only
`.gitkeep`.

### P8 — installation is the practical blocker, not the code
There is no crates.io release and no prebuilt binary; every install path is
`git clone` + `cargo install --path`, and the NVIDIA path additionally wants
the CUDA toolkit. For an audience used to `pip install`, this is a bigger
adoption barrier than anything about the trainer itself.
`evidence:` crates.io sparse index returns 404 for loractl (control: ripgrep
returns 200); `gh release view v0.21.0` reports 0 assets; README.md:84-105.

### P9 — "does it actually train?" will be unanswerable from the repo
No loss curve, no before/after sample grid, no third-party reproduction. The
one interop image proves a LoRA *loads and conditions*, not that training
converges to something good. Every persona lists this as unverifiable.
`evidence:` `docs/evidence/` has one file; no loss-curve artifact in the tree.

### P10 — the honest "no cross-tool benchmark" stance will be misread as "no benchmarks"
The repo deliberately refuses to publish a comparable `s/it`, and explains why
(README.md:398-400, roadmap "What these numbers do and do not license"). A
drive-by reader takes the absence as "unbenchmarked" rather than as rigor.
`evidence:` README.md:394-400 states the measured 4.5 s/step and immediately
disclaims cross-tool comparability.

---

## Deliberately not predicted

Left open so the panel has room to be informative:

- **Where** the drive-by actually bounces (`BOUNCE_POINT`). P1/P6 predict *that*
  someone bounces and on what topic; the exact line is the measurement.
- Which specific README claims lack a supporting artifact (persona 4's lane).
- Whether AI authorship shows up as a **defect at a file:line** rather than as
  a style impression (persona 3's lane) — this is the prediction most likely to
  be WRONG, and finding out is the point.
- Anything about felt UX against a real competitor run — out of scope for a
  panel; needs a human running both tools.

## Instrument controls active this run

| Control | Purpose |
|---|---|
| Tier trees staged on disk | tier-0/1 personas physically cannot read ADRs, `CLAUDE.md`, `.claude/`, or the roadmap |
| Seeded contradiction | tier-2 README says `~1.8 s/step`, contradicting roadmap `4482.18 ms`. Persona 4 missing it invalidates its clean findings |
| Competitor READMEs on disk | sd-scripts and ai-toolkit cloned locally; no comparison from memory |
| Drop rate | findings without `file:line` or a quoted string are dropped by the collator; >50% drop discards the persona |
| Mechanical baseline | `scripts/slop-metrics.sh` output passed in as ground truth so personas do not estimate counts |
