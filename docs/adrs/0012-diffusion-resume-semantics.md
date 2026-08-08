---
id: ADR-0012
status: Accepted
date: 2026-08-07
---

# 0012 — Diffusion resume: `steps` is a total, provenance is read, and the optimizer state lives in a sidecar

- **Status:** Accepted
- **Date:** 2026-08-07
- **Milestones:** post-M15;
  [#180](https://github.com/laurigates/loractl/issues/180)
- **Deciders:** loractl maintainers
- **Builds on:** [ADR-0002](0002-adapter-format-and-sample-semantics.md) (the
  adapter is a real, interoperable artifact — which is why the optimizer state
  cannot go inside it), and the `__metadata__` contract added for
  [#154](https://github.com/laurigates/loractl/issues/154)
  (`crates/loractl-core/src/metadata.rs`), whose `ss_*` keys this ADR promotes
  from *provenance a consumer displays* to *provenance loractl itself reads*.

## Context

Resume existed on the diffusion path before #180, but nobody had decided it.
Its whole specification was an `if` in `DiffusionTrainer`: if
`output.dir/output.name.safetensors` exists, import its factors and carry on.
Three consequences, none of them chosen:

1. **The trigger was the *side effect* of a previous run**, not an input to
   this one. There was no way to say "resume *that* file", and no way to say
   "don't" short of moving the directory.
2. **The step counter restarted at 1.** A 200-step artifact re-run with
   `steps: 500` trained 500 more steps, exported a header claiming 500, and
   emitted `step` events numbered 1..=500 — while `Started.total_steps`, sent
   before the file was even opened, said 500 too. Every number was internally
   consistent and none of them described the adapter.
3. **AdamW re-warmed from zero.** The moments and the bias-correction step
   count are not in the kohya export (they are not something ComfyUI reads), so
   a resumed run spent its first tens of steps rebuilding an optimizer state it
   had already paid for. This is the *lossy* half, and it is invisible: the run
   trains, the loss is noise-dominated by construction (M14), and nothing looks
   wrong.

Meanwhile the repo had already grown the thing that makes a *decided* resume
possible: #154's `__metadata__` header records `ss_steps`,
`ss_max_train_steps`, and `ss_training_finished_at` on every export. Resume was
still keying on the file's mere existence while the file itself was carrying
the answer.

## Decision 1 — `resume:` is a top-level config block, not a field on `output:`

`resume: { from, auto, allow_unfinished }`
(`crates/loractl-core/src/config.rs`), with `--resume <file>` / `--no-resume` /
`--resume-unfinished` layered on top by post-extraction mutation, per the house
config-layering pattern (YAML → `LORACTL_` env → flags).

Rejected: `output.resume_from`. Resume is an **input** to a run; `OutputConfig`
describes what a run *writes*, and the source is frequently in another
directory (the recovery workflow resumes `run-3/checkpoint-50.safetensors` into
a clean `output.dir`). Mechanically it is also the cheaper of the two: adding a
field to `OutputConfig` would have broken 24 exhaustive struct literals in the
test suite against `TrainConfig`'s 2.

Consequence handled at the API boundary: `resume.from` is a *read* path
arriving over `POST /runs`, and `output.dir` was already confined while this
was not. `crates/loractl-api/src/paths.rs::confine_resume` closes it.
`output.name`'s single-component rule deliberately does **not** apply — naming
a sibling run's checkpoint is the workflow, not an escape.

## Decision 2 — `steps` is a **total**, not a remainder

An artifact recording 200 steps, resumed with `steps: 500`, executes
201..=500.

The alternative — `steps` meaning "this many more" — was rejected on a wire
argument rather than a taste one. `Started.total_steps` is emitted **before**
the resume source is opened, so under remainder semantics an SSE consumer would
be told 500 and then watch a run that means something else, with no event able
to correct it without a new field. Total semantics also match what
`ss_max_train_steps` has meant in every export this repo has ever written, so
one number can serve both.

Two consequences, both taken deliberately:

- **Re-running an already-complete config trains zero further steps.** That is
  a behaviour change from the pre-#180 "train `steps` more". It is idempotent
  rather than an error, the final export is re-written, and the resume
  `Warning` names both numbers.
- **A resumed run emits fewer `step` events than `started.total_steps`.**
  Documented in `docs/api/events.md`; a progress bar that assigns rather than
  increments (already the documented contract) is unaffected.

## Decision 3 — the implicit trigger is kept, as `auto: true`

Removing it was on the table: the whole complaint in #180 is that resume used
to be implicit. It is kept because the surprise it causes is recoverable in one
flag (`--no-resume`) while removing it silently strands every workflow that
re-runs a config to extend an adapter — those runs would start over from
random factors and *look* fine. The fix for "implicit" is that it is now
**named and documented**, not that it is gone.

## Decision 4 — provenance has three states, and only the `auto` path is policed

`ResumeProvenance` (`crates/loractl-core/src/resume.rs`) reads the header into
**finished** (`ss_training_finished_at` present), **unfinished**, and
**unrecorded**.

- Auto + unfinished is a hard `bail!` quoting the path, `ss_steps`,
  `ss_max_train_steps`, and `resume.allow_unfinished`. The final export is only
  ever written on a completed run, so an unfinished file *at that path* means a
  checkpoint was copied over it or a write was interrupted — and the cost of
  guessing wrong is the multi-hour run that follows. A warning scrolls past;
  this does not.
- Explicit `resume.from` is **never** subject to it. Every
  `checkpoint-N.safetensors` is unfinished by construction, and naming one is
  the statement of intent that the check exists to ask for.
- **Unrecorded is not unfinished.** `metadata.embed: false` (`--no-metadata`)
  writes no header at all, and neither does a third-party file. Folding that
  into the unfinished branch would refuse every `--no-metadata` user's resume;
  instead the run resumes and the `Warning` says the step count could not be
  restored.

A second, independent guard covers *content* rather than provenance:
`import_adapters` refuses a non-finite factor by tensor name, mirroring
`adapter.rs`'s `all_finite`. Provenance and corruption are different failures,
and the escape hatch for one must not wave the other through.

## Decision 5 — optimizer state in a sidecar, keyed by **site path**

`{name}.optim.safetensors` beside the *source*, holding `moment_1`,
`moment_2`, **and** AdamW's `time`, plus the f16 loss scale and clean streak.

Rejected — **inside the kohya export**: ADR-0002 made that file an interop
artifact, and #137/#154 made its key set and header something pinned against
consumers' actual code. Optimizer moments have no consumer out there; adding
them would roughly double the artifact for tensors no reader wants, and any
reader that *did* pattern-match them would be matching something we invented.
The proof that this held: `tests/krea2_lora_keys.rs`,
`tests/adapter_export.rs`, `tests/adapter_roundtrip.rs` and the metadata-key
tests are untouched by #180.

Rejected — **burn's `Recorder` dump of `Optimizer::Record`**: that map is
keyed by `ParamId`, which burn 0.21 generates **randomly per construction**. A
`ParamId`-keyed dump therefore restores *nothing* in the next process, with no
error — the exact "wrote a file, read it back, silently meant nothing" shape
this repo keeps running into. Keying by the LoRA site path is the only key that
survives a process boundary. (Recorded as a trap in
`.claude/rules/burn-optimizer-and-dropout.md` §3, along with the fact that the
record's map is `hashbrown`'s, whose type error reads as if it were `std`'s.)

`time` is persisted explicitly because it is not implied by the moments and
dropping it changes the very next update's magnitude. Key suffixes
(`.lora_down.exp_avg`, …) are composed in one function so that no key can end
in `.weight`/`.alpha` and be matched by a LoRA loader's key map if the file is
ever dropped into `models/loras/`.

The naming is the one point worth revisiting: `{name}.optim.safetensors` will
appear in a ComfyUI LoRA dropdown if a user copies the whole output directory.
Mitigated by a `loractl_artifact_kind = "optimizer-state"` header key and by
unmatched key suffixes. The alternative — an `optim/` subdirectory — was
rejected because it splits one checkpoint across two paths.

## Decision 6 — a sidecar whose provenance disagrees is **skipped and named**, not fatal

The sidecar records `loractl_steps_done`; the adapter records `ss_steps`. A
disagreement means the two files come from different runs — which the
*documented recovery workflow produces*: copy `checkpoint-2.safetensors` over
the final export and the step-4 sidecar is still sitting beside it. Shapes
match in that case, so no other check can see it; without this one, weights
from step 2 get step-4 moments and a step-4 bias correction, with no error.

Erroring was rejected: re-warming AdamW is survivable, and a hard failure would
stop the recovery workflow dead at exactly the moment the operator is already
recovering from something. So the sidecar is skipped, and the resume `Warning`
names the file and both step counts. Not checked where undecidable —
`Unrecorded` provenance has no `ss_steps` to compare against.

## Decision 7 — the announcement reuses `TrainEvent::Warning`

One `Warning` per resumed run, naming the source, the step it continues from,
what **was** restored, and what was **not**. No new variant and no new field,
so `docs/api/events.md`'s golden (`tests/event_json.rs`) is correctly
untouched — the wire contract did not change, only what flows over it.

It carries no machine-readable fields by design: a client that needs the step
number reads it off the `step` events, whose numbering continues.

The clause that must never be dropped is the one about what is *not* restored:
**the RNG stream**. burn 0.21 exposes no save/restore, so a resumed run draws
timesteps and noise from a stream that has not consumed the draws of the steps
it skipped. A resumed run is a **continuation, not a bit-identical replay**,
and the event says so rather than leaving it to be discovered. This is also why
the offline tests assert *step numbering* and *moment divergence* rather than
weight equality against an uninterrupted run — that assertion would be false,
and writing it would have been faking the criterion.

## Decision 8 — the synthetic `BurnTrainer` has no resume

It writes burn-native snapshots rather than the kohya export, `metadata:` is a
no-op on that path (so there is no provenance to read), and burn 0.21's lazy
`Param` init makes seed-based reconstruction of its frozen base
RNG-history-dependent (`.claude/rules/burn-lazy-param-init.md`). Recorded in a
comment at the loop and made *observable* by
`tests/resume.rs::the_synthetic_trainer_does_not_resume`, so "intentional"
cannot decay into "someone added it and nobody noticed".

## Consequences

- A resumed run is honest about being a continuation, and the numbers in the
  header, the events, and the config now all mean the same thing.
- Interop is unchanged: same file, same keys, same header. The extra file is
  ignorable and self-describing.
- The `.optim.safetensors` roughly doubles the bytes written per checkpoint
  (arithmetic, not measured — tens of MB for a LoRA).
- Two things stay unproven on this box and want a GPU/real-run pass: the
  sidecar's write cost at real rank and site count, and resume across a real
  multi-hour Krea 2 run rather than the tiny fixture.

## What would revive this

- **burn gaining RNG save/restore** would make bit-identical replay possible,
  which changes Decision 7's text and lets the test suite assert weight
  equality.
- **A second optimizer** (8-bit Adam was explicitly rejected in M13 as
  pointless for adapter-only state) would need the sidecar's key schema to
  carry which optimizer wrote it; today `ARTIFACT_KIND` records only that it is
  optimizer state.
- **Multi-file / sharded adapters** would break the "sidecar beside the source,
  same stem" rule, which assumes one artifact per run.
