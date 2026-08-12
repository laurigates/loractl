# Reception panel — rollup (2026-08-11)

Machine rollup: [`rollup.txt`](rollup.txt). Raw persona reports: [`raw/`](raw/).
Pre-registration: [`predictions.md`](predictions.md) (committed 9e33234, before dispatch).
Finding-by-finding partition: [`partition.tsv`](partition.tsv).

**Verdict: the panel earned its cost.** 12 verified findings absent from the
pre-registration, against a threshold of 3. Every one was checked against the
real repository before landing here; one panel finding was **refuted** by doing
so, and one persona was discarded outright.

---

## 1. What it actually answered

| Question | Answer |
|---|---|
| Is this slop? | **No, and this is now measured rather than argued.** 10.51 test fns/kLOC and 3.46 asserts/test put it in the ripgrep/bat band and ~7× above sd-scripts, ~130× above ai-toolkit. Trivial-only tests 0.25%, the lowest in the set. |
| UX vs others? | **Yes, README-to-README, and it is the most valuable lane.** Four concrete daily-workflow gaps, each cited against the competitor's own docs on disk. |
| Does it work well? | **No.** Unchanged. Needs a third party running it. |
| Performance ballpark? | **No**, and the question stays ill-posed. But see F11: the resolution people actually train at is unmeasured, which is a different and answerable question. |

## 2. Blockers — these decide the reception, in this order

Adoption-blocking for r/StableDiffusion specifically. All verified in code.

1. **No mid-training image preview, and the knob that looks like it errors out.**
   `diffusion_trainer.rs:325-328` *rejects* `output.sample_every > 0` with "the
   diffusion trainer has no sample path". sd-scripts has
   `--sample_every_n_steps`/`--sample_prompts`; ai-toolkit writes samples into
   the run folder. Look-at-a-sample-and-adjust **is** the LoRA training loop for
   this audience. Nothing else on this list matters as much. *(F04)*
2. **Krea 2 is the only real target — and ai-toolkit already trains it.**
   `comparables/ai-toolkit/README.md:35` lists `krea/Krea-2-Raw` among ~40
   models. The pre-registration predicted "Krea 2 only" was the biggest unstated
   fact; the panel sharpened it into something worse — the single supported
   model is not a differentiator. *(F02, the one KNOWN finding)*
3. **One dataset folder.** `DatasetConfig.path` is a single `PathBuf`
   (`config.rs:439-441`). No per-concept subsets, no `num_repeats`, no
   regularization images. Practitioners balance several folders per character
   LoRA; sd-scripts documents this at `config_README-en.md:147`. *(F06)*
4. **AdamW only.** `OptimConfig` (`config.rs:515-520`) exposes `lr` and
   `weight_decay` and nothing else — no optimizer choice, no LR schedule, no
   warmup. `ss_optimizer` is *written as metadata*, which reads as support and
   is not. *(F05)*

Items 1, 3 and 4 are product gaps, not documentation gaps. They are the honest
answer to "what will people demand that the repo cannot supply" — the question
the plan added, and the most actionable thing here.

## 3. One security-adjacent finding

**`POST /runs` confines two paths and documents that it confines all of them.**
`routes.rs:116-124` runs `confine_output` on `output.dir`/`output.name` and
`confine_resume` on `resume.from`. `dataset.path` and
`model.denoiser`/`vae`/`text_encoder`/`tokenizer` go through unvalidated, on an
endpoint that is **unauthenticated by default**. The doc comment at
`routes.rs:100-105` claims "the config the trainer runs with carries the
*resolved* paths, never the client's raw strings, so no later code path can be
handed an unvalidated value by mistake" — which is not true of those fields.

Second pass confirmed no ADR covers it: ADR-0012 Decision 1 settles `resume.from`
and asserts `output.dir` "was already confined", and names the model/dataset
paths nowhere. *(F07 — file an issue)*

## 4. Cheap fixes — all verified, all mechanical

| Fix | Evidence |
|---|---|
| README says "Three crates"; the workspace has four (`loractl-bench`) | `README.md:40` vs `Cargo.toml:3-8` |
| Backend list omits `candle` | `README.md:351` vs `config.rs:635` |
| Precision list omits `bf16` | `README.md:353` vs `config.rs:695` |
| Preset list omits `krea2-comfyui` (config ships) | `README.md:59` vs `cli.rs:173` |
| Comment claims `train()` is "never executed by any test" | `cli.rs:718` vs `tests/verbosity_wiring.rs:42` |
| `LORACTL_SKIP_GRAD_CHECK` is presence-keyed, so `=0` also disables it; undocumented | `diffusion_trainer.rs:1732` |
| Latent-cache correctness rides on a hand-bumped `-t2` literal; reader validates rank, not length | `diffusion_trainer.rs:309` |

Plus the pre-registered cosmetics that no persona needed to find: the
`managed-by-opentofu` topic on a public repo, no homepage URL, the ADR-0009 gap,
the empty `docs/prps/`.

## 5. A correction in the project's favour

The practitioner could not determine whether LoRA **rank/alpha/target modules**
are user-configurable and filed it as unverifiable. They are — `LoraConfig`
(`config.rs:315-335`) carries `rank`, `alpha`, `dropout`, and per-target regex
overrides with their own rank/alpha, which is *more* capable than sd-scripts'
flat `--network_dim`/`--network_alpha`. This is a **docs-placement problem**:
the capability exists and is invisible from where a practitioner looks. It is
the clearest instance of the category the plan wanted separated out.

## 6. What the controls showed — read this before trusting anything above

- **The seeded contradiction was caught.** The auditor's staged README said
  `~1.8 s/step`; it bounced there and traced it to `roadmap.md:314`'s 4482.18 ms
  unprompted. Its clean bill of health on the *real* claims is therefore worth
  something. (The real README says ~4.5 s/step and is correct.)
- **One persona was discarded: the drive-by.** 2 of 3 findings carried no
  `file:line` (67% > the 50% threshold). It also asserted ADR *content* —
  "ADRs describe unsolved VRAM and precision hazards" — having seen only ADR
  *filenames* in a directory listing, and starred the project on that basis.
  This is precisely the "confidently wrong in a way that feels like validation"
  failure the plan flagged as the open risk. The mechanical rule caught it; no
  judgment call was involved.
- **One finding was refuted by measurement, not by argument.** `rust-craft`
  claimed deleting the dropout call at `lora.rs:108` "leaves the whole suite
  green". Applying that exact mutation fails `tests/dropout.rs:117` in 2.37 s —
  a regression test written for this exact hole. The persona had not opened
  that file. It re-asserted the finding on second pass; the mutation outranks it.
- **Calibration: the drive-by does discriminate, but is reputation-contaminated.**
  Same brief on three repos gave three verdicts. On sd-scripts it wrote
  "Recognized as kohya-ss on line 1" and returned all-YES with `BOUNCE_POINT=NONE`
  — deference, not reading. The usable comparison is loractl vs ai-toolkit, both
  unknown to the reader: loractl YES/YES/YES/YES, ai-toolkit YES/NO/NO/NO, and
  the discriminator was a visible quickstart. **loractl's README beats
  ai-toolkit's on first contact.**
- **No convergence.** Every finding is single-source, because the tiers are
  disjoint by construction. Cross-persona agreement is therefore unavailable as
  evidence — which is why each finding was verified against the code by hand
  instead. Treat any unverified finding here as unconfirmed.
- **No contamination.** Zero hits for `CLAUDE.md` / `.claude/rules` / roadmap
  citations in the tier-0/1 reports. The staged trees held.

## 7. Go / no-go

**Do not post to r/StableDiffusion yet — and when you do, consider posting
somewhere else first.**

The panel surfaced a venue mismatch that was not on the original list of four
questions. What is strongest here — the architectural invariant, numerics
verified against pinned upstream sources, the refusal to publish a misleading
cross-tool benchmark, 100% conventional commits, interop contracts generated
rather than hand-copied — are **r/rust and r/LocalLLaMA virtues**. What r/SD
evaluates on is: does it train my model, on my folder of images, with my
optimizer, and can I see a sample while it runs. Today that is four "no"s.

Sequenced:

1. **Before any post**: the §4 fixes (an afternoon), the `managed-by-opentofu`
   topic and homepage URL, and one sentence in the first screen saying Krea 2
   is the only supported image model. Cheapest possible removal of the "didn't
   read the README" class of reply.
2. **Then post to r/rust**, pitched on the architecture and the measurement
   discipline. That audience can evaluate what is actually good here, and the
   four blockers are not disqualifying for them.
3. **Before r/StableDiffusion**: land mid-training sampling (§2.1). It is the
   single highest-leverage item — it converts the tool from a research harness
   into something with a usable daily loop.
4. **The honest alternative to another panel run**: post it somewhere small and
   read real replies. The panel cannot tell you whether it trains well, and
   after this run there is nothing left for it to debug on the README that
   another dispatch would surface.

## 8. Rerun policy

- **`repo-first-contact` (cheap lane)** — `slop-metrics.sh` + comparables +
  drive-by: rerun after each README edit. Note the drive-by needs a stricter
  evidence rule before it is trustworthy alone; this run it self-eliminated.
- **`repo-review-panel` (expensive lane)** — do **not** rerun on loractl
  without a fresh pre-registration. Its README-debugging value is now spent;
  a second run over the same README would score NEW≈0 by construction, which
  is the stopping rule doing its job rather than a failure.
