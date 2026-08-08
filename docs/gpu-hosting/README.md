# Hosted GPU providers for loractl — cost study

**Prices collected 2026-08-08.** Data files in this directory: [`providers.csv`](providers.csv)
(50 providers), [`gpu-instances.csv`](gpu-instances.csv) (221 SKUs, 9 of them deliberately
price-null), [`scenarios.csv`](scenarios.csv) (179 costed rows — 113 from the study proper plus
66 added post-hoc for the 24–32 GB tier, see the correction below).

## The decision

loractl's real-GPU proofs currently run on a self-hosted RTX 4090 (24 GB) in the owner's
apartment, dispatched via `gpu.yml` to a runner that is registered once and lives forever. The
goal is to move that to a hosted provider: cheaper, faster, and it stops heating the apartment.
[ADR-0005](../adrs/0005-int4-training-vram-bound.md) established that int4 LoRA training on the
~12.8B Krea 2 denoiser is VRAM-bound and rode the 24 GB ceiling until it OOMed. **That premise was
retired before this study ran — see the correction immediately below.**

> ### ⚠️ Correction: the 24 GB tier is not dead, and this study was scoped as if it were
>
> The research briefs for this study asserted "24 GB is NOT enough; 48 GB is the first tier that
> plausibly unblocks it." **That was stale.** [ADR-0005 Addendum 3](../adrs/0005-int4-training-vram-bound.md)
> (2026-07-25, #134 / PR #135) records that manual per-block gradient checkpointing landed and the
> int4 512 px step now measures **19.4 GB peak device memory on the 24 GB RTX 4090 — zero panics,
> 3/3 steps, 196/196 sites**, with ~4 GB of headroom. The step that could not fit, fits.
>
> Consequently the 24–32 GB tier was excluded from the provider research and from the original 113
> cost rows. It has since been costed post-hoc from the same verified price data using the same
> model, and appended to [`scenarios.csv`](scenarios.csv) (66 rows, tagged
> `[ADDED POST-HOC: 24-32GB tier]` in their `assumptions` field). Those rows are **not** independently
> source-verified beyond the price verification the SKUs already carried.
>
> **What it changes:** a cheaper floor exists. RunPod Community RTX A5000 24 GB is **$17.69/month**
> for all three scenarios combined, against $42.51 for the recommended 48 GB A40 — 58% less.
>
> **What it does not change: the recommendation stands, for a reason worth stating.** On RunPod's
> *vetted* Secure Cloud the 48 GB A40 costs **$0.44/hr while the 24 GB RTX 4090 costs $0.69/hr** —
> the bigger card is the cheaper card. So on trusted first-party hardware there is no 24 GB saving
> to capture, and no reason to run at 4 GB of headroom instead of 28. The 24 GB tier only wins on
> Community Cloud (third-party hosts) or on Salad/Vast.ai, both of which fail the persistent-volume
> requirement outright.
>
> **Caveats on the fit itself, from the ADR:** the 19.4 GB figure was measured on an Ada RTX 4090,
> not on the Ampere A5000/3090 that make the cheap rows cheap; the ADR flags its own provenance as
> unpinned (the raw `STEP_PROBE_SUMMARY` line was never captured, so GB-vs-GiB is unresolved); and
> the fit gate is *zero panics*, which a different card, driver or resolution could fail. Treat
> 24 GB as proven on a 4090 and unproven elsewhere.

48 GB is therefore a comfort margin rather than a requirement; 80 GB+ removes the constraint
entirely and buys throughput headroom this study does not measure.

**Three hard requirements**, pass/fail:

1. **Per-second or per-minute billing** — no minimum commitment, no whole-hour rounding.
2. **API/Terraform provisioning** — a GitHub Actions job can create and destroy a machine.
3. **Persistent volume (or equivalent fast cache)** — the ~20 GB of weights (13 GB denoiser,
   4 GB text encoder, VAE) must survive between runs.

**EU data residency is a bonus, not a gate.** It breaks ties and it matters for the training
scenario, where private model weights sit on someone else's hardware for hours. This study prices
it explicitly rather than filtering on it.

---

## Recommendation

### Primary: RunPod Secure Cloud, EU datacenter

RunPod is the only provider in this study that passes all three hard requirements with
**source-verified prices**, in a **named EU member-state datacenter**, across the whole VRAM
ladder from a $0.44/hr A40 to a $1.99/hr 96 GB RTX PRO 6000 — so escalating tiers is a SKU string
change, not a migration. Six EU datacenters are confirmed from RunPod's own docs (Czechia, France,
Netherlands, Romania, Sweden, plus Iceland which is EEA not EU). Network volumes survive pod
termination at $0.07/GB-month, billing is per-second on compute, and there is a real REST API, a
CLI, and a Terraform provider.

### Fallback: Verda (formerly DataCrunch), Helsinki

The most solidly verified provider here — every on-demand price matched the live page to the cent,
and all three datacenters are in Finland, so a Verda GPU is EU-resident by construction rather than
by region selection. The 96 GB RTX PRO 6000 at $1.89/hr and the H100 at $3.25/hr are the cheapest
source-verified EU options at those tiers anywhere in this study, and spot is a flat ~50% off. Its
two weaknesses are why it is the fallback and not the primary: billing is **10-minute prepaid
blocks** (refunded if you terminate early, but not per-second), and there is an unresolved
report on Verda's own forum that volumes attached to *pre-empted spot* instances get **deleted**
rather than detached — which is precisely the property the weight cache depends on.

### What the recommended setup costs

All three scenarios in one month on RunPod Secure Cloud in an EU datacenter, sharing one 50 GB
network volume:

| Setup | Monthly |
|---|---|
| All-48 GB (A40 @ $0.44/hr) — 88.67 GPU-hours + $3.50 volume | **$42.51** |
| 48 GB for CI + dev, 96 GB RTX PRO 6000 @ $1.99/hr for the 60 h training run | **$135.51** |
| All-96 GB (RTX PRO 6000 @ $1.99/hr) | **$179.95** |

For comparison, the same three scenarios on RunPod **Community** Cloud (third-party hosts, EU) with
an RTX A6000 at $0.33/hr come to **$32.76/month**, and on AWS `g6e.xlarge` in Spain to
**$179.37/month** (89.33 h at $1.961 + $4.18 gp3) — roughly **4.2×** the recommendation for the
same 48 GB of VRAM.

### Cheapest option per scenario

These are not the same provider, and the cheapest overall is not EU-resident:

| Scenario | Cheapest overall | Cheapest EU-resident | Cheapest EU on vetted datacenter hardware |
|---|---|---|---|
| **ci-smoke** (40 × 12 min) | Thunder Compute A6000 — **$5.77** (US only) | RunPod Community A6000 — **$6.36** | RunPod Secure A40 — **$7.31** |
| **interactive-dev** (20 h) | Thunder Compute A6000 — **$9.50** (US only) | RunPod Community A6000 — **$10.10** | RunPod Secure A40 — **$12.30** |
| **training** (60 h) | RunPod Community A6000 — **$23.30** (already EU) | RunPod Community A6000 — **$23.30** | RunPod Secure A40 — **$29.90** |

---

## What EU residency costs

**Very little, and on the training scenario nothing at all.** This is the headline finding of the
study and it was not the expected one.

| Scenario | Cheapest overall | Cheapest EU-resident | Delta | % |
|---|---|---|---|---|
| ci-smoke | $5.77 (Thunder Compute, US) | $6.36 (RunPod Community, EU) | **+$0.59** | **+10.2%** |
| interactive-dev | $9.50 (Thunder Compute, US) | $10.10 (RunPod Community, EU) | **+$0.60** | **+6.3%** |
| training | $23.30 (RunPod Community, EU) | $23.30 (same row) | **+$0.00** | **0%** |

At the 60-hour training scale the cheapest option in the entire study *is already* EU-resident, so
there is no premium to pay. Across all three scenarios combined the EU premium is **$1.19/month**.

**At the 96 GB tier EU residency is free or better than free.** The cheapest 96 GB-class GPU-hour
that passes all three requirements *and* was source-verified is RunPod Community at **$1.69/hr in
an EU datacenter**; the cheapest *non*-EU equivalent is Hyperstack's RTX PRO 6000 SE at **$1.85/hr
in Norway** (EEA, not EU, and not source-verified). The EU option is the cheaper one. On vetted
first-party EU hardware the figures are Verda **$1.89/hr** (Finland) and RunPod Secure **$1.99/hr**
— still within about 8% of the global floor. Two cheaper 96 GB figures exist and neither is usable:
Vast.ai at $1.0015 fails the persistent-volume requirement, and Google Cloud's $1.205215 is the
most suspect number in the dataset (see *What is not verified*).

**Recommendation on this point: take EU residency. It is free at the tier that matters.**

### But "cheapest EU" and "EU on hardware you'd trust with private weights" are different numbers

Both the cheapest-overall and the cheapest-EU rows above land on RunPod **Community** Cloud, which
is vetted *third-party* hosts, not RunPod's own T3/T4 datacenters. Moving to Secure Cloud — same
provider, same EU region, same API — costs:

| Scenario | vs cheapest overall | % |
|---|---|---|
| ci-smoke | +$1.54 | +26.7% |
| interactive-dev | +$2.80 | +29.5% |
| training | +$6.60 | +28.3% |

So the real priced choice is roughly **+28% for vetted datacenter hardware**, and **~0–10% for EU
jurisdiction** — the trust axis costs about three times what the residency axis does.

### What is actually given up by going non-EU

Not "compliance" in the abstract. Concretely:

- **The 13 GB Krea 2 denoiser and any training dataset sit on a third-party host for up to 60
  hours per month.** On a marketplace tier that host may be an individual, not a company.
- **Jurisdiction of the host.** Thunder Compute (the cheapest option) is North-America-only and
  confirmed so; US jurisdiction applies to the machine holding the weights. RunPod is a
  US-headquartered company operating EU datacenters — that is EU *residency*, not EU
  *sovereignty*. Verda (Finnish operator, Finnish datacenters) is the only finalist where both hold.
- **Absence of a DPA.** None of the marketplace tiers surfaced a data-processing agreement in this
  research. Nobody checked, and it is not something to assume.
- **On Vast.ai specifically**, volumes are physically bound to one stranger's machine; if that host
  is rented out or leaves the marketplace, the cache — and anything on it — is simply unreachable.

None of this is a reason not to choose a non-EU option. It is what the $0.59–$1.19/month buys.

---

## Finalist comparison

Monthly totals are the **training** scenario (60 h + 50 GB volume, on-demand) unless noted.
"Reqs" is the 0–3 hard-requirement score.

| Provider | Reqs | EU | Billing | Cheapest 48 GB | Cheapest 80 GB+ | Training $/mo (48 GB) | Verified |
|---|---|---|---|---|---|---|---|
| **RunPod (Secure)** | 3 | ✅ 6 EU DCs | per-second | A40 $0.44 | A100 80 $1.39 · **96 GB $1.99** | $29.90 | CORRECTED |
| **RunPod (Community)** | 3 | ✅ | per-second | A6000 $0.33 | A100 80 $1.19 | $23.30 | CORRECTED |
| **Verda** | 3 | ✅ Finland | 10-min prepaid | A6000 $0.61 · L40S $1.37 | **96 GB $1.89** · H100 $3.25 | $46.60 | CONFIRMED |
| **Thunder Compute** | 3 | ❌ US only | per-minute | **A6000 $0.35** | A100 80 $1.09 | $23.50 | CORRECTED |
| **Hyperstack** | 3 | ⚠️ Norway (EEA) | per-minute | A6000 $0.50 | 96 GB $1.85 · A100 $1.35 | $33.50 | NOT_VERIFIED |
| **Nebius** | 3 | ✅ Finland | per-second | L40S $1.55 | H100 $3.85 | $97.00 | CORRECTED |
| **Lambda** | 3 | ⚠️ DE region, SKU stock unconfirmed | per-minute | A6000 $1.09 | H100 PCIe $3.29 | $75.40 | CORRECTED |
| **Beam Cloud** | 3 | ❓ unknown | per-second, boot free | A6000 $0.82 | A100 80 $2.25 | $49.25 | NOT_VERIFIED |
| **Cerebrium** | 3 | ⚠️ EU menu excludes L40S/A100 | per-second | L40S $1.95 (US only) | H100 $3.40 (EU ok) | $117.07 | CORRECTED |
| **DigitalOcean** | 3 | ⚠️ H100 only in AMS3 | per-second, 5-min min | L40S $1.57 (Toronto) | H100 $4.41 (EU) | $94.20 | CONFIRMED |
| **Modal** | 3 | ✅ but ×1.5–1.75 | per-second | L40S $1.95 base / $2.93 EU | H100 $3.95 / $5.92 EU | $117.07 / $175.61 EU | CORRECTED |
| **Exoscale** | 3 | ✅ Frankfurt, Zagreb | per-second | A40 $1.05 † | 96 GB $2.15 † | $129.00 † | **UNVERIFIABLE** |
| **AWS EC2** | 3 | ✅ 7 EU regions | per-second, 60 s min | g6e.xlarge $1.961 | p5.4xlarge $8.944 (London only) | $121.84 | CORRECTED |
| **Azure** | 3 | ✅ | per-minute | **none in EU** | A100 80 $4.408 | — | NOT_VERIFIED |
| **Google Cloud** | 3 | ✅ Finland | per-second | — | a2-ultragpu-1g $5.536 all-in | $332.16 | NOT_VERIFIED |
| **UpCloud** | **2** ✗billing | ✅ Helsinki | **per-hour** | L40S $1.27 | H100 $1.89 | $88.70 | CORRECTED |
| **Vast.ai** | **2** ✗volume | ✅ 12 EU countries | per-second | RTX 6000 Ada $0.54 (EE) | no EU stock >48 GB | — | NOT_VERIFIED |
| **Koyeb** | **2** ✗volume (10 GB cap) | ❓ | per-second | A6000 $0.75 · L40S $1.20 | A100 $1.60 | — | NOT_VERIFIED |
| **Hetzner** | **0** | ✅ DE/FI | monthly rental | — | **GEX131 96 GB $1.585/hr flat** | ~$989/mo dedicated | NOT_VERIFIED |

† Exoscale prices could not be retrieved from any Exoscale page — see *What is not verified*.

Reconciled superlatives across the whole dataset: the **cheapest 48 GB-class GPU-hour anywhere** is
RunPod Community RTX A6000 at **$0.33/hr**; the cheapest on vetted datacenter hardware is RunPod
Secure A40 at **$0.44/hr**. The **cheapest 80 GB+ passing all three requirements** is Thunder
Compute A100 80 GB at **$1.09/hr** (US), or RunPod Community A100 PCIe at **$1.19/hr** in the EU —
Vast.ai's $1.0015 RTX PRO 6000 is cheaper but fails the volume requirement. The **cheapest 96 GB
that is source-verified and passes all three** is RunPod Community at **$1.69/hr** (EU), or
Verda at **$1.89/hr** on first-party Finnish hardware.

---

## Billing granularity dominates the economics of short bursty runs

This is the single most decision-relevant number in the study, and it is easy to miss because it
does not appear in any hourly rate.

The ci-smoke scenario is 40 runs a month of 12 minutes each — 8 hours of actual compute, 8.5 hours
including boot. On a per-second biller you pay for 8.5 hours. On a **per-hour** biller you pay for
**40 hours**, because each 12-minute run consumes a whole started hour.

**UpCloud, ci-smoke, 1× L40S 48 GB at $1.27/hr:**

| | |
|---|---|
| Compute actually used | 8.00 h → $10.16 |
| **Rounding penalty** | **32.00 h → $40.64** |
| 50 GB MaxIOPS volume (Helsinki) | $12.50 |
| **Total** | **$63.30/month** |

The same workload on RunPod Secure at $0.44/hr costs **$7.31**. UpCloud's L40S is *cheaper per hour*
than AWS's, Nebius's, Modal's, DigitalOcean's and Exoscale's — and it is 8.7× the cost of the
recommendation on this scenario, entirely because of rounding. It is a partner-tier Terraform
provider, a 99.999% SLA, free egress and a Helsinki datacenter, and the billing model disqualifies
it for bursty CI.

Note the asymmetry: the same UpCloud L40S on **interactive-dev** is $37.90 and on **training** is
$88.70 — competitive, because contiguous multi-hour sessions round to at most +1 hour. **Whole-hour
billing is only fatal for the bursty scenario.** If the CI smokes stayed on the apartment 4090 and
only training moved, hourly billers would be back in contention. That is a real option worth
considering, not a rounding error.

Every other billing model in scope is fine for this workload: per-second (RunPod, Nebius, AWS,
DigitalOcean, Modal, Cerebrium, Exoscale, GCP, Beam), per-minute (Thunder, Lambda, Hyperstack,
Azure), or Verda's 10-minute prepaid blocks with automatic refund of the unused portion. Watch two
minimums: DigitalOcean's 5-minute floor (does not bind at 12 minutes) and AWS's 60-second floor.

**Caveat on the 12-minute figure: it is an assumption, not a measurement.** The repo's `gpu.yml`
sets `timeout-minutes: 60`, and a cold `cargo` build over burn + cubecl + tokenizers is heavy.
If a real ci-smoke run is 25 minutes rather than 12, every ci-smoke number here roughly doubles
except the per-hour ones, which do not move at all until you cross 60 minutes. **Measure one real
run before committing.**

---

## What breaks in `gpu.yml`, and what the migration involves

The current workflow targets `pop-os-rtx4090-loractl`, a **long-lived** self-hosted runner
registered once against the repo. Three things stop working when that machine becomes ephemeral.

**1. The runner must register itself, per job, and deregister cleanly.** A permanently registered
runner has no lifecycle; an ephemeral one does. The correct mechanism is GitHub's
**just-in-time (JIT) runner config** — `generate-jitconfig` mints a single-use runner token, the
machine consumes it, runs exactly one job, and auto-deregisters. This avoids the orphaned-runner
cleanup race that the older `config.sh --ephemeral` + registration-token path has. The canonical
reference implementation is `machulav/ec2-github-runner` with `use-jit: true`: a three-job workflow
where `start` provisions and returns a runner label, the real job sets
`runs-on: ${{ needs.start.outputs.label }}`, and `stop` destroys the machine. That shape ports to
any provider with a create/destroy API; only the provisioning call changes.

**2. The ~20 GB weight cache has to come from somewhere other than the box's own disk.** Today the
weights simply live at
`/mnt/sabrent/comfyui-workspace/ComfyUI/models/...` forever. On an ephemeral runner there are three
credible patterns, and the choice constrains the provider:

- **Attached network volume** (RunPod network volume, Nebius Shared Filesystem, Verda block volume,
  AWS EBS, Lambda filesystem). Cleanest. Note RunPod pins a pod to the volume's datacenter, and the
  volume must be attached at *creation* time — you cannot attach it later.
- **Pre-baked image** — an AMI or container image with the weights inside. Works everywhere, no
  volume needed, but rebuilding the image on every model change is friction, and a 20 GB image pull
  is not free unless the provider caches layers.
- **Object storage + download on boot** — simplest, and the one this migration is explicitly trying
  to avoid, since it re-pays 20 GB of transfer per run.

Two providers make this decision for you: **Koyeb caps persistent volumes at 10 GB**, so the weights
do not fit at all, and **Vast.ai volumes are local to a single host**, so the cache is hostage to
one machine.

**3. Nothing measures whether the toolchain builds.** Nobody in this research checked host CUDA
driver version, container image support, or whether a Rust + CUDA 13 toolchain actually compiles on
any of these providers. loractl pins MSRV 1.92 and the `cuda` feature needs `nvcc` at build time.
This is a hard practical gate and it is the top open question — see below.

**Rough migration shape**, provider-agnostic:

```
jobs:
  start:   # call provider API -> create GPU instance with volume attached + JIT runner config
  gpu:     # runs-on: ${{ needs.start.outputs.label }}  — the existing gpu.yml body, unchanged
  stop:    # if: always()  — destroy the instance (NOT stop; see below)
```

The `if: always()` on teardown is load-bearing and provider-specific in a way that bites:
**Hyperstack bills full compute on a SHUTOFF VM** (you must delete or hibernate), **DigitalOcean
bills a powered-off Droplet until it is destroyed**, and **Vast.ai bills storage for every second
an instance exists regardless of state**. A teardown step that stops rather than destroys silently
converts a $7/month CI bill into a $300/month one.

Two options skip most of this work entirely: **machine.dev** provisions ephemeral GPU runners
natively (`runs-on:` a label, no orchestration code, L40S in Spain, spot at $0.94/hr) but its
persistent-volume story is unverified — its storage doc 404s, which is exactly the requirement that
matters. **RunsOn** and **Cirun.io** are BYO-cloud control planes; both have free tiers that likely
cover a public personal repo like loractl at $0 licence cost, and RunsOn's pre-baked-AMI pattern is
the cleanest answer to the 20 GB problem — at AWS GPU prices, the most expensive tier here.

---

## Trade-offs and risks

**Which providers fail which hard requirement.** Failing sub-hour billing: UpCloud, OVHcloud,
Scaleway (contested), Vultr, Crusoe, Hetzner, Together AI, Actuated. Failing persistent volume:
Vast.ai (local-only volumes), Koyeb (10 GB cap), Salad (100 MB per file, 30-day deletion), Hetzner
(bare metal, no detachable volume), GitHub-hosted GPU runners, Replicate. Failing API provisioning:
Hetzner's GEX line (the `hcloud` provider covers Hetzner Cloud only), CUDO (public on-demand
platform closed 2026-03-31), Replicate (predictions API, not job execution).

**Products that do not exist.** Fly.io GPUs were deprecated 2026-07-31 and unavailable after
2026-08-01 — eight days before this study; any comparison quoting Fly GPU prices is stale. Depot has
**no** GPU runners despite widespread third-party claims to the contrary (the claim refers to its
container-build product). `philips-labs/terraform-aws-github-runner` is archived; the maintained
project is `github-aws-runners`. CUDO's Terraform provider is still in the registry but targets a
withdrawn platform.

**Marketplace rentals are not datacenter VMs.** On Vast.ai and Salad the physical host is a stranger
— an individual with a gaming PC, in most cases, not a company with a datacenter. Concretely: the
operator is not identified to you, there is no SLA and no DPA, a host can vanish mid-run (Vast's own
docs tell you to back up to cloud storage rather than trust the machine), and when a host disappears
its **volume goes with it** because Vast volumes are physically local to one machine. Salad's own
docs say nodes "can disconnect at any time" *even at the highest priority level*. RunPod Community
Cloud is a middle ground: third-party hardware, but RunPod vets the hosts and the tier is *not*
preemptible — it cannot be outbid mid-run, which makes it materially more predictable than Vast for
a 60-hour training job. TensorDock claims it revokes host SSH access so hosts cannot read customer
data; that is a vendor claim, not something verified here.

**Spot interruption during a long training run.** A 60-hour run will be interrupted at least once on
most preemptible tiers. Genuine preemptible pricing exists at Nebius (L40S $0.74, H100 $2.15), Verda
(~50% off across the fleet), GCP (A100 80 GB at 87% off in Finland, H100 at 90.4% off), Azure
($0.815 for an A100 80 GB in North Europe) and UpCloud. Three traps: (a) **`spot_usd_hr` means three
different things** in this data — genuine preemptible (Nebius, GCP, Azure, UpCloud, Verda),
Vast.ai's *minimum bid floor* (not a clearing price), and RunPod Community (not preemptible at all);
cross-provider spot comparison is invalid without saying which. (b) **AWS spot prices are null here**
— AWS publishes no unauthenticated spot feed, and the figures that previously appeared were
on-demand multiplied by a Spot Advisor discount band. The bands themselves are confirmed, and they
say something uncomfortable: the small single-GPU `g6e` shapes sit in the **>20%/month interruption**
band while the 8-GPU `p5` shapes are **<5%** — exactly backwards from what bursty CI wants.
(c) **Azure's H200 spot meters are priced identically to on-demand** — a spot tier exists so the API
answers, but buying it saves nothing.

**Residency gaps that look like EU but are not.** Norway (Hyperstack, Genesis Cloud, Shadeform's
Oslo capacity) and Iceland (Crusoe, RunPod EUR-IS, Nebius eu-north2) are **EEA/GDPR but not EU
member states**. Switzerland (Exoscale's Geneva and Zurich zones) is neither. The UK (Cerebrium's
`eu-west-2`, AWS's London p5.4xlarge — the only single-H100 shape in the region) is post-Brexit and
not the EU. Several EU claims elsewhere in the source data turned out to be conflations of "the
company is European" with "a GPU of the required tier schedules in a named EU-member region"; those
have been corrected here, but the pattern is worth watching in any follow-up.

**Quota and access friction is a multi-day tax no price captures.** AWS, GCP and Azure all gate GPUs
behind per-region quota requests that become Support cases; on-demand and spot are *separate*
quotas, and GCP needs *two* approvals per model (regional **and** global — missing the global one
silently blocks you even after the regional one lands). Azure adds a trap the others do not: its own
docs warn a deployment can fail on **capacity** even with quota approved. Exoscale requires manual
account screening "granted with priority to established businesses", which may simply refuse a solo
developer — a human gate no API can route around.

**Cheapest is not fastest, and this study does not measure throughput.** An Ampere-era A40 at
$0.44/hr that takes 1.8× longer per step than a $0.99/hr Ada L40S is not cheaper. Every
recommendation above is built on price and requirements only. Running `just bench` on two candidate
SKUs — say a RunPod A40 and a RunPod L40S in the same EU datacenter — would settle it in an
afternoon and is worth more than any further price research.

---

## What is not verified

**UNVERIFIABLE figures carried forward** (present in the CSVs, flagged in `verification_status`):

- **Exoscale — every price.** The pricing page renders figures client-side; the served HTML contains
  only Angular template placeholders (`{{ prices.opencompute.gpu3.small[currency] | ... }}`). No
  numeric price is retrievable from the pricing page, the per-GPU pages, or the calculator. The
  figures shown ($1.05 A40, $2.15/$4.17/$8.19 RTX Pro 6000) are corrected numerals after removing a
  fabricated 1.1535 EUR→USD conversion — Exoscale publishes CHF/EUR/USD as byte-identical numerals,
  so no conversion applies. Treat as indicative only. During verification a first fetch of this page
  returned a confident, fully-formed price table that was **fabricated by the summarizing model**;
  those numbers were discarded and must not resurface.
- **AWS spot — all null by design.** Not missing data; AWS publishes no unauthenticated spot price
  feed. Any AWS spot number you see elsewhere in a comparison is derived, not quoted.
- **TensorDock — everything except the H100.** The main pricing table is stamped "Last Updated:
  July 24, 2024" and carries the vendor's own inaccuracy disclaimer. Drift is proven: the table says
  RTX 4090 $0.35, the live 4090 page says "from $0.37". The availability endpoint returns HTTP 200
  with an empty hostnodes object — not one live host could be enumerated.
- **Google Cloud — every figure except `a2-ultragpu-1g`.** All published GCP GPU prices are
  **accelerator-only**; vCPU and RAM bill as separate SKUs, so a bare GCP number against a whole-VM
  number understates GCP by 20–30%. Only the one worked all-in example ($5.536/hr) is comparable and
  it is the only GCP row in `scenarios.csv`. The **"RTX 6000 96GB @ $1.205215/hr in europe-north1"**
  figure is the single most suspect number in the whole dataset — roughly half the next-cheapest
  96 GB GPU anywhere, and a sibling SKU reads "1 gpu slice" at $0.495. **It anchors nothing here.**
- **Scaleway — all 48 GB+ prices.** Zone-gated and never rendered; `?zone=fr-par-2` is ignored
  server-side and the public product-catalog API returns no GPU products at all. The one L40S figure
  ($1.70) is a "from €1.47/hr" converted at the same unchecked 1.1535 rate that inflated Exoscale.
- **Gcore — every price.** "From €x/hour" floor prices excluding VAT, converted at an **assumed**
  1.17 USD/EUR rate that was never fetched. The least solid numbers in the study.
- **Oracle Cloud — every price.** Every `oracle.com` pricing route returned HTTP 403. The GPU shape
  ladder is confirmed first-party and is likely disqualifying on its own: every OCI shape at 48 GB or
  above is a 4- or 8-GPU **bare-metal node**; the only single-GPU VMs are A10 24 GB or V100 16 GB.
- **Lambda — the 1× H100 SXM and 1× B200 rows.** The page publishes *ranges* ($3.99–4.29,
  $6.69–6.99); both endpoints are real but the 1×-vs-8× mapping was not readable.
- **Nebius B300.** Three reads of the prices page returned three different answers (absent /
  "Contact us" / $7.85). Not certified, though no contradicting number was found.
- **Cerebrium L40 (the only EU-schedulable 48 GB SKU).** Available in both GA EU regions; its price
  is simply not broken out on the pricing page.
- **Genesis Cloud, Prime Intellect, FluidStack, CUDO, Voltage Park (beyond one figure), Fly.io** —
  no price obtainable at all. Circulating third-party figures for these were deliberately **not**
  recorded.

**Storage costs omitted (totals are lower bounds):** DigitalOcean (Volume per-GB price not
published/fetched), Google Cloud (PD price not captured), Exoscale (unverifiable).

**Providers that were never source-verified** (`verification_status = NOT_VERIFIED`) — everything
outside the twelve finalists. Notably including four that appear in the cost model: **Hyperstack**,
**Azure**, **Google Cloud** and **Beam Cloud**. Azure's data quality is nonetheless high (official
unauthenticated Retail Prices API, whole-VM prices); Hyperstack's and Beam's rest on a single
marketing page read.

**Whole tiers never researched:** the European sovereign-cloud tier — STACKIT, Open Telekom Cloud,
Leafcloud, Qarnot, Sesterce, Nscale, Ori — and much of the budget tier — Novita, Hyperbolic, Deep
Infra, Jarvislabs, Lightning AI, Northflank, Akamai/Linode. Given that the EU premium turned out to
be ~$1/month, the sovereign tier is the gap most likely to change the *qualitative* answer (EU
jurisdiction end to end) rather than the price.

**Nobody checked the thing that actually gates this migration.** Host CUDA/driver version, container
image support, and whether a Rust + CUDA 13 toolchain (MSRV 1.92, `nvcc` required at build time for
the `cuda` feature) builds and runs on any of these providers. **This is the top open question** and
it is worth resolving before any purchasing decision: a provider that cannot compile the project is
free at any price.

**The 12-minute ci-smoke run length is an assumption, not a measurement** — see the billing section.

---

## Methodology

- **Prices collected 2026-08-08.** GPU pricing moves fast and marketplace prices (Vast.ai, RunPod
  Community, Shadeform) float continuously — every marketplace figure is a snapshot, not a rate card.
- Every price in `gpu-instances.csv` carries a `source_url` for the page it was read from. Where a
  price could not be fetched, `on_demand_usd_hr` is **empty**, never a number from memory.
- USD figures are the provider's own published USD wherever one exists. Derived FX conversions are
  labelled as such and treated as suspect — one fabricated conversion inflated an entire provider's
  prices by 15.35% before it was caught.
- **Cost model:** ci-smoke = 40 runs × 12 min + per-run boot (provider-documented, else 120 s
  assumed and marked); interactive-dev = 20 h; training = 60 h. A 50 GB volume is held all month in
  every scenario. `assumptions` in `scenarios.csv` shows the arithmetic per row, including the
  whole-hour rounding penalty broken out into `overhead_usd`.
- **Scope of `scenarios.csv`:** for each provider passing all three hard requirements with a fetched
  price, the cheapest SKU in the 48 GB tier and the cheapest in the 80 GB+ tier. UpCloud (2/3, fails
  billing) is included deliberately — it is the clearest illustration of the rounding effect and
  dropping it would hide the study's most decision-relevant finding. No 2-of-3 near-miss undercut
  the cheapest fully-passing option by more than 30%, so none qualified on that rule.
- **To re-run:** the per-provider pricing URLs in `gpu-instances.csv` are the fetch list. Three
  providers need a browser rather than a plain fetch (Exoscale, Scaleway's zone selector, GCP's SKU
  explorer); three need authentication (Prime Intellect, Vultr's marketing pages via its public API
  instead, Oracle). The billing-granularity and persistent-volume claims are the ones worth
  re-checking first — they change the answer far more than a 10% price move does.
