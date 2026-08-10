---
paths:
  - "crates/**/*.rs"
  - ".github/workflows/**"
  - "justfile"
---

# Two `gpu.yml` Failures That Report the Wrong Cause

The self-hosted 4090 runner produces two failures whose error text points
somewhere other than the actual problem. Both cost a live misdiagnosis on
2026-08-05, and both are now preflighted or configured away in `gpu.yml` — this
file is why, so a future session recognizes the signature if it returns.
Sibling to [`cubecl-pool-reclaim-stale-page.md`](cubecl-pool-reclaim-stale-page.md)
(the ADR-0005 OOM-vs-race reclassification) and the user-global
`disk-full-recovery.md`.

## 1. `ld terminated with signal 7 [Bus error]` means a FULL DISK

```
collect2: fatal error: ld terminated with signal 7 [Bus error], core dumped
error: could not compile `loractl-core` (test "dataset_pipeline")
```

Reads as an LLVM/`rust-lld` bug — it even prints an LLVM stack trace and asks
you to file one. It is ENOSPC. `rust-lld` **mmaps its output file**, so a
filesystem that fills mid-link faults on the mapped page and the kernel raises
**SIGBUS**, not a clean `No space left on device`.

Two things make it hard to place:

- **Routine ENOSPC is not kernel-logged**, so `journalctl` is clean. An empty
  log rules out an OOM-*kill*; it does **not** rule out a full disk.
- **The evidence deletes itself.** The job dies, the runner wipes its target
  dir, and `df` afterwards shows plenty free. The only way to see it is to
  sample **during** the link.

```sh
ssh popos.intra.lakuz.com 'for i in $(seq 1 20); do date +%H:%M:%S; df -h / | awk "NR==2{print \$4}"; sleep 12; done'
```

> Measured 2026-08-05: `/` went **13G free → 5.5G → 359M (100%)** in ~25 s while
> the target dir reached **14 GB**, then the job died and the directory was wiped.

**More headroom is not the fix.** Freeing 7 GB produced an identical failure at
21 GB free — the concurrent links at the end of a debug build are the peak, and
the root disk is 156 GB, ~90% full at rest, and **shared with a second runner**
(`custom-attention-engine-framework`). Any headroom-based fix is one cleanup
away from breaking again.

The fix is to shrink the artifacts: `gpu.yml` sets
`CARGO_PROFILE_DEV_DEBUG` / `CARGO_PROFILE_TEST_DEBUG` to `line-tables-only`.
DWARF is the bulk (one `libloractl_core` rlib is ~107 MB); backtrace file/line
survives, which is what a failing smoke is read for, and codegen is untouched.
**Measured effect: 14 GB → 1.3 GB**, and `cuda smokes` went green after four
consecutive failures. `--release` builds were never affected, which is why the
bench compiled cleanly throughout.

## 2. `non-finite loss (NaN) at step 1` usually means a CONTENDED GPU

The message names f16 range overflow **unconditionally**, so on an f32 config it
advises the precision already in use:

```
Error: non-finite loss (NaN) at step 1 — numeric overflow. With
compute.precision: f16 this means an activation exceeded f16's range; try f32
```

The real cause is normally that another process holds the card. Allocations
fail, the forward computes garbage, and the loss goes non-finite. **The
discriminator sits above the NaN in the log**, on a `DS*` thread:

```
thread 'DSD-0-0' panicked at cubecl-cuda-0.10.0/src/compute/stream.rs:101
  couldn't find resource for that handle: Memory page 0 doesn't exist
```

Per ADR-0005 that is OOM fallout, not a reclaim race — a failed allocation
leaves its handle at the uninitialized `{pool:0,page:0}` default. cubecl
swallows these (`WARN Task failed`), so the NaN is the only thing that surfaces.
An exit code of **139** (SIGSEGV) is a second tell.

**Check contention before believing any VRAM number:**

```sh
ssh popos.intra.lakuz.com 'nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv; curl -s http://127.0.0.1:8188/queue'
```

> Observed 2026-08-05: ComfyUI held **17,624 MiB of 24,564 MiB with an empty
> queue** — idle cached weights — against the 19.4 GB this config needs. Freeing
> it took the same dispatch straight to a clean result.

Release an idle ComfyUI cache (only with an empty queue; it reloads on the next
generation):

```sh
curl -X POST http://127.0.0.1:8188/free -H 'Content-Type: application/json' -d '{"unload_models":true,"free_memory":true}'
```

`gpu.yml`'s bench job now preflights this and refuses below 4 GiB free. The
`check_step_loss` message itself is still wrong for f32 configs — tracked
separately; it should read the configured precision before giving f16 advice.

## 3. `no tokenizer found` + `http status: 401` means a GATED repo, nine minutes in

A ComfyUI-layout run ships no `tokenizer.json`, so `hf::fetch_qwen3vl_tokenizer`
falls back to `krea/Krea-2-Raw`. That repo is **gated** (`gated: "auto"`,
verified against the HF API 2026-08-07) — `hf.rs` still calls it ungated in two
places, and `config.rs` still tells users the ComfyUI flow "needs nothing set
here":

```
  load: text encoder (4.9 GiB) from qwen3vl_4b_fp8_scaled.safetensors
Error: encoding caption for <first image>
Caused by:
    0: no tokenizer found (no model.tokenizer override, no tokenizer/tokenizer.json under base)
    2: http status: 401
```

Unauthenticated gives **401**; a valid token whose account has not accepted the
terms gives **403**.

What makes it expensive rather than merely wrong: the tokenizer is not needed
until the **first caption is encoded**, so the failure lands *after* the 4.9 GiB
text encoder has loaded — **~9 minutes per attempt**, and once per arm of any
sweep. A 2026-08-07 matrix run burned eight arms rediscovering it.

**Anyone with a warm `$HF_HOME/loractl/qwen3vl-4b-tokenizer.json` never sees
it**, which is why it went unnoticed: the cache predates the gating. The
corollary is a second trap — *pointing `HF_HOME` somewhere new silently
invalidates that cache* and sends the next run to the network. That is exactly
how the 2026-08-07 run found it.

Resolve it before loading anything. `hf.rs` pins the file's SHA-256, so **any**
byte-identical copy is provably the right tokenizer — there is no judgement call
about tokenizer identity, only a hash check:

```sh
sha256sum "${HF_HOME:-$HOME/.cache}/loractl/qwen3vl-4b-tokenizer.json"
# expect be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506
```

Then pass it explicitly (`model.tokenizer`, or `--tokenizer` in
`scripts/residency-matrix.sh`, whose preflight does exactly this check). Tracked
as #200.

## The shared lesson

The first two **report a plausible wrong cause rather than erroring honestly**,
and in both the honest evidence is one cheap measurement away — `df` during the
link, `nvidia-smi` before the run. Neither is recoverable from the error text
alone, and one of them (the NaN) actively points at the wrong subsystem.

The third is the variant worth recognizing separately: its error text is
**accurate**, and it still costs an afternoon, because it arrives ~9 minutes and
one 4.9 GiB load into every attempt, and because the code's own comments assert
the opposite ("ungated", "needs nothing set here"). The remedy is the same shape
as the other two — move the check to the front, where it costs milliseconds — but
the tell is different: not a misleading message, a **late** one resting on a
stale premise. When a precondition is cheap to verify and expensive to discover,
verify it in a preflight rather than at the point of use.

When a GPU-runner failure names a component, check the *resource* at the failure
point before believing it — the same law as the user-global
`diagnose-at-the-failure-point` rule, and as ADR-0005's own reclassification of
these very panics from "reclaim race" to "genuine OOM".
