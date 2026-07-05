# 0003 — HTTP API: event serialization, streaming, and run lifecycle (M5)

- **Status:** Accepted
- **Date:** 2026-07-05
- **Milestone:** M5 (issue #4 — "expose the event stream over HTTP so a GUI
  can be built independently")
- **Deciders:** loractl maintainers

## Context

Core's design rule has promised since M1 that "a future HTTP API would
serialize the *same* events as SSE/JSON" — the API is just another renderer
over the one event pipeline, exactly like the CLI's progress bar. M5 (#4)
makes that real: a new `loractl-api` crate that starts training runs over
HTTP and streams their `TrainEvent`s to clients.

Facts that shaped the decisions below (each verified against the repo, not
assumed):

- `TrainEvent` has six variants; before M5, `Warning(String)` was the only
  newtype (non-struct) variant. serde **cannot internally tag** a newtype
  variant wrapping a primitive — `#[serde(tag = "type")]` on
  `Warning(String)` fails at serialization time, not compile time.
- `serde` and `serde_json` were **already unconditional runtime
  dependencies of `loractl-core`** (`TrainConfig` derives both ways; the
  trainer writes JSON sample reports). Deriving `Serialize` on `TrainEvent`
  adds no new dependency.
- `Trainer::train` is synchronous and blocking, with a
  `&mut dyn FnMut(TrainEvent)` sink — the bridge to an async HTTP server has
  to cross the sync/async boundary somewhere.
- `MockTrainer` finishes in microseconds: any design where a client can only
  see events emitted *after* it connects is structurally flaky (the run is
  over before `curl` gets there).

## Decision

### 1. `Serialize` is derived on core's `TrainEvent` — the events are the wire schema

`TrainEvent` gets `#[derive(Serialize)]` with
`#[serde(tag = "type", rename_all = "snake_case")]`: internally tagged, flat
JSON objects (`{"type":"step","step":42,...}`). No DTO mirror in the API
crate, and no `Deserialize` (clients consume events; nothing parses them back
into core).

Because serde cannot internally tag a newtype variant, `Warning(String)` was
reshaped to `Warning { message: String }` — a three-site, compiler-caught
edit (`cli.rs` match arm, two `burn_trainer.rs` constructors). The
alternative (adjacent tagging, which tolerates newtypes) would have nested
every payload under a `"data"` key and made warning's payload a bare string
while every other event is an object — a worse wire shape to fix a variant we
could simply reshape.

**Consequence:** the JSON wire format is now part of core's public contract.
It is pinned byte-for-byte by `train_event_wire_shapes`
(`crates/loractl-core/tests/event_json.rs`), whose golden strings are
reproduced verbatim in `docs/api/events.md` — the test is the doc-drift
tripwire.

### 2. Run failure is an API-layer `failed` event, not a core variant

Core's failure channel is `Trainer::train`'s `anyhow::Result` — duplicating
it as a core event would create two sources of truth. Instead the API's run
supervisor converts both failure modes into a seventh, API-owned terminal
event: trainer `Err` → `{"type":"failed","error":"<anyhow chain>"}`; a
panicked trainer thread → `{"type":"failed","error":"trainer panicked"}`.
Every stream therefore ends with exactly one terminal event — `finished`
XOR `failed` — and there is no status endpoint: replay-from-start (below)
makes the stream itself the status.

### 3. Cursor-over-pre-serialized-history + a `watch` wake channel

Each run keeps an append-only history — `Mutex<RunLog { events:
Vec<StoredEvent>, done: bool }>` — where every event is serialized **once**
at emission time on the training thread (`StoredEvent { kind: &'static str,
json: String }`). A `tokio::sync::watch::Sender<()>` is the wake signal.
Subscribers each hold their own cursor over the history: snapshot
`events[cursor..]` + `done` under the lock, yield, and either close (done and
drained) or park on `watch::Receiver::changed()`.

Chosen over a `tokio::sync::broadcast` channel to eliminate lag/loss
semantics entirely: broadcast drops events for slow receivers (`Lagged`),
which would have meant documenting "step charts may have gaps." With the
cursor model there is **zero event loss ever** — one code path serves live,
mid-run, and finished subscribers, and the replay history doubles as the
buffer. Pre-serialization removes serialize-at-edge panics from the SSE path
and makes replaying to N clients a string clone.

**The missed-wake fix is load-bearing.** A subscriber that consumes the
`finished` wake and re-parks *before* the supervisor sets `done` would hang
forever. Therefore the supervisor's `done` flip is **always** followed by a
`send_replace(())` on **every** arm — including the `Ok` arm where the
trainer emitted `Finished` itself — and subscribers subscribe to the watch
channel *before* their first snapshot. Spurious wakes cost one empty loop
iteration; missed wakes were the bug. The sentinel tests are
`live_tail_delivers_events_before_run_completes` and
`trainer_ok_without_finished_still_closes_stream` (removing the post-done
wake makes both fail at their 5 s timeouts).

### 4. `spawn_blocking` + a supervisor task; runs always train to completion

`train()` runs on tokio's blocking pool inside a `tokio::spawn`ed supervisor
that is the choke point for every terminal path (Ok / Err / panic — see
decision 2). The blocking closure first runs
`create_dir_all(&config.output.dir)`, mirroring the CLI, which also creates
the output dir before training.

Lifecycle: a client disconnect only drops that subscriber's cursor — **the
run always trains to completion** (tested: `client_disconnect_does_not_kill_run`).
Server shutdown is abrupt (SIGINT kills the blocking threads); clients
observe a close without a terminal event, which the contract defines as
"server died — reconnect and replay." A trainer that returns `Ok` without
emitting `Finished` (a contract violation) still gets `done` + wake, so
streams close rather than hanging subscribers.

The `Mutex` is `std::sync::Mutex`, never held across an `.await`: writers are
sync code on the blocking thread; readers copy-then-release. The sink never
awaits or blocks (a watch send is sync; the lock is held for a push), so
there is no backpressure onto training and no deadlock by construction.

### 5. The `TrainerFactory` seam

`app(factory: TrainerFactory)` takes
`Arc<dyn Fn() -> Box<dyn Trainer + Send> + Send + Sync>`; `main.rs` holds the
single real `BurnTrainer` line (the analogue of the CLI's one constructor
line), and the integration tests inject mock/failing/panicking/gated/silent
trainers — the seam is demanded by the offline test gate, not speculative. A
fresh trainer is built per `POST /runs` (`train(&mut self)` + concurrent
runs).

Note: `Trainer` has **no `Send` supertrait** — the `Box<dyn Trainer + Send>`
bound compiles because current impls are unit structs. A future `!Send`
trainer breaks this seam at compile time; that is the desired failure mode
(loudly, at the boundary), recorded here so it isn't mistaken for an
accident.

### 6. No event envelope — the SSE `id:` field carries ordering

An `Envelope { seq, ts_ms, event }` wrapper was considered and rejected. The
ordering/dedup/reconnect key a GUI needs is provided by the SSE `id:` field
(the 0-based history index) with zero JSON schema surface; `ts_ms` would
break the byte-for-byte golden strings and a GUI can timestamp on receipt.
**Revive trigger:** a GUI that genuinely needs *server-side* timestamps
(e.g. accurate step timing across reconnects).

### 7. YAGNI cuts, each with its revive trigger

| Cut | Revive trigger |
|---|---|
| `GET /runs` (run listing) | The first multi-run GUI view |
| `GET /runs/{id}` (status) | Not expected — replay-from-start makes the stream the status; revisit only if replay cost ever matters |
| `DELETE /runs/{id}` (cancel) | Needs a core cancellation hook first — `Trainer::train` has no way to be interrupted; wrong milestone to change that contract |
| CORS | The first browser-origin GUI (a non-localhost `Origin`) |
| Auth / TLS | The first non-localhost bind (`LORACTL_API_ADDR` beyond 127.0.0.1) |
| History eviction / persistence | A long-lived server deployment — today history is unbounded and process-local (dev tool; restart clears it), and this is the first thing such a deployment must revisit |
| Health endpoint | The first deployment behind a probe/load balancer |
| Graceful shutdown / drain | Same long-lived-deployment trigger as eviction |

### 8. Evolution rules (additive-only)

Within a major version the server only **adds** event types and fields; it
never renames or removes them. Clients MUST ignore unknown `type` values and
unknown fields. These rules are part of the client contract in
`docs/api/events.md`.

## Alternatives Considered

**A DTO mirror of `TrainEvent` in the API crate (core stays serialization-
free).** Rejected: core already depends on serde + serde_json, and core's own
crate doc declares the events are the wire schema. A DTO duplicates six
variants plus a `From` impl to defend against instability that does not
exist in a single-owner workspace.

**Adjacent tagging (`#[serde(tag = "type", content = "data")]`).** Rejected —
see decision 1: worse wire shape to avoid a three-line reshape.

**A `Failed` variant in core's `TrainEvent`.** Rejected — see decision 2:
core's `Result` is the failure channel; a core event would be a second
source of truth that trainers could forget to emit.

**`tokio::sync::broadcast` for fan-out.** Rejected — see decision 3: lossy
for slow receivers, needs a separate replay path for late subscribers, and
its lag semantics would have to be documented as client-visible gaps.

**Live-only streaming (no replay).** Rejected: `MockTrainer` finishes before
a client can connect, making the primary acceptance flow structurally flaky;
replay also deletes the need for a status endpoint.

## Consequences

**Positive**

- A GUI can now be built against a documented, tested, additive-only wire
  contract without touching Rust — the M1 thesis ("GUI-optional by
  construction") is demonstrated, not just asserted.
- Zero event loss for any subscriber, one code path for live/late/finished
  clients, and deterministic replay (byte-identical bodies across repeated
  GETs — tested).
- The golden test makes wire-format drift a test failure instead of a silent
  GUI breakage.

**Negative / costs**

- The wire format is now core API surface: changing an event shape is a
  breaking change pinned by a test, no longer an internal refactor.
- Unbounded per-run history: memory grows with run length × run count until
  process restart. Accepted for a localhost dev tool; the first thing a
  long-lived deployment revisits (decision 7).
- A POSTed run commits a blocking-pool thread to completion — there is no
  cancellation until core grows a cancellation hook.
- `loractl-api` adds axum/tokio to the workspace's dependency tree (the CLI
  path is unaffected; the API crate is a separate binary).

## References

- Issue #4 (M5), roadmap in `README.md`.
- `docs/api/events.md` — the client-facing wire contract this ADR's
  decisions produce.
- `crates/loractl-core/src/event.rs` — the `Serialize` derive and `Warning`
  reshape; `crates/loractl-core/tests/event_json.rs` — the byte-for-byte
  golden.
- `crates/loractl-api/src/state.rs` — history, watch wake, supervisor,
  `failed` event; `crates/loractl-api/src/routes.rs` — the two handlers and
  SSE framing; `crates/loractl-api/tests/api.rs` — the 11 offline
  integration tests.
- `docs/adrs/0001-first-real-target-model.md`,
  `docs/adrs/0002-adapter-format-and-sample-semantics.md` — the M3/M4
  decisions this milestone builds on (0002 documents the M4 events this API
  now streams).
