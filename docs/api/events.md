# loractl HTTP API — the event wire contract (M5)

`loractl-api` is a renderer over `loractl-core`, exactly like the CLI: where
the CLI turns `TrainEvent`s into a progress bar, the API serializes the *same*
events as JSON over Server-Sent Events (SSE). This document is the contract a
GUI (or any HTTP client) builds against. Design rationale lives in
[ADR-0003](../adrs/0003-http-api-event-streaming.md).

## Endpoints

| Endpoint | Request | Success | Errors |
|---|---|---|---|
| `POST /runs` | `Content-Type: application/json`, body = a JSON `TrainConfig` (same schema as the YAML config file) | `201` `{"id":1,"events_url":"/runs/1/events"}` | `400` `{"error":"…"}` — `output.dir`/`output.name` escape the [output base](#output-paths-are-confined)<br>`422` invalid body — plain-text diagnostic, see below<br>`429` `{"error":"…"}` — the [concurrency cap](#concurrency-cap) is saturated<br>(none of these create a run, and none burn an id) |
| `GET /runs/{id}/events` | — | `200` `text/event-stream`: full replay from event 0, then live tail, with keep-alive comments | `404` `{"error":"unknown run id"}` — the id was never issued **or its run has been evicted** (see [Run retention](#run-retention)) |

That is the whole M5 surface. There is no run listing, no status endpoint, no
cancellation, **no auth** — see ADR-0003's cut list and revive triggers. The
default `127.0.0.1` bind is what keeps an unauthenticated `POST /runs` safe;
the confinement and cap below are the guards that hold if you move that bind.

Error bodies are **not uniform**. `400` (path rejection), `404`, and `429` are
JSON `{"error": "…"}`, but the `422` comes from axum's `Json` extractor and is
**plain text** (`content-type: text/plain; charset=utf-8`) describing the
deserialization failure, e.g.:

```
Failed to deserialize the JSON body into the target type: missing field `model` at line 1 column 2
```

Do not parse `422` bodies as JSON — surface them as a human-readable
diagnostic. A syntactically malformed body — not even valid JSON — is also a
**plain-text** `400` from the extractor, so a client that wants to distinguish
"bad path" from "bad JSON" must check `content-type`, not just the status.

## Output paths are confined

`output.dir` and `output.name` become real filesystem writes (the run's
directory, its checkpoints, its `.safetensors` adapter). The endpoint is
unauthenticated, so those fields are **not** taken as literal paths: they name
a location *inside* a base directory the server owns.

| | |
|---|---|
| Env var | `LORACTL_OUTPUT_BASE` |
| Default | `./runs` (relative to the server's working directory) |

The rules, all enforced before any run is created:

- **`output.dir` must be relative and free of `..` components.** An absolute
  path (`/etc/loractl`) or a traversal (`../../..`) is a `400`.
- **`output.name` must be a plain file name** — no separators, no `..`, not
  empty. It is joined onto the dir, so it escapes just as easily.
- **Symlinks may not escape the base**, even ones already on disk.

An accepted request runs against the **resolved absolute path**, which is what
the `checkpoint` / `sample` / `finished` events report — so `"dir": "demo"`
with the default base surfaces as `.../runs/demo/lora.safetensors`, not
`demo/lora.safetensors`. Nothing outside the base is ever created.

## Concurrency cap

Each run occupies a thread doing real compute for as long as it trains, so the
number of **simultaneous** runs is bounded. Past the cap, `POST /runs` returns
`429` immediately rather than queueing — a client learns at once instead of
timing out.

| | |
|---|---|
| Env var | `LORACTL_MAX_CONCURRENT_RUNS` |
| Default | `4` |

A slot is released the moment a run emits its terminal event, so the retry is
simply the same `POST` again. This is a resource guard, not a scheduler: there
is no queue and no fairness between clients.

## Event shapes

Every `data:` payload is a flat JSON object whose `type` field is the
discriminator. Six event types come from core's `TrainEvent`:

```json
{"type":"started","total_steps":1000}
{"type":"step","step":42,"loss":1.2345,"lr":0.0001}
{"type":"checkpoint","step":250,"path":"output/checkpoint-250.safetensors"}
{"type":"sample","step":500,"path":"output/sample-500.png"}
{"type":"warning","message":"lr clipped"}
{"type":"finished","adapter_path":"output/lora.safetensors"}
```

**These six examples are enforced byte-for-byte by the
`train_event_wire_shapes` golden test**
(`crates/loractl-core/tests/event_json.rs`) — the strings above are copied
verbatim from that test. If this document and that test ever disagree, the
document has drifted; fix the document.

The seventh type is the API-layer terminal event, produced by `loractl-api`
itself (in `crates/loractl-api/src/state.rs`, not by core) when a run fails —
whether the trainer returned an error or panicked:

```json
{"type":"failed","error":"creating output dir output: Permission denied (os error 13)"}
```

`error` is a human-readable message (the full `anyhow` error chain for trainer
errors, or `"trainer panicked"` for a panic).

## SSE framing

Each event is one SSE frame carrying all three fields:

```
id: 4
event: finished
data: {"type":"finished","adapter_path":"output/lora.safetensors"}

```

- **`id:`** — the 0-based per-run history index. Stable across replays, so it
  is the ordering/deduplication key: standard `EventSource` semantics, and it
  leaves `Last-Event-ID` resume open without the server implementing it.
- **`event:`** — the same string as the JSON `type` field. `EventSource`
  listeners can bind per event name (unknown names are auto-ignored); since
  `data` repeats `type`, fetch-stream parsers may ignore SSE event names
  entirely.
- **Keep-alive comment frames** (lines starting with `:`) flow during long
  gaps between events. Clients must skip comment lines.

## Lifecycle contract

1. `GET /runs/{id}/events` **always replays from index 0**, then live-tails.
   A reconnect is a full replay; deduplicate by `id:`.
2. Every stream ends with **exactly one terminal event** — `finished` or
   `failed` — after which the server closes the connection.
3. **A stream that closes *without* a terminal event means the server died**
   (or the transport failed) — reconnect and replay.
4. Evolution: the server only **adds** event types and fields; it never
   renames or removes them within a major version. Clients MUST ignore
   unknown `type` values and unknown fields.
5. Serialization notes: paths are UTF-8 strings; `serde_json` serializes
   non-finite floats as `null`, so clients must tolerate `"loss": null`; an
   event the server cannot serialize (e.g. a non-UTF-8 path) is replaced
   server-side by `{"type":"warning","message":"unserializable event"}`.
6. Caveats: run ids are process-local and **not stable across server
   restarts**; two runs sharing the same `output.dir` clobber each other's
   checkpoints (a pre-existing `TrainConfig` property, not an API one).
7. A run's history is **not retained forever** — see [Run retention](#run-retention).

## Run retention

The event history *is* the replay buffer (ADR-0003), so it is held in memory
for the life of the run. To give that a ceiling, the server keeps only the
**N most-recently-completed runs**; older completed runs are evicted and their
events dropped, so `GET /runs/{id}/events` on an evicted id is a plain `404`,
indistinguishable from an id that was never issued.

| | |
|---|---|
| Env var | `LORACTL_RUN_RETENTION` |
| Default | `32` |
| Applies to | **completed** runs only |

Two guarantees a client can rely on:

- **An in-flight run is never evicted**, whatever `LORACTL_RUN_RETENTION` is
  set to — only a run that has emitted its terminal event (`finished` or
  `failed`) is a candidate.
- **An open stream is never cut short by eviction.** A subscriber that is
  already streaming an evicted run drains its full history and receives its
  terminal event; eviction only removes the run from the lookup registry, so
  it is a *future* `GET` that 404s.

Consume a run's events while it is fresh; a client that needs history beyond
the retention window must persist it itself. Setting the value higher trades
memory for a longer replay window; `0` evicts each run the moment it finishes.

## Trying it with curl

Start the server (bind address via `LORACTL_API_ADDR`, default
`127.0.0.1:3000`):

```sh
just serve
```

In another terminal, write a config and start a run. Note `output.dir` is
**relative** — it resolves under `LORACTL_OUTPUT_BASE` (default `./runs`), and
an absolute path or a `..` would be rejected with a `400`:

```sh
cat > /tmp/run.json <<'JSON'
{
  "steps": 5,
  "seed": 42,
  "model": { "base": "synthetic" },
  "lora": { "rank": 4, "alpha": 4.0 },
  "dataset": { "path": "./data" },
  "output": { "dir": "demo", "checkpoint_every": 2 }
}
JSON
curl -sX POST localhost:3000/runs -H 'content-type: application/json' -d @/tmp/run.json
```

```json
{"id":1,"events_url":"/runs/1/events"}
```

Stream the run's events (replays from the start, then tails live):

```sh
curl -N localhost:3000/runs/1/events
```

```
id: 0
event: started
data: {"type":"started","total_steps":5}

id: 1
event: warning
data: {"type":"warning","message":"M2 BurnTrainer trains a synthetic LoRA-MLP classifier demo; real base-model + image-dataset ingestion arrives in a later milestone. Build with --features mnist and set model.base=\"mnist\" to train on MNIST."}

id: 2
event: step
data: {"type":"step","step":1,"loss":2.4849,"lr":0.0001}

...

id: 9
event: finished
data: {"type":"finished","adapter_path":"/srv/loractl/runs/demo/lora.safetensors"}
```

(Loss values vary run to run; the frame sequence and shapes are what the
contract pins. With `steps: 5` and `checkpoint_every: 2` the full stream is
`started`, `warning`, five `step`s with `checkpoint`s after steps 2 and 4,
then `finished` — ids 0 through 9. `adapter_path` is the request's relative
`"dir": "demo"` **resolved** under the server's output base, so its prefix is
whatever `LORACTL_OUTPUT_BASE` points at.)

An unknown run id returns a JSON 404:

```sh
curl -s localhost:3000/runs/99/events
```

```json
{"error":"unknown run id"}
```
