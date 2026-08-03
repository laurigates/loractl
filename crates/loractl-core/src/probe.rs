//! ADR-0005 attribution probe markers (#132).
//!
//! When the `LORACTL_RETENTION_LEDGER` env var names a file, this module
//! appends `PHASE` marker lines to it, segmenting a run into
//! forward/backward/optimizer windows per step.
//!
//! During the #132 attribution round a burn-autodiff fork pin (PR #133's
//! workspace `[patch.crates-io]` block) also appended one line per
//! checkpoint/retention event to the *same* file, and the markers existed to
//! segment that event stream. The pin was removed once #132 closed, so today
//! only the `PHASE` markers are written; see the `ledger-probe` justfile
//! recipe for how to re-pin the fork if the event lines are needed again.
//!
//! Not rendering: nothing here writes to stdout/stderr or the event sink —
//! it is opt-in diagnostics to a caller-named file, a no-op unless the env
//! var is set. Each marker opens the file in append mode and closes it, so
//! there is no shared handle with burn's writer; `O_APPEND` keeps whole-line
//! writes ordered within the single training process.

use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

fn path() -> Option<&'static PathBuf> {
    PATH.get_or_init(|| std::env::var_os("LORACTL_RETENTION_LEDGER").map(PathBuf::from))
        .as_ref()
}

/// Append a `PHASE\t<name>\t<step>` marker to the retention ledger.
/// No-op when the ledger is inactive.
pub fn phase(name: &str, step: u64) {
    if let Some(p) = path()
        && let Ok(mut f) = std::fs::File::options().create(true).append(true).open(p)
    {
        let _ = writeln!(f, "PHASE\t{name}\t{step}");
    }
}
