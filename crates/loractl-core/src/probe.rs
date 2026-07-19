//! ADR-0005 attribution probe markers (#132).
//!
//! When the `LORACTL_RETENTION_LEDGER` env var names a file, the burn-autodiff
//! fork pin (see the workspace `[patch.crates-io]`) appends one line per
//! checkpoint/retention event to it. This module appends `PHASE` marker lines
//! to the *same* file so the event stream can be segmented into
//! forward/backward/optimizer windows per step.
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
