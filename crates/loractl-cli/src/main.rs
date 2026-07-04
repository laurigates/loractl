//! `loractl` binary entry point. All logic lives in [`cli`]; `main` just
//! wires it up, brings up telemetry, and lets errors propagate to a non-zero
//! exit.

mod cli;

use anyhow::Result;

fn main() -> Result<()> {
    // Bring up GlitchTip telemetry + tracing before anything else so early
    // failures are captured. The guard lives for the whole process; dropping
    // it at the end of `main` flushes buffered events.
    let _telemetry = cli::init_telemetry();

    let result = cli::run();
    if let Err(err) = &result {
        // Report the fatal top-level error to GlitchTip before we exit. Panics
        // are captured automatically by the Sentry panic integration.
        sentry::integrations::anyhow::capture_anyhow(err);
    }
    result
}
