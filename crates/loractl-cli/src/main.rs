//! `loractl` binary entry point. All logic lives in [`cli`]; `main` just
//! wires it up and lets errors propagate to a non-zero exit.

mod cli;

use anyhow::Result;

fn main() -> Result<()> {
    cli::run()
}
