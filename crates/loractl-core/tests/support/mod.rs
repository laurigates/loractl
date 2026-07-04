//! Shared test-support helpers for `loractl-core`'s integration tests.
//!
//! Included via `mod support;` from each test file that needs it. Named
//! `support/mod.rs` (not `support.rs`) so Cargo treats it as a plain module
//! rather than compiling it as its own top-level test binary — only direct
//! files under `tests/` get that treatment.

use std::path::PathBuf;

/// A unique temp output dir so concurrent test runs don't collide or litter
/// the repo. Removed on drop.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("loractl-{tag}-{}-{nanos}", std::process::id()));
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
