//! One check on what the encoder cache hands back: the conditioning stack is
//! exactly `max_length` positions long.
//!
//! The cache ([`crate::dataset`]) keys on
//! [`encoder_fingerprint`](crate::diffusion_trainer::encoder_fingerprint) and
//! validates *rank* on read (`cond_shape.len() != 4`), never *shape*. Between
//! those two facts sits a whole class of silent staleness: any change to how
//! the conditioning is built that does not move the fingerprint leaves the old
//! tensors readable, correctly ranked, and wrong. #163 is the worked example —
//! deriving the caption template's prefix/suffix offsets from the tokenizer
//! changed the emitted stack on the offline stub from 18 positions (two of them
//! template text) to 16, under an unchanged `tinykrea2-ml16-enc32` — and its
//! only shipped mitigation was a doc comment asking a human to delete a
//! directory.
//!
//! The fingerprint is the primary defence and it has been bumped (`-t2`). This
//! is the mechanical one: `max_length` is a property of the variant the run is
//! *currently* configured for, so a cached stack of any other length cannot
//! belong to this run whatever wrote it. It costs one `dims()` per example on
//! tensors already in memory, and it turns "trains on misaligned conditioning,
//! no error, no shape mismatch, worse adapter" into a bail naming the cache
//! directory.
//!
//! It also catches the half-warm case the fingerprint cannot: two alignments
//! inside one `.loractl-cache/`, where the examples encoded before a change
//! and after it are both accepted by a rank check.
//!
//! Takes plain lengths rather than tensors so it is testable without a
//! backend, and lives outside `dataset.rs` so the check reads as what it is —
//! a guard on the *conditioning contract* ([`crate::qwen3vl`]), not part of
//! the file/bucket/cache pipeline.

use anyhow::{Result, bail};
use std::path::Path;

/// Bail unless every cached conditioning stack is `max_length` positions long.
///
/// `dataset` is the dataset directory (the cache lives in its
/// `.loractl-cache/`) and `fingerprint` the key the run just read under; both
/// are quoted so the operator is told what to delete rather than left to
/// deduce it.
pub fn check_conditioning_lengths(
    lengths: impl IntoIterator<Item = usize>,
    max_length: usize,
    dataset: &Path,
    fingerprint: &str,
) -> Result<()> {
    for (index, len) in lengths.into_iter().enumerate() {
        if len != max_length {
            bail!(
                "cached conditioning for example {index} is {len} positions \
                 long, but this run's variant emits {max_length} — the cache \
                 under {}/.loractl-cache (fingerprint {fingerprint}) was \
                 written by a different conditioning build and its positions \
                 no longer line up with the captions. Delete that directory \
                 and re-run the encode phase; training on it would cost \
                 adapter quality and raise no error.",
                dataset.display()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stack_of_the_configured_length_passes() {
        check_conditioning_lengths([16, 16, 16], 16, Path::new("/data"), "tinykrea2-ml16")
            .expect("matching lengths are the normal case");
        // Empty is not a length error — an empty dataset is `scan_dataset`'s
        // to refuse, and this guard must not invent a second opinion.
        check_conditioning_lengths([], 16, Path::new("/data"), "tinykrea2-ml16").unwrap();
    }

    /// The #163 shape verbatim: the pre-change stack carried the template's
    /// two leading positions, so it is LONGER, correctly ranked, and silently
    /// misaligned.
    #[test]
    fn a_pre_163_stack_is_refused_by_length_and_names_the_cache() {
        let err = check_conditioning_lengths(
            [16, 18],
            16,
            Path::new("/data/dogs"),
            "tinykrea2-ml16-enc32-t2",
        )
        .expect_err("18 positions cannot belong to a 16-position run");
        let msg = format!("{err}");
        assert!(msg.contains("example 1"), "{msg}");
        assert!(msg.contains("18 positions"), "{msg}");
        assert!(msg.contains("emits 16"), "{msg}");
        assert!(msg.contains("/data/dogs/.loractl-cache"), "{msg}");
        assert!(msg.contains("tinykrea2-ml16-enc32-t2"), "{msg}");
    }
}
