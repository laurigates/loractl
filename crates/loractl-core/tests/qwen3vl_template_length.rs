//! The caption-conditioning length invariant (#163), pinned offline.
//!
//! `Qwen3VlConditioner` used to slice the conditioning at a **hardcoded** 34
//! and budget the body against a hardcoded 5 — transcribed from
//! `krea-ai/krea-2`'s `encoder.py` and never checked against the tokenizer
//! that was actually loaded. Two failure modes rode on that, and the second
//! is the reason a shape test alone is not enough:
//!
//! 1. **Wrong emitted length.** `emitted = max_length - suffix_start + suffix_len`,
//!    so the MMDiT gets `max_length` text positions only if those two agree.
//! 2. **Right shape, wrong content.** The prefix length cancels out of the
//!    emitted length entirely (`+prefix` in the body budget, `−prefix` in the
//!    slice), so a wrong prefix produces a *correctly shaped* stack whose
//!    leading positions are still template text. Nothing downstream can see
//!    it.
//!
//! Both are now unrepresentable: `new` measures the template through the
//! loaded tokenizer. These tests pin **the invariant**, not the numbers —
//! deliberately, because pinning 34/5 offline would need the real Qwen BPE
//! vocabulary (a multi-GB download that is not, and should not be, checked
//! in). The 34/5 claim is asserted where it can actually be tested, on the
//! real tokenizer, by `tests/qwen3vl_real.rs` (`--features qwen3vl-real`),
//! against a golden the reference generator derives from that tokenizer
//! rather than transcribing.
//!
//! Two tokenizers are exercised, so no single fixture's quirk can make the
//! assertions vacuous: the checked-in tiny-krea2 stub (`Whitespace`
//! pre-tokenizer, 36/7 — note that is *not* 34/5, i.e. the offline path was
//! silently misaligned before this fix), and a `WhitespaceSplit` twin built
//! here whose token count the test derives **independently** of the
//! `tokenizers` crate, via `str::split_whitespace`.
//!
//! Scope note: the "prefix + caption tokenizes as prefix-ids ++ caption-ids"
//! property the mask assertions lean on holds because `PROMPT_PREFIX` ends in
//! a newline and both tokenizers here split on whitespace. For a real BPE it
//! is near-certain but not guaranteed, which is why the real path is covered
//! by the id-level comparison in `qwen3vl_real.rs` instead.

use burn::backend::NdArray;
use burn::tensor::Int;
use burn::tensor::Tensor;
use loractl_core::qwen3vl::{
    PROMPT_PREFIX, PROMPT_SUFFIX, Qwen3VlConditioner, Qwen3VlConfig, Qwen3VlEncoder,
};
use std::path::{Path, PathBuf};

type B = NdArray;

/// The checked-in stub the whole offline tiny-krea2 path tokenizes with.
const STUB_TOKENIZER: &str = "tests/fixtures/tiny-krea2/tokenizer/tokenizer.json";

/// A unique temp dir, removed on drop (same shape as `diffusion_trainer.rs`'s
/// — the workspace has no `tempfile` dev-dependency and this does not earn
/// one).
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("loractl-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    /// Write a `tokenizer.json` with the given `WordLevel` vocab behind a
    /// `WhitespaceSplit` pre-tokenizer — whose tokenization is, by
    /// definition, `str::split_whitespace`. That is what lets the test know
    /// the expected token counts without asking the crate under test.
    fn whitespace_split_tokenizer(&self, name: &str, vocab: &str) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(
            &path,
            format!(
                r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
                    "normalizer":null,"pre_tokenizer":{{"type":"WhitespaceSplit"}},
                    "post_processor":null,"decoder":null,
                    "model":{{"type":"WordLevel","vocab":{vocab},"unk_token":"<unk>"}}}}"#
            ),
        )
        .unwrap();
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The twin's vocabulary: pad + unk + the three caption words. Every template
/// word maps to `<unk>` — irrelevant here, since only lengths and the mask
/// are asserted.
const TWIN_VOCAB: &str = r#"{"<unk>":0,"<|endoftext|>":1,"a":2,"red":3,"fox":4}"#;

/// A conditioner over a random-init tiny trunk: only shapes and the mask are
/// under test, never the hidden values.
fn conditioner(tokenizer: &Path, max_length: usize) -> anyhow::Result<Qwen3VlConditioner<B>> {
    let device = Default::default();
    let encoder = Qwen3VlEncoder::<B>::init(Qwen3VlConfig::tiny(), &device);
    Qwen3VlConditioner::new(encoder, tokenizer, max_length)
}

/// `Qwen3VlConditioner` is not `Debug` (it holds a tokenizer and a trunk), so
/// `Result::expect_err` is unavailable — take the error by hand.
fn build_error(tokenizer: &Path, max_length: usize, why: &str) -> String {
    match conditioner(tokenizer, max_length) {
        Ok(_) => panic!("{why}"),
        Err(e) => e.to_string(),
    }
}

fn row(mask: Tensor<B, 2, Int>) -> Vec<i64> {
    mask.into_data().convert::<i64>().into_vec::<i64>().unwrap()
}

/// Failure mode 1: the MMDiT must get exactly `max_length` text positions,
/// whatever the tokenizer makes of the template.
#[test]
fn emitted_conditioning_is_exactly_max_length() {
    let tmp = TempDir::new("qwen3vl-tmpl");
    let twin = tmp.whitespace_split_tokenizer("twin.json", TWIN_VOCAB);
    let device = Default::default();

    for tokenizer in [Path::new(STUB_TOKENIZER), twin.as_path()] {
        // 8 is the smallest that clears the stub's 7-token suffix (see
        // `max_length_below_the_suffix_is_a_clear_error`); 33 is odd on
        // purpose, so nothing can pass by dividing evenly.
        for max_length in [8usize, 16, 33] {
            let c = conditioner(tokenizer, max_length).expect("build conditioner");
            let (_, _, [_, s]) = c.tokenize(&["a red fox"]).expect("tokenize");
            assert_eq!(
                s,
                max_length + c.prefix_len(),
                "{tokenizer:?} @ {max_length}: token grid is body + suffix = max_length + prefix"
            );

            let (cond, mask) = c.encode_captions(&["a red fox"], &device).expect("encode");
            // tiny(): select_layers = [1, 3] -> 2 selected states, hidden 32.
            assert_eq!(
                cond.dims(),
                [1, max_length, 2, 32],
                "{tokenizer:?} @ {max_length}: conditioning must be max_length long"
            );
            assert_eq!(mask.dims(), [1, max_length]);
        }
    }
}

/// Failure mode 2 — the silent one. The lengths must be the *tokenizer's*,
/// and the slice must land on the first caption token, which only a
/// content-level assertion can see: every dims check above passes just as
/// happily with the slice one position off.
#[test]
fn prefix_and_suffix_lengths_are_the_tokenizers_own() {
    let tmp = TempDir::new("qwen3vl-twin");
    let twin = tmp.whitespace_split_tokenizer("twin.json", TWIN_VOCAB);
    let device = Default::default();
    let max_length = 16;
    let c = conditioner(&twin, max_length).expect("build conditioner");

    // `WhitespaceSplit` == `str::split_whitespace`, so the expected counts are
    // derived here rather than transcribed (21 and 2 as of this template).
    assert_eq!(c.prefix_len(), PROMPT_PREFIX.split_whitespace().count());
    assert_eq!(c.suffix_len(), PROMPT_SUFFIX.split_whitespace().count());

    let (_, mask) = c.encode_captions(&["a red fox"], &device).expect("encode");
    // 3 caption tokens, then the body's padding, then the suffix — and
    // nothing of the template ahead of them. An off-by-one slice lengthens
    // the leading run of 1s while every shape stays right.
    let mut want = vec![1i64; 3];
    want.resize(max_length - c.suffix_len(), 0);
    want.extend(std::iter::repeat_n(1i64, c.suffix_len()));
    assert_eq!(row(mask), want, "the slice must start at the caption");
}

/// The checked-in stub's measured numbers, so a future edit to the fixture's
/// vocabulary or to the template fails *here*, with a reason, rather than as
/// a mystery loss change in the tiny-krea2 end-to-end tests.
#[test]
fn checked_in_stub_tokenizer_lengths_are_pinned() {
    let device = Default::default();
    let max_length = 16; // TinyKrea2's `encoder_max_length`.
    let c = conditioner(Path::new(STUB_TOKENIZER), max_length).expect("build conditioner");

    // Measured, not transcribed: the stub's `Whitespace` pre-tokenizer
    // (`\w+|[^\w\s]+`) shreds the template's `<|im_start|>`-style markers into
    // pieces the real BPE keeps whole, so it lands on 36/7 where the real
    // tokenizer lands on 34/5. Before #163 the hardcoded 34 therefore left
    // the first two conditioning positions holding template text on every
    // offline run of this path.
    assert_eq!((c.prefix_len(), c.suffix_len()), (36, 7));

    let (_, mask) = c.encode_captions(&["a red fox"], &device).expect("encode");
    let mut want = vec![1i64; 3];
    want.resize(max_length - c.suffix_len(), 0);
    want.extend(std::iter::repeat_n(1i64, c.suffix_len()));
    assert_eq!(row(mask), want);
}

/// Derived lengths make `body_len`'s subtraction runtime data, so the
/// degenerate case has to be an error rather than a `usize` underflow panic.
#[test]
fn max_length_below_the_suffix_is_a_clear_error() {
    let msg = build_error(
        Path::new(STUB_TOKENIZER),
        3,
        "max_length 3 is below the stub's 7 suffix tokens and must not build",
    );
    assert!(msg.contains('3') && msg.contains('7'), "unhelpful: {msg}");
}

/// The pad-token lookup moved to construction (it is tokenizer state, not
/// per-batch state), so a tokenizer that cannot pad fails where the path that
/// named it is still in scope.
#[test]
fn a_tokenizer_without_the_pad_token_fails_at_construction() {
    let tmp = TempDir::new("qwen3vl-nopad");
    let no_pad = tmp.whitespace_split_tokenizer("no_pad.json", r#"{"<unk>":0,"a":1}"#);
    let msg = build_error(
        &no_pad,
        16,
        "a tokenizer without the pad token must not build",
    );
    assert!(msg.contains("<|endoftext|>"), "unhelpful: {msg}");
}
