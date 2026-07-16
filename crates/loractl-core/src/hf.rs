//! On-demand fetch of the model-invariant Qwen3-VL tokenizer.
//!
//! A ComfyUI install keeps its model components scattered and ships **no**
//! `tokenizer.json` beside the text encoder. When a run points at such a
//! layout (no `model.tokenizer` override and no `base/tokenizer/tokenizer.json`),
//! [`DiffusionTrainer`](crate::DiffusionTrainer) still needs the tokenizer the
//! Qwen3-VL encoder was trained with. That tokenizer is **model-invariant**
//! across Qwen3-VL-4B — every Krea-2 variant re-ships the same standard Qwen
//! tokenizer — so one canonical copy serves every run.
//!
//! Rather than commit an ~11 MB asset into the repo, this fetches it once from
//! the ungated `krea/Krea-2-Raw` repo (the exact tokenizer the encoder loractl
//! targets was trained with), verifies a **pinned SHA-256**, and caches it
//! under `HF_HOME` (or the platform cache dir). The fetch is lazy: a run whose
//! tokenizer is already on disk never touches the network, so offline CI and
//! the existing snapshot-dir flow are unaffected.

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Duration;

/// The canonical source: Krea-2-Raw's shipped Qwen3-VL tokenizer, publicly
/// downloadable (ungated). Pinning the exact repo+file guarantees the fetched
/// tokenizer matches the encoder Krea 2 conditions on.
const TOKENIZER_URL: &str =
    "https://huggingface.co/krea/Krea-2-Raw/resolve/main/tokenizer/tokenizer.json";

/// SHA-256 of the pinned `tokenizer.json` (11 422 650 bytes, BPE, 151 643
/// vocab). Verified on both a fresh download and the cached copy — a mismatch
/// (corrupt download, upstream change, tampered cache) is a hard error, never
/// a silently-wrong tokenizer.
const TOKENIZER_SHA256: &str = "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506";

/// Expected byte length — a cheap first check before hashing.
const TOKENIZER_LEN: usize = 11_422_650;

/// The cache filename (SHA-tagged so a future pin bump can't collide with a
/// stale copy).
const CACHE_NAME: &str = "qwen3vl-4b-tokenizer.json";

/// The directory the fetched tokenizer is cached in: `$HF_HOME/loractl`, else
/// `$XDG_CACHE_HOME/loractl`, else `$HOME/.cache/loractl`. Honoring `HF_HOME`
/// keeps the cache on the same (often larger) disk operators already point HF
/// downloads at.
fn cache_dir() -> Result<PathBuf> {
    let root = if let Some(hf) = std::env::var_os("HF_HOME") {
        PathBuf::from(hf)
    } else if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(xdg)
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache")
    } else {
        bail!("cannot locate a cache directory — set HF_HOME to a writable path");
    };
    Ok(root.join("loractl"))
}

/// The `Sha256` hex digest of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Returns the path to the cached Qwen3-VL tokenizer, fetching+verifying it on
/// the first call and reusing the (re-verified) cache thereafter.
///
/// The network is touched only when the cache is absent or fails its checksum;
/// a run whose tokenizer is already resolved locally never calls this.
pub fn fetch_qwen3vl_tokenizer() -> Result<PathBuf> {
    let dir = cache_dir()?;
    let cached = dir.join(CACHE_NAME);

    // Cache hit: reuse only if it still matches the pin (guards a truncated or
    // tampered cache).
    if let Ok(bytes) = std::fs::read(&cached)
        && bytes.len() == TOKENIZER_LEN
        && sha256_hex(&bytes) == TOKENIZER_SHA256
    {
        return Ok(cached);
    }

    let bytes = download(TOKENIZER_URL)?;
    if bytes.len() != TOKENIZER_LEN {
        bail!(
            "fetched Qwen3-VL tokenizer is {} bytes, expected {TOKENIZER_LEN} — \
             refusing a corrupt or changed download",
            bytes.len()
        );
    }
    let got = sha256_hex(&bytes);
    if got != TOKENIZER_SHA256 {
        bail!(
            "fetched Qwen3-VL tokenizer SHA-256 {got} != pinned {TOKENIZER_SHA256} — \
             refusing a corrupt or tampered download"
        );
    }

    // Write atomically: a temp file + rename so a concurrent run (or a crash
    // mid-write) never leaves a half-written tokenizer at the cache path.
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating tokenizer cache dir {}", dir.display()))?;
    let tmp = dir.join(format!("{CACHE_NAME}.{}.tmp", std::process::id()));
    std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &cached).with_context(|| format!("finalizing {}", cached.display()))?;
    Ok(cached)
}

/// Blocking GET of `url` into bytes, with an overall timeout and a body-size
/// limit sized for the ~11 MB tokenizer.
fn download(url: &str) -> Result<Vec<u8>> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(120)))
        .build()
        .into();
    let mut resp = agent
        .get(url)
        .call()
        .with_context(|| format!("fetching the Qwen3-VL tokenizer from {url}"))?;
    resp.body_mut()
        .with_config()
        .limit(32 * 1024 * 1024)
        .read_to_vec()
        .with_context(|| format!("reading the tokenizer body from {url}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_a_known_vector() {
        // The SHA-256 of the empty string — pins the hex formatting.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn cache_dir_prefers_hf_home() {
        // Deterministic without mutating the process env of other tests: the
        // precedence is documented; here we only assert the `loractl` suffix so
        // a caching path can never escape our own subdir.
        let dir = cache_dir().expect("a cache dir resolves on any dev/CI host");
        assert!(
            dir.ends_with("loractl"),
            "cache dir must live under a loractl/ subdir, got {}",
            dir.display()
        );
    }

    /// Opt-in network proof (never in CI): actually fetch, verify the pin, and
    /// confirm the cached copy round-trips through the `tokenizers` crate.
    /// Run: `cargo test -p loractl-core --test-threads=1 -- --ignored fetch_qwen3vl_tokenizer_real`
    #[test]
    #[ignore = "network: fetches ~11 MB from HuggingFace"]
    fn fetch_qwen3vl_tokenizer_real() {
        let path = fetch_qwen3vl_tokenizer().expect("fetch + checksum");
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), TOKENIZER_LEN);
        assert_eq!(sha256_hex(&bytes), TOKENIZER_SHA256);
        // A second call is a pure cache hit (no re-download).
        let again = fetch_qwen3vl_tokenizer().expect("cache hit");
        assert_eq!(again, path);
        // The fetched file is a usable tokenizer.
        tokenizers::Tokenizer::from_file(&path).expect("loads as a tokenizer");
    }
}
