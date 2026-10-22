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
//! the **gated** `krea/Krea-2-Raw` repo (the exact tokenizer the encoder
//! loractl targets was trained with), verifies a **pinned SHA-256**, and caches
//! it under `HF_HOME` (or the platform cache dir). The fetch is lazy: a run
//! whose tokenizer is already on disk never touches the network, so offline CI
//! and the existing snapshot-dir flow are unaffected.
//!
//! # The repo is gated (#200)
//!
//! `krea/Krea-2-Raw` is `gated: "auto"`: an anonymous GET answers **401**, and
//! a valid token whose account has not accepted the terms answers **403**. So
//! the download sends `Authorization: Bearer` when a token is discoverable
//! (`discover_token`: `HF_TOKEN`, then `$HF_HOME/token`, then
//! `~/.cache/huggingface/token`), and maps both statuses to an error that says
//! what to do. The token is only ever put in that header — never in an error
//! message, a log line, or a redirect to another host.
//!
//! A warm cache predates the gating and never reaches the network, which is
//! why the 401 went unnoticed. Callers should resolve the tokenizer **before**
//! loading multi-GB checkpoints so a missing credential fails at startup.

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The canonical source: Krea-2-Raw's shipped Qwen3-VL tokenizer. The repo is
/// **gated** (`gated: "auto"`, #200) — downloading needs a HuggingFace token
/// from an account that has accepted the terms at [`GATE_PAGE`]. Pinning the
/// exact repo+file guarantees the fetched tokenizer matches the encoder Krea 2
/// conditions on; the SHA-256 pin below is what makes any byte-identical copy
/// (e.g. a `model.tokenizer` override) provably the right one.
const TOKENIZER_URL: &str =
    "https://huggingface.co/krea/Krea-2-Raw/resolve/main/tokenizer/tokenizer.json";

/// Where a user accepts the gate's terms.
const GATE_PAGE: &str = "https://huggingface.co/krea/Krea-2-Raw";

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

/// The size + digest a download must match. A struct (rather than the two
/// constants inline) so tests can serve a tiny body from a local stub and still
/// exercise the real check.
struct Pin<'a> {
    len: usize,
    sha256: &'a str,
}

const PINNED: Pin<'static> = Pin {
    len: TOKENIZER_LEN,
    sha256: TOKENIZER_SHA256,
};

/// A trimmed, non-empty token, or `None`.
fn clean_token(raw: &str) -> Option<String> {
    let t = raw.trim();
    (!t.is_empty()).then(|| t.to_owned())
}

/// Find a HuggingFace token, in the order the official client uses:
///
/// 1. `hf_token_env` (the value of `HF_TOKEN`);
/// 2. `<hf_home>/token`;
/// 3. `<home>/.cache/huggingface/token` (the default `HF_HOME`).
///
/// Inputs are explicit so the order is unit-testable without mutating the
/// process environment. An unreadable or empty candidate is skipped, not an
/// error: no token is a valid state (a public repo needs none), and its
/// absence is reported by the 401 mapping in [`download`].
fn discover_token(
    hf_token_env: Option<&str>,
    hf_home: Option<&Path>,
    home: Option<&Path>,
) -> Option<String> {
    if let Some(t) = hf_token_env.and_then(clean_token) {
        return Some(t);
    }
    let candidates = [
        hf_home.map(|h| h.join("token")),
        home.map(|h| h.join(".cache").join("huggingface").join("token")),
    ];
    candidates.into_iter().flatten().find_map(|p| {
        std::fs::read_to_string(p)
            .ok()
            .as_deref()
            .and_then(clean_token)
    })
}

/// [`discover_token`] over the real process environment.
fn token_from_env() -> Option<String> {
    let hf_token = std::env::var("HF_TOKEN").ok();
    let hf_home = std::env::var_os("HF_HOME").map(PathBuf::from);
    let home = std::env::var_os("HOME").map(PathBuf::from);
    discover_token(hf_token.as_deref(), hf_home.as_deref(), home.as_deref())
}

/// Returns the path to the cached Qwen3-VL tokenizer, fetching+verifying it on
/// the first call and reusing the (re-verified) cache thereafter.
///
/// The network is touched only when the cache is absent or fails its checksum;
/// a run whose tokenizer is already resolved locally never calls this. The
/// source repo is gated, so a cold-cache fetch needs a token — see the module
/// docs; call this **early** so a missing credential is found at startup.
pub fn fetch_qwen3vl_tokenizer() -> Result<PathBuf> {
    fetch_with(
        &cache_dir()?,
        TOKENIZER_URL,
        token_from_env().as_deref(),
        &PINNED,
    )
}

/// [`fetch_qwen3vl_tokenizer`] with the cache dir, URL, token and pin injected.
fn fetch_with(dir: &Path, url: &str, token: Option<&str>, pin: &Pin<'_>) -> Result<PathBuf> {
    let cached = dir.join(CACHE_NAME);

    // Cache hit: reuse only if it still matches the pin (guards a truncated or
    // tampered cache).
    if let Ok(bytes) = std::fs::read(&cached)
        && bytes.len() == pin.len
        && sha256_hex(&bytes) == pin.sha256
    {
        return Ok(cached);
    }

    let bytes = download(url, token)?;
    if bytes.len() != pin.len {
        bail!(
            "fetched Qwen3-VL tokenizer is {} bytes, expected {} — \
             refusing a corrupt or changed download",
            bytes.len(),
            pin.len
        );
    }
    let got = sha256_hex(&bytes);
    if got != pin.sha256 {
        bail!(
            "fetched Qwen3-VL tokenizer SHA-256 {got} != pinned {} — \
             refusing a corrupt or tampered download",
            pin.sha256
        );
    }

    // Write atomically: a temp file + rename so a concurrent run (or a crash
    // mid-write) never leaves a half-written tokenizer at the cache path.
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating tokenizer cache dir {}", dir.display()))?;
    let tmp = dir.join(format!("{CACHE_NAME}.{}.tmp", std::process::id()));
    std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &cached).with_context(|| format!("finalizing {}", cached.display()))?;
    Ok(cached)
}

/// The actionable text for a rejected gated download. `had_token` selects
/// between "you sent nothing" and "what you sent was refused"; neither variant
/// includes the token itself.
fn gate_error(status: u16, had_token: bool) -> String {
    let fixes = format!(
        "accept the terms at {GATE_PAGE} and set HF_TOKEN (or run `huggingface-cli login`), \
         or set model.tokenizer to a local tokenizer.json (it is verified against a pinned \
         SHA-256, so any byte-identical copy works)"
    );
    match (status, had_token) {
        (401, false) => format!(
            "HTTP 401: the Qwen3-VL tokenizer's repo is gated and no HuggingFace token was \
             found — {fixes}"
        ),
        (401, true) => format!(
            "HTTP 401: HuggingFace rejected the token (expired, revoked, or mistyped) — {fixes}"
        ),
        _ => format!(
            "HTTP 403: the token is valid but this account has not accepted the repo's terms — \
             {fixes}"
        ),
    }
}

/// Blocking GET of `url` into bytes, with an overall timeout and a body-size
/// limit sized for the ~11 MB tokenizer. `token`, when present, is sent as
/// `Authorization: Bearer`; it is not forwarded across a redirect to another
/// host (`RedirectAuthHeaders::Never`, set explicitly below), which matters
/// because HF redirects the file to a pre-signed CDN URL.
fn download(url: &str, token: Option<&str>) -> Result<Vec<u8>> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(120)))
        .redirect_auth_headers(ureq::config::RedirectAuthHeaders::Never)
        .build()
        .into();
    let mut req = agent.get(url);
    if let Some(token) = token {
        req = req.header("Authorization", format!("Bearer {token}"));
    }
    let mut resp = match req.call() {
        Ok(resp) => resp,
        Err(ureq::Error::StatusCode(status @ (401 | 403))) => {
            bail!(
                "fetching the Qwen3-VL tokenizer from {url}: {}",
                gate_error(status, token.is_some())
            )
        }
        Err(e) => {
            return Err(e).with_context(|| format!("fetching the Qwen3-VL tokenizer from {url}"));
        }
    };
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

    // ---- offline tests for the gated-download path (#200) -------------------

    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A unique scratch dir, removed on drop (no `tempfile` dev-dep in core).
    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            static N: AtomicU32 = AtomicU32::new(0);
            let dir = std::env::temp_dir().join(format!(
                "loractl-hf-test-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Serve exactly one request on a loopback port with a canned response.
    /// Returns the URL and a handle yielding the raw request text, so a test
    /// can assert on the headers the client actually sent.
    fn stub(status_line: &'static str, body: Vec<u8>) -> (String, std::thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/tokenizer.json", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut req = Vec::new();
            let mut buf = [0u8; 1024];
            while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = sock.read(&mut buf).unwrap();
                if n == 0 {
                    break;
                }
                req.extend_from_slice(&buf[..n]);
            }
            let head = format!(
                "HTTP/1.1 {status_line}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            sock.write_all(head.as_bytes()).unwrap();
            sock.write_all(&body).unwrap();
            String::from_utf8_lossy(&req).into_owned()
        });
        (url, handle)
    }

    const BODY: &[u8] = b"{\"stub\":\"tokenizer\"}";

    #[test]
    fn token_discovery_order_is_env_then_hf_home_then_default_cache() {
        let s = Scratch::new();
        let hf_home = s.0.join("hfhome");
        let home = s.0.join("home");
        std::fs::create_dir_all(&hf_home).unwrap();
        std::fs::create_dir_all(home.join(".cache/huggingface")).unwrap();
        std::fs::write(hf_home.join("token"), "  from-hf-home\n").unwrap();
        std::fs::write(home.join(".cache/huggingface/token"), "from-default\n").unwrap();

        // 1. the env var wins over both files, and is trimmed.
        assert_eq!(
            discover_token(Some(" from-env "), Some(&hf_home), Some(&home)).as_deref(),
            Some("from-env")
        );
        // 2. no env: $HF_HOME/token beats the default location.
        assert_eq!(
            discover_token(None, Some(&hf_home), Some(&home)).as_deref(),
            Some("from-hf-home")
        );
        // 3. no env, no HF_HOME: ~/.cache/huggingface/token.
        assert_eq!(
            discover_token(None, None, Some(&home)).as_deref(),
            Some("from-default")
        );
    }

    #[test]
    fn token_discovery_skips_empty_and_missing_candidates() {
        let s = Scratch::new();
        let hf_home = s.0.join("hfhome"); // exists, but its token is blank
        let home = s.0.join("home"); // has no token file at all
        std::fs::create_dir_all(&hf_home).unwrap();
        std::fs::write(hf_home.join("token"), " \n").unwrap();

        // A blank env var and a blank file fall through rather than yielding
        // an empty bearer token.
        assert_eq!(
            discover_token(Some("  "), Some(&hf_home), Some(&home)),
            None
        );
        assert_eq!(discover_token(None, None, None), None);
    }

    /// A pin matching [`BODY`].
    fn body_pin_sha() -> String {
        sha256_hex(BODY)
    }

    #[test]
    fn authenticated_download_sends_the_bearer_header_and_caches_the_verified_body() {
        let s = Scratch::new();
        let sha = body_pin_sha();
        let (url, req) = stub("200 OK", BODY.to_vec());
        let path = fetch_with(
            &s.0,
            &url,
            Some("hf_secret123"),
            &Pin {
                len: BODY.len(),
                sha256: &sha,
            },
        )
        .expect("a 200 with the pinned body succeeds");
        assert_eq!(std::fs::read(&path).unwrap(), BODY);
        let req = req.join().unwrap().to_ascii_lowercase();
        assert!(
            req.contains("authorization: bearer hf_secret123"),
            "the token must be sent as a Bearer header, got:\n{req}"
        );
    }

    #[test]
    fn anonymous_download_sends_no_authorization_header() {
        let s = Scratch::new();
        let sha = body_pin_sha();
        let (url, req) = stub("200 OK", BODY.to_vec());
        fetch_with(
            &s.0,
            &url,
            None,
            &Pin {
                len: BODY.len(),
                sha256: &sha,
            },
        )
        .unwrap();
        let req = req.join().unwrap().to_ascii_lowercase();
        assert!(!req.contains("authorization"), "got:\n{req}");
    }

    #[test]
    fn a_401_without_a_token_says_how_to_get_one() {
        let s = Scratch::new();
        let sha = body_pin_sha();
        let (url, _req) = stub("401 Unauthorized", Vec::new());
        let err = fetch_with(
            &s.0,
            &url,
            None,
            &Pin {
                len: BODY.len(),
                sha256: &sha,
            },
        )
        .unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("401"), "{text}");
        assert!(text.contains("no HuggingFace token was found"), "{text}");
        assert!(text.contains(GATE_PAGE), "{text}");
        assert!(text.contains("HF_TOKEN"), "{text}");
        assert!(text.contains("model.tokenizer"), "{text}");
        assert!(
            !s.0.join(CACHE_NAME).exists(),
            "a failed fetch must cache nothing"
        );
    }

    #[test]
    fn a_401_with_a_token_says_the_token_was_rejected_and_never_echoes_it() {
        let s = Scratch::new();
        let sha = body_pin_sha();
        let (url, _req) = stub("401 Unauthorized", Vec::new());
        let err = fetch_with(
            &s.0,
            &url,
            Some("hf_secret123"),
            &Pin {
                len: BODY.len(),
                sha256: &sha,
            },
        )
        .unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("rejected the token"), "{text}");
        assert!(
            !text.contains("hf_secret123"),
            "the token must never appear in an error: {text}"
        );
    }

    #[test]
    fn a_403_says_the_token_is_valid_but_the_terms_are_not_accepted() {
        let s = Scratch::new();
        let sha = body_pin_sha();
        let (url, _req) = stub("403 Forbidden", Vec::new());
        let err = fetch_with(
            &s.0,
            &url,
            Some("hf_secret123"),
            &Pin {
                len: BODY.len(),
                sha256: &sha,
            },
        )
        .unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("403"), "{text}");
        assert!(text.contains("token is valid"), "{text}");
        assert!(text.contains("not accepted"), "{text}");
        assert!(text.contains(GATE_PAGE), "{text}");
        assert!(!text.contains("hf_secret123"), "{text}");
    }

    #[test]
    fn other_statuses_are_not_misreported_as_the_gate() {
        let s = Scratch::new();
        let sha = body_pin_sha();
        let (url, _req) = stub("500 Internal Server Error", Vec::new());
        let err = fetch_with(
            &s.0,
            &url,
            None,
            &Pin {
                len: BODY.len(),
                sha256: &sha,
            },
        )
        .unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("500"), "{text}");
        assert!(!text.contains("gated"), "{text}");
    }

    #[test]
    fn an_authenticated_download_still_fails_the_pinned_checks() {
        // Auth must not weaken provenance: a 200 with the wrong bytes is
        // refused (length, then digest), and nothing is cached.
        let s = Scratch::new();
        let sha = body_pin_sha();

        let (url, _req) = stub("200 OK", BODY.to_vec());
        let err = fetch_with(
            &s.0,
            &url,
            Some("t"),
            &Pin {
                len: BODY.len() + 1,
                sha256: &sha,
            },
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("bytes, expected"), "{err:#}");

        let (url, _req) = stub("200 OK", BODY.to_vec());
        let wrong = "0".repeat(64);
        let err = fetch_with(
            &s.0,
            &url,
            Some("t"),
            &Pin {
                len: BODY.len(),
                sha256: &wrong,
            },
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("SHA-256"), "{err:#}");
        assert!(!s.0.join(CACHE_NAME).exists());
    }

    #[test]
    fn a_verified_cache_hit_never_touches_the_network() {
        let s = Scratch::new();
        let sha = body_pin_sha();
        std::fs::write(s.0.join(CACHE_NAME), BODY).unwrap();
        // Port 1 on loopback: connecting would fail, so Ok proves no request.
        let path = fetch_with(
            &s.0,
            "http://127.0.0.1:1/never",
            None,
            &Pin {
                len: BODY.len(),
                sha256: &sha,
            },
        )
        .expect("warm cache resolves offline");
        assert_eq!(path, s.0.join(CACHE_NAME));
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
