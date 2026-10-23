//! Decode-fraction probe (#217) — what does the dataset scan's pixel work cost,
//! in isolation, serial versus the bounded-window parallel form?
//!
//! The encode phase's wall clock cannot answer that. Measured on the 4090 box
//! (512px, `linoyts/Tuxemon`) decode + Lanczos resize + crop is ~36 ms/image
//! against an encode phase of 598.7 s/image before #211 and 68.6 s/image after,
//! so the decode is 0.006% / 0.05% of the phase: an 8x decode speedup moves a
//! 40-image cold encode by about a second, against run-to-run variance in
//! minutes. A comparison at that level returns noise whichever way the
//! implementation went — and would return the same noise against a sabotaged
//! decode. This probe removes the 4B-parameter text-encoder forward from the
//! measurement, which is what makes the decode resolvable at all, so it can
//! price #197's rewrite (a rayon pre-pass, no full-frame `to_image()` copy, a
//! row-wise split-plane CHW write) and catch a regression in it.
//!
//! CPU-only: no GPU, no model weights, no backend. Runs in seconds.
//!
//! ## What is timed
//!
//! The real public path, `decode_image_for_bucket` — read, decode, Lanczos3
//! cover-resize, center-crop, CHW f32 pack — over the first N images of the
//! directory, in two arms:
//!
//! - `serial`: one image at a time, the pre-#178 interleaving.
//! - `window`: a bounded rayon window, mirroring `prepare_dataset`'s pre-pass
//!   (`dataset.rs`): `window = rayon::current_num_threads().clamp(1, 16)`
//!   images decoded in parallel, collected position-indexed, then the next
//!   window. `RAYON_NUM_THREADS=1` collapses it to the serial arm, exactly as
//!   it does in `prepare_dataset`.
//!
//! `decode_window()` and the pre-pass body are private to `dataset.rs` and the
//! real pre-pass is welded to the encode closures and the latent cache, so
//! calling `prepare_dataset` would time cache I/O and a stand-in encoder as
//! well — the opposite of isolating the decode. The window loop is therefore
//! mirrored here against the one public function that does the pixel work,
//! rather than widening `dataset.rs`'s API for a measurement tool. The cost of
//! that choice is that the mirror can drift from `decode_window()`'s clamp; the
//! `MODEL` line prints the window it used so a reader can compare.
//!
//! Only the decode is inside the timer. Digesting each buffer (the identity
//! check below) and dropping it happen outside it, in both arms alike.
//!
//! ## Output
//!
//! Grep-parseable lines from the `loractl-bench` schema:
//!
//! - `MODEL label=decode_probe …` — the shape behind the figures: image count,
//!   source (`real` or `synthetic`), mean source megapixels, bucket, thread
//!   count, window, and `excludes=` naming what is NOT in the number.
//! - `RESULT label=decode_serial ms=… total_ms=… images=… sanity=… x2_ratio=…`
//!   and `RESULT label=decode_window …` — `ms=` is **per image**.
//! - `RESULT label=decode_delta ms=… saved_total_ms=… speedup_x=…` — the
//!   absolute per-image figure the issue asks for (serial minus window; negative
//!   when the pool costs more than it saves), with the ratio beside it rather
//!   than instead of it.
//! - `SANITY …` per arm — each arm is timed over N and over 2N images (the list
//!   repeated), and total time must roughly double. An elided decode would not.
//!
//! A `sanity=SUSPECT` arm voids that arm's timing. One benign cause on the
//! `window` arm: when N is not a multiple of the window, the N-image pass ends
//! in a short tail window that the 2N pass amortizes differently, so the ratio
//! drifts off 2 without any work having been elided — pick N as a multiple of
//! the printed `window=` (the default 16 is, for 1, 2, 4, 8 or 16 threads).
//! The probe also fails (exit non-zero) if the two arms do not produce
//! byte-identical buffers, or if any buffer is the wrong length or fails
//! `plausible` (all zero / non-finite) — a faster number from a different
//! output is not a speedup.
//!
//! ## Usage
//!
//! ```text
//! just decode-probe                                  # synthetic images, offline
//! just dataset-fetch tuxemon tmp/datasets/tuxemon-56 56
//! just decode-probe tmp/datasets/tuxemon-56 --images 40 --bucket 512x512
//! ```
//!
//! Decode cost is a function of source resolution: the pinned Tuxemon set
//! averages 2.4 Mpx, which is what a user actually feeds this. The synthetic
//! fallback (no directory argument) is JPEG at ~2.5 Mpx in three aspects, so its
//! shape is representative but its pixels are noise — treat the numbers as a
//! smoke figure for the harness, not as the dataset's. The `MODEL` line says
//! `source=synthetic` so a log cannot be misread.
//!
//! Flags: `<dir>` (positional, optional), `--images N` (default 16, capped at
//! what the directory holds), `--bucket WxH` (default 512x512, each side a
//! multiple of 16), `--synthetic-size WxH` (the fallback's base source size,
//! default 1920x1280).

use anyhow::{Context, Result, bail, ensure};
use loractl_bench::{BenchResult, ModelLine, Sanity, plausible};
use loractl_core::dataset::{BUCKET_ALIGN, Bucket, decode_image_for_bucket};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The parsed command line.
#[derive(Debug, PartialEq, Eq)]
struct Args {
    /// `None` runs the synthetic fallback.
    dir: Option<PathBuf>,
    /// `None` means "default, capped at what the directory holds".
    images: Option<usize>,
    bucket: (u32, u32),
    synthetic_size: (u32, u32),
}

const DEFAULT_IMAGES: usize = 16;

fn parse_args_from<I: Iterator<Item = String>>(argv: I) -> Result<Args> {
    let mut args = Args {
        dir: None,
        images: None,
        bucket: (512, 512),
        synthetic_size: (1920, 1280),
    };
    let mut argv = argv;
    while let Some(arg) = argv.next() {
        let mut value = |flag: &str| -> Result<String> {
            argv.next().with_context(|| format!("{flag} needs a value"))
        };
        match arg.as_str() {
            "--images" => {
                let raw = value("--images")?;
                let n: usize = raw
                    .parse()
                    .with_context(|| format!("--images expects a positive integer, got {raw:?}"))?;
                ensure!(n > 0, "--images must be > 0");
                args.images = Some(n);
            }
            "--bucket" => args.bucket = parse_size("--bucket", &value("--bucket")?)?,
            "--synthetic-size" => {
                args.synthetic_size = parse_size("--synthetic-size", &value("--synthetic-size")?)?;
            }
            flag if flag.starts_with("--") => bail!("unknown flag {flag}"),
            _ => {
                ensure!(args.dir.is_none(), "only one image directory may be given");
                args.dir = Some(PathBuf::from(arg));
            }
        }
    }
    let (bw, bh) = args.bucket;
    ensure!(
        bw % BUCKET_ALIGN == 0 && bh % BUCKET_ALIGN == 0,
        "--bucket {bw}x{bh}: each side must be a multiple of {BUCKET_ALIGN}"
    );
    Ok(args)
}

/// `"512x384"` → `(512, 384)`, both sides non-zero.
fn parse_size(flag: &str, raw: &str) -> Result<(u32, u32)> {
    let parsed = raw
        .split_once(['x', 'X'])
        .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)));
    match parsed {
        Some((w, h)) if w > 0 && h > 0 => Ok((w, h)),
        _ => bail!("{flag} expects WxH with positive integers, got {raw:?}"),
    }
}

/// The image files of `dir` in filename order — the order `scan_dataset` walks,
/// so `--images N` is a strict prefix of a larger N, same as `dataset-fetch`.
fn list_images(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("reading image directory {}", dir.display()))?
    {
        let path = entry?.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        if matches!(ext.as_deref(), Some("png" | "jpg" | "jpeg")) {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

/// Deterministic pseudo-noise over a gradient: enough high-frequency content
/// that JPEG does real entropy decoding (a flat image would decode
/// implausibly fast), reproducible so two runs decode the same bytes.
fn synthetic_image(width: u32, height: u32, seed: u64) -> image::RgbImage {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    image::RgbImage::from_fn(width, height, |x, y| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let noise = (state & 0x3f) as u32;
        let ramp = |v: u32, span: u32| ((u64::from(v) * 96) / u64::from(span.max(1))) as u32;
        image::Rgb([
            (ramp(x, width) + noise) as u8,
            (ramp(y, height) + noise) as u8,
            (ramp(x + y, width + height) + noise) as u8,
        ])
    })
}

/// Write `count` JPEGs into `dir`, cycling three aspects off `base` (landscape,
/// portrait, 16:9) so the resize/crop sees more than one scale. Returns the
/// paths in filename order.
fn write_synthetic(dir: &Path, count: usize, base: (u32, u32)) -> Result<Vec<PathBuf>> {
    let (w, h) = base;
    let shapes = [(w, h), (h, w), (w, (w * 9 / 16).max(1))];
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let mut paths = Vec::with_capacity(count);
    for i in 0..count {
        let (sw, sh) = shapes[i % shapes.len()];
        let path = dir.join(format!("synthetic-{i:04}.jpg"));
        synthetic_image(sw, sh, i as u64 + 1)
            .save(&path)
            .with_context(|| format!("writing {}", path.display()))?;
        paths.push(path);
    }
    Ok(paths)
}

/// FNV-1a over the buffer's f32 bit patterns — exact (it distinguishes -0.0
/// and NaN payloads), and cheap enough to run untimed on every buffer.
fn digest(data: &[f32]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
        (h ^ u64::from(v.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// One arm's output: decode time, and a digest per image in input order.
struct Arm {
    elapsed: Duration,
    digests: Vec<u64>,
}

/// Reject a buffer that is the wrong size or the dead-graph kind (all zero,
/// non-finite), then digest it. Untimed.
fn vet(data: &[f32], bucket: Bucket, path: &Path) -> Result<u64> {
    let want = 3 * bucket.width as usize * bucket.height as usize;
    ensure!(
        data.len() == want,
        "{}: decoded {} f32s, expected {want}",
        path.display(),
        data.len()
    );
    ensure!(
        plausible(data),
        "{}: decoded buffer is all-zero or non-finite",
        path.display()
    );
    Ok(digest(data))
}

/// The serial arm: one decode at a time. Only the decode is inside the timer.
fn run_serial(paths: &[PathBuf], bucket: Bucket) -> Result<Arm> {
    let mut elapsed = Duration::ZERO;
    let mut digests = Vec::with_capacity(paths.len());
    for path in paths {
        let t = Instant::now();
        let data = decode_image_for_bucket(path, bucket)?;
        elapsed += t.elapsed();
        digests.push(vet(&data, bucket, path)?);
    }
    Ok(Arm { elapsed, digests })
}

/// MIRRORS the private `decode_window()` in `crates/loractl-core/src/dataset.rs`
/// (`rayon::current_num_threads().clamp(1, 16)`). If that function changes,
/// `the_window_mirrors_dataset_rs_decode_window` fails and this must follow.
fn default_window() -> usize {
    rayon::current_num_threads().clamp(1, MAX_WINDOW)
}

/// The cap in `dataset.rs::decode_window()`. Pinned against the real source by
/// `the_window_mirrors_dataset_rs_decode_window`.
const MAX_WINDOW: usize = 16;

/// The windowed arm: `window` images decoded in parallel, collected
/// position-indexed (so output order is input order), then the next window —
/// the shape of `prepare_dataset`'s pre-pass without its cache and encoder.
fn run_windowed(paths: &[PathBuf], bucket: Bucket, window: usize) -> Result<Arm> {
    let window = window.max(1);
    let mut elapsed = Duration::ZERO;
    let mut digests = Vec::with_capacity(paths.len());
    for chunk in paths.chunks(window) {
        let t = Instant::now();
        let decoded: Vec<Result<Vec<f32>>> = chunk
            .par_iter()
            .map(|path| decode_image_for_bucket(path, bucket))
            .collect();
        elapsed += t.elapsed();
        for (path, data) in chunk.iter().zip(decoded) {
            digests.push(vet(&data?, bucket, path)?);
        }
    }
    Ok(Arm { elapsed, digests })
}

/// The identity gate: a speedup from a different output is not a speedup.
fn check_identical(serial: &[u64], windowed: &[u64], paths: &[PathBuf]) -> Result<()> {
    ensure!(
        serial.len() == windowed.len(),
        "arms decoded different image counts: serial {} vs window {}",
        serial.len(),
        windowed.len()
    );
    for (i, (s, w)) in serial.iter().zip(windowed).enumerate() {
        ensure!(
            s == w,
            "arms disagree on {}: serial digest {s:016x} vs window {w:016x}",
            paths[i].display()
        );
    }
    Ok(())
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Time one arm over `paths` and over `paths` repeated twice, returning the
/// N-image pass (whose digests are checked) and its 2x sanity verdict.
fn timed_arm(run: &dyn Fn(&[PathBuf]) -> Result<Arm>, paths: &[PathBuf]) -> Result<(Arm, Sanity)> {
    let n = paths.len() as u32;
    let one = run(paths)?;
    let doubled: Vec<PathBuf> = paths.iter().chain(paths).cloned().collect();
    let two = run(&doubled)?;
    let sanity = Sanity::from_avgs(one.elapsed / n, two.elapsed / (2 * n));
    Ok((one, sanity))
}

/// Everything the probe prints, as lines, so tests can read them.
fn probe(paths: &[PathBuf], bucket: Bucket, source: &str, window: usize) -> Result<Vec<String>> {
    ensure!(!paths.is_empty(), "no images to decode");
    let n = paths.len();

    // Warm-up, untimed: page cache for the files, the rayon pool's threads, and
    // the allocator. Without it the first arm pays for all three.
    run_windowed(&paths[..window.min(n)], bucket, window)?;
    run_serial(&paths[..1], bucket)?;

    let (serial, serial_sanity) = timed_arm(&|p| run_serial(p, bucket), paths)?;
    let (windowed, window_sanity) = timed_arm(&|p| run_windowed(p, bucket, window), paths)?;
    check_identical(&serial.digests, &windowed.digests, paths)?;

    let mpx = paths
        .iter()
        .map(|p| image::image_dimensions(p).map(|(w, h)| f64::from(w) * f64::from(h) / 1e6))
        .sum::<Result<f64, _>>()
        .context("reading source dimensions")?
        / n as f64;

    let per_image = |d: Duration| d / n as u32;
    let (s_ms, w_ms) = (
        ms(serial.elapsed) / n as f64,
        ms(windowed.elapsed) / n as f64,
    );
    let mut delta = BenchResult::new("decode_delta", Duration::ZERO)
        .with(
            "saved_total_ms",
            format!("{:.4}", ms(serial.elapsed) - ms(windowed.elapsed)),
        )
        .with("speedup_x", format!("{:.3}", s_ms / w_ms));
    // `BenchResult::new` takes a Duration, which cannot be negative; the delta
    // is negative exactly when the pool costs more than it saves.
    delta.ms = s_ms - w_ms;

    let model = ModelLine::new("decode_probe")
        .with("images", n)
        .with("source", source)
        .with("src_mpx_mean", format!("{mpx:.2}"))
        .with("bucket", format!("{}x{}", bucket.width, bucket.height))
        .with("threads", rayon::current_num_threads())
        .with("window", window)
        .with("identical", 1)
        .with(
            "excludes",
            "vae_encode,text_encode,cache_io,digest,drop,device_upload",
        );

    Ok(vec![
        model.to_string(),
        BenchResult::new("decode_serial", per_image(serial.elapsed))
            .with("total_ms", format!("{:.4}", ms(serial.elapsed)))
            .with("images", n)
            .with_sanity(serial_sanity)
            .to_string(),
        serial_sanity.to_string(),
        BenchResult::new("decode_window", per_image(windowed.elapsed))
            .with("total_ms", format!("{:.4}", ms(windowed.elapsed)))
            .with("images", n)
            .with_sanity(window_sanity)
            .to_string(),
        window_sanity.to_string(),
        delta.to_string(),
    ])
}

/// A scratch directory under the OS temp dir, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Result<Self> {
        let dir = std::env::temp_dir().join(format!("loractl-{tag}-{}", std::process::id()));
        // A stale directory from a killed run with a recycled pid would mix in
        // old images; start empty.
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Resolve the inputs and run: a real directory, or the synthetic fallback.
fn run(args: &Args) -> Result<Vec<String>> {
    let bucket = Bucket {
        width: args.bucket.0,
        height: args.bucket.1,
    };
    let window = default_window();
    match &args.dir {
        Some(dir) => {
            let mut paths = list_images(dir)?;
            ensure!(
                !paths.is_empty(),
                "no .png/.jpg/.jpeg images in {} (omit the directory for the synthetic fallback)",
                dir.display()
            );
            paths.truncate(args.images.unwrap_or(DEFAULT_IMAGES));
            probe(&paths, bucket, "real", window)
        }
        None => {
            let scratch = Scratch::new("decode-probe")?;
            let count = args.images.unwrap_or(DEFAULT_IMAGES);
            let paths = write_synthetic(&scratch.0, count, args.synthetic_size)?;
            probe(&paths, bucket, "synthetic", window)
        }
    }
}

fn main() -> Result<()> {
    let args = parse_args_from(std::env::args().skip(1))?;
    for line in run(&args)? {
        println!("{line}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(args: &[&str]) -> impl Iterator<Item = String> {
        args.iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn no_arguments_means_the_synthetic_fallback_at_defaults() {
        let args = parse_args_from(argv(&[])).expect("bare invocation parses");
        assert_eq!(
            args,
            Args {
                dir: None,
                images: None,
                bucket: (512, 512),
                synthetic_size: (1920, 1280),
            }
        );
    }

    #[test]
    fn every_flag_reaches_the_args() {
        let args = parse_args_from(argv(&[
            "some/dir",
            "--images",
            "40",
            "--bucket",
            "688x384",
            "--synthetic-size",
            "64X48",
        ]))
        .expect("flags parse");
        assert_eq!(args.dir, Some(PathBuf::from("some/dir")));
        assert_eq!(args.images, Some(40));
        assert_eq!(args.bucket, (688, 384));
        assert_eq!(args.synthetic_size, (64, 48));
    }

    #[test]
    fn bad_flags_fail_loudly() {
        for bad in [
            &["--images", "0"][..],
            &["--images", "many"],
            &["--images"],
            &["--bucket", "512"],
            &["--bucket", "0x512"],
            &["--bucket", "500x512"], // not a multiple of BUCKET_ALIGN
            &["--bogus"],
            &["a", "b"], // two directories
        ] {
            assert!(
                parse_args_from(argv(bad)).is_err(),
                "{bad:?} must be rejected"
            );
        }
    }

    #[test]
    fn the_synthetic_fallback_writes_decodable_images_in_three_aspects() {
        let scratch = Scratch::new("decode-probe-test-synth").unwrap();
        let paths = write_synthetic(&scratch.0, 4, (64, 48)).unwrap();
        assert_eq!(paths.len(), 4);
        // Filename order is generation order, so a prefix is stable.
        let mut sorted = paths.clone();
        sorted.sort();
        assert_eq!(paths, sorted);
        let dims: Vec<_> = paths
            .iter()
            .map(|p| image::image_dimensions(p).expect("decodable header"))
            .collect();
        assert_eq!(dims, [(64, 48), (48, 64), (64, 36), (64, 48)]);
        // And `list_images` sees exactly them.
        assert_eq!(list_images(&scratch.0).unwrap(), paths);
    }

    #[test]
    fn synthetic_images_are_deterministic_and_not_flat() {
        let a = synthetic_image(32, 32, 7);
        assert_eq!(a, synthetic_image(32, 32, 7));
        assert_ne!(a, synthetic_image(32, 32, 8));
        let distinct: std::collections::HashSet<_> = a.pixels().map(|p| p.0).collect();
        assert!(distinct.len() > 64, "noise must defeat entropy shortcuts");
    }

    #[test]
    fn list_images_filters_by_extension_and_sorts() {
        let scratch = Scratch::new("decode-probe-test-list").unwrap();
        for name in ["b.PNG", "a.jpg", "c.txt", "d.jpeg"] {
            std::fs::write(scratch.0.join(name), b"x").unwrap();
        }
        let names: Vec<_> = list_images(&scratch.0)
            .unwrap()
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["a.jpg", "b.PNG", "d.jpeg"]);
    }

    /// Drift guard for the mirrored window: `decode_window()` is private, so
    /// read its source and assert the body is exactly what `default_window`
    /// reproduces. A comment naming the function is not a check; this is.
    #[test]
    fn the_window_mirrors_dataset_rs_decode_window() {
        let src = include_str!("../src/dataset.rs");
        let body = format!(
            "fn decode_window() -> usize {{\n    rayon::current_num_threads().clamp(1, {MAX_WINDOW})\n}}"
        );
        assert!(
            src.contains(&body),
            "dataset.rs::decode_window() changed; update default_window()/MAX_WINDOW \
             in decode_probe.rs to match"
        );
        assert!((1..=MAX_WINDOW).contains(&default_window()));
    }

    /// The windowed arm must hand back input order at every window size,
    /// including the degenerate 1 and one larger than the list — and agree
    /// with the serial arm byte for byte.
    #[test]
    fn the_window_size_changes_nothing_observable() {
        let scratch = Scratch::new("decode-probe-test-window").unwrap();
        let paths = write_synthetic(&scratch.0, 5, (80, 60)).unwrap();
        let bucket = Bucket {
            width: 32,
            height: 48,
        };
        let serial = run_serial(&paths, bucket).unwrap();
        assert_eq!(serial.digests.len(), 5);
        // Images differ from one another, or "identical" would be vacuous.
        let unique: std::collections::HashSet<_> = serial.digests.iter().collect();
        assert_eq!(unique.len(), 5);
        for window in [1, 2, 3, 99] {
            let windowed = run_windowed(&paths, bucket, window).unwrap();
            check_identical(&serial.digests, &windowed.digests, &paths)
                .unwrap_or_else(|e| panic!("window {window}: {e}"));
        }
    }

    /// Teeth for the identity gate: a perturbed digest, or a dropped image,
    /// must fail rather than be reported as a speedup.
    #[test]
    fn the_identity_gate_rejects_a_different_output() {
        let paths = vec![PathBuf::from("a.jpg"), PathBuf::from("b.jpg")];
        assert!(check_identical(&[1, 2], &[1, 2], &paths).is_ok());
        let err = check_identical(&[1, 2], &[1, 3], &paths).unwrap_err();
        assert!(err.to_string().contains("b.jpg"), "{err}");
        assert!(check_identical(&[1, 2], &[1], &paths).is_err());
    }

    #[test]
    fn vet_rejects_dead_and_misshapen_buffers() {
        let bucket = Bucket {
            width: 16,
            height: 16,
        };
        let path = Path::new("x.jpg");
        let good = vec![0.5f32; 3 * 16 * 16];
        assert!(vet(&good, bucket, path).is_ok());
        assert!(vet(&vec![0.0; 3 * 16 * 16], bucket, path).is_err(), "dead");
        assert!(vet(&good[..10], bucket, path).is_err(), "wrong length");
        let mut nan = good;
        nan[3] = f32::NAN;
        assert!(vet(&nan, bucket, path).is_err(), "non-finite");
    }

    #[test]
    fn a_missing_or_empty_directory_is_an_error_not_a_silent_fallback() {
        let scratch = Scratch::new("decode-probe-test-empty").unwrap();
        let mut args = parse_args_from(argv(&[])).unwrap();
        args.dir = Some(scratch.0.clone());
        assert!(run(&args).unwrap_err().to_string().contains("no .png"));
        args.dir = Some(scratch.0.join("absent"));
        assert!(run(&args).is_err());
    }

    /// End to end on a tiny synthetic set: the printed lines carry the schema
    /// the doc promises, and the delta is the arithmetic difference of the two
    /// arms' `ms=`.
    #[test]
    fn the_probe_prints_the_documented_lines() {
        let args = Args {
            dir: None,
            images: Some(3),
            bucket: (32, 32),
            synthetic_size: (96, 64),
        };
        let lines = run(&args).expect("synthetic run succeeds");
        let by_prefix = |p: &str| {
            lines
                .iter()
                .find(|l| l.starts_with(p))
                .unwrap_or_else(|| panic!("no {p:?} line in {lines:#?}"))
        };
        let model = by_prefix("MODEL label=decode_probe ");
        for term in [
            "images=3",
            "source=synthetic",
            "bucket=32x32",
            "identical=1",
        ] {
            assert!(model.contains(term), "{model}");
        }
        assert!(model.contains("excludes="), "{model}");
        let field = |line: &str, key: &str| -> f64 {
            line.split_whitespace()
                .find_map(|t| t.strip_prefix(key))
                .unwrap_or_else(|| panic!("no {key} in {line}"))
                .parse()
                .unwrap()
        };
        let serial = by_prefix("RESULT label=decode_serial ");
        let window = by_prefix("RESULT label=decode_window ");
        let delta = by_prefix("RESULT label=decode_delta ");
        assert!(serial.contains("images=3") && serial.contains("x2_ratio="));
        assert!(window.contains("images=3") && window.contains("x2_ratio="));
        // The lines print four decimals, so allow that rounding.
        let diff = field(serial, "ms=") - field(window, "ms=");
        assert!(
            (field(delta, "ms=") - diff).abs() < 2e-4,
            "{delta} vs {diff}"
        );
        assert!(delta.contains("saved_total_ms=") && delta.contains("speedup_x="));
        assert_eq!(lines.iter().filter(|l| l.starts_with("SANITY ")).count(), 2);
    }
}
