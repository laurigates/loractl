//! CPU-backend matmul probe: does `burn-flex` match `burn-ndarray`'s threaded
//! GEMM on the shape our dataset-encode phase is actually bound by?
//!
//! ## Why this exists
//!
//! `loractl-core` carries a direct `burn-ndarray` dependency purely to re-enable
//! that crate's default `multi-threads` feature, which `default-features =
//! false` on the `burn` umbrella silently drops (see the comment block in
//! `Cargo.toml`, and upstream [tracel-ai/burn#5332]). That is worth ~8.7x on the
//! CPU encode phase, so it is load-bearing — and burn's maintainers have since
//! said on that issue that **`burn-ndarray` is slated for deprecation, with
//! `burn-flex` the surviving CPU backend**, and asked whether flex matches the
//! performance we are getting from ndarray.
//!
//! This probe answers that question with the repo's own measurement primitives
//! rather than a stopwatch: the `RESULT`/`SANITY`/`MODEL` line schema, the
//! wall-sync timer, and the dead-graph guards (`loractl-bench`, #110).
//!
//! ## The shape
//!
//! `[1, seq, hidden] @ [1, hidden, hidden]`, defaulting to the
//! **seq 512 / hidden 2560** shape the numbers on burn#5332 were measured at
//! (2·512·2560·2560 = 6.71 GFLOP per matmul: 94.2 ms single-threaded vs 32.0 ms
//! threaded, on a 24-core x86_64 host). Keeping the default identical is the
//! point — it makes any number this probe prints directly comparable to the
//! ones already on the issue.
//!
//! ## The arms, and why they are compile-time
//!
//! Threading and SIMD are Cargo features, not runtime switches, so the arms are
//! separate builds. `burn-flex`'s own defaults are `["std", "simd", "rayon"]`,
//! and the umbrella's `default-features = false` drops all three. `simd` comes
//! back through burn's `simd` passthrough **only if `flex` is also enabled** —
//! `burn-flex?/simd` is the weak form, so it keys off burn's own optional
//! `burn-flex` dep rather than off the crate being in the graph — and `rayon`
//! has no passthrough at any feature combination. So this crate declares
//! `burn-flex` directly and re-enables them by name, as it already does for
//! burn-ndarray:
//!
//! | build | backend | threading | SIMD |
//! |---|---|---|---|
//! | `--features flex` | Flex | off | off |
//! | `--features flex-rayon` | Flex | rayon | off |
//! | `--features flex-simd` | Flex | off | macerator/gemm |
//! | `--features flex-rayon,flex-simd` | Flex | rayon | macerator/gemm |
//! | (any of the above) | NdArray | always on¹ | off² |
//!
//! ¹ the lib's direct `burn-ndarray` dep pins `multi-threads` unconditionally;
//! cap it at runtime with `RAYON_NUM_THREADS=1 MATMUL_NUM_THREADS=1` for a
//! single-threaded reference arm. ² deliberately: ndarray's `simd` breaks
//! `tests/grad_checkpointing.rs`'s bit-identical replay assertion (Cargo.toml).
//!
//! `just flex-probe` runs the whole matrix and prints it as a table.
//!
//! ## What it checks besides speed
//!
//! A faster backend that computes something else is not a faster backend. When
//! both arms are compiled in, the probe compares flex's output against
//! ndarray's element-wise and prints a `PARITY` line; every arm is also run
//! through the `plausible()` dead-graph guard and the 2×-iters `SANITY` ratio,
//! so an elided or NaN-poisoned matmul is reported as such instead of as a very
//! fast one.
//!
//! On a small or contended host a `sanity=SUSPECT` verdict is often the harness
//! being right about the *host* rather than about the backend: the 2×-iters pass
//! runs with no warmup of its own, so a thread pool still spinning up during the
//! first pass makes the second look superlinearly cheap (a ratio below 2.0
//! rather than above). Raise `--iters` and re-run. If it stays SUSPECT, the
//! timing is not quotable — that is the whole point of the line.
//!
//! ## What it deliberately does NOT answer
//!
//! Whether we can *migrate* to flex. That needs two things this probe does not
//! do: the real encode phase on real Qwen weights (this is one matmul), and a
//! re-run of `tests/grad_checkpointing.rs` against a flex-backed trainer — the
//! bit-identity assertion that ndarray's SIMD already fails. Treat a good number
//! here as a green light to try those, not as a migration verdict.
//!
//! Usage:
//!   cargo run --release -p loractl-core --features flex-rayon,flex-simd \
//!     --example cpu_backend_probe -- [--seq 512] [--hidden 2560] [--iters 10]
//!
//! [tracel-ai/burn#5332]: https://github.com/tracel-ai/burn/issues/5332

use anyhow::{Result, bail};
use burn::backend::NdArray;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use loractl_bench::{ModelLine, plausible};
use loractl_core::bench::BurnOpBench;

/// Deterministic pseudo-random values in [-1, 1] — identical on every backend
/// (no backend RNG involved), so every arm multiplies the identical matrices
/// and the `PARITY` comparison is a statement about the backends alone.
fn det_vals(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// One timed arm: the per-iteration `RESULT` line plus the materialized output
/// (for the cross-backend `PARITY` check and the dead-graph guard).
struct Arm {
    ms: f64,
    sanity: &'static str,
    x2_ratio: f64,
    out: Vec<f32>,
}

impl Arm {
    /// Say so, loudly, when the 2×-iters ratio lands outside `[SANITY_LOW,
    /// SANITY_HIGH]`. A `RESULT` line already carries `sanity=SUSPECT`, but the
    /// repo's rule is that such a timing must not be quoted at all — worth more
    /// than one token in a wide line nobody reads to the end of.
    fn warn_if_suspect(&self, label: &str) {
        if self.sanity != "ok" {
            println!(
                "WARNING {label} 2x-iters ratio {:.3} is out of band — work was elided or \
                 per-iteration cost is unstable; do NOT quote this timing",
                self.x2_ratio
            );
        }
    }
}

/// Time `[1, seq, hidden] @ [1, hidden, hidden]` on backend `B`.
///
/// The timed closure submits the matmul and nothing else — no readback, which
/// would fence inside the timed region. The output is materialized once,
/// afterwards, outside the measurement.
fn run_arm<B: Backend>(label: &str, seq: usize, hidden: usize, bench: BurnOpBench) -> Result<Arm> {
    let device = B::Device::default();
    let a = Tensor::<B, 3>::from_data(
        TensorData::new(det_vals(seq * hidden, 0x5eed_1234), [1, seq, hidden]),
        &device,
    );
    let b = Tensor::<B, 3>::from_data(
        TensorData::new(det_vals(hidden * hidden, 0x0b16_b00b), [1, hidden, hidden]),
        &device,
    );

    let result = bench.run::<B>(label, &device, || {
        let out = a.clone().matmul(b.clone());
        std::hint::black_box(&out);
    })?;

    let out = a
        .matmul(b)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("reading back {label}: {e:?}"))?;

    // 2·M·N·K for a dense matmul; the MODEL line below carries the terms.
    let flop = 2.0 * seq as f64 * hidden as f64 * hidden as f64;
    let (sanity, x2_ratio) = match result.sanity {
        Some(s) => (s.verdict(), s.ratio),
        None => ("none", f64::NAN),
    };
    let gflops = flop / (result.ms / 1e3) / 1e9;

    println!("{result} gflops={gflops:.2}");
    Ok(Arm {
        ms: result.ms,
        sanity,
        x2_ratio,
        out,
    })
}

/// Element-wise agreement between two arms' outputs: `(max_abs, ref_absmax,
/// rel_to_scale)`.
///
/// Reported, not asserted: the useful failure here is a *number*, since a
/// backend that disagrees in the 7th digit (a different reduction order) and one
/// that disagrees in the 1st (a broken kernel) need different responses.
///
/// The error is normalized against the reference's **magnitude scale**, not
/// per-element. Per-element relative error is the tempting metric and it is
/// useless here: the product of two zero-mean random matrices has entries
/// arbitrarily close to zero, so `|Δ| / |ref|` is dominated by whichever element
/// happened to land nearest zero and reports a huge number for outputs that
/// agree to a part in a million. (Measured: a run agreeing to `max_abs = 4.2e-5`
/// on entries of magnitude ~30 reported `max_rel = 2.0e-1` under that metric.)
#[cfg(feature = "flex")]
fn parity(reference: &[f32], other: &[f32]) -> (f64, f64, f64) {
    let mut max_abs = 0.0_f64;
    let mut ref_absmax = 0.0_f64;
    for (r, o) in reference.iter().zip(other) {
        max_abs = max_abs.max((*r as f64 - *o as f64).abs());
        ref_absmax = ref_absmax.max((*r as f64).abs());
    }
    let rel_to_scale = if ref_absmax > 0.0 {
        max_abs / ref_absmax
    } else {
        f64::NAN
    };
    (max_abs, ref_absmax, rel_to_scale)
}

/// `--flag value` parsing, matching the other probes in this directory.
fn arg_usize(args: &[String], flag: &str, default: usize) -> Result<usize> {
    match args.iter().position(|a| a == flag) {
        Some(i) => match args.get(i + 1) {
            Some(v) => Ok(v.parse()?),
            None => bail!("{flag} needs a value"),
        },
        None => Ok(default),
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seq = arg_usize(&args, "--seq", 512)?;
    let hidden = arg_usize(&args, "--hidden", 2560)?;
    let iters = arg_usize(&args, "--iters", 10)? as u32;
    let warmup = arg_usize(&args, "--warmup", 3)? as u32;
    let bench = BurnOpBench { iters, warmup };

    let threads = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "unknown".into());
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| "unset".into());
    // The two knobs the threaded arms respond to: burn-ndarray's `multi-threads`
    // pulls both `ndarray/rayon` and `matrixmultiply/threading`, and those read
    // different variables. Echo them so a reported number always carries the
    // thread configuration it was produced under.
    println!(
        "CONFIG seq={seq} hidden={hidden} iters={iters} warmup={warmup} \
         available_parallelism={threads} RAYON_NUM_THREADS={} MATMUL_NUM_THREADS={} \
         flex={} flex_rayon={} flex_simd={}",
        env("RAYON_NUM_THREADS"),
        env("MATMUL_NUM_THREADS"),
        cfg!(feature = "flex"),
        cfg!(feature = "flex-rayon"),
        cfg!(feature = "flex-simd"),
    );
    println!(
        "{}",
        ModelLine::new("cpu_matmul")
            .with("formula", "2*seq*hidden*hidden")
            .with("batch", 1)
            .with("seq", seq)
            .with("hidden", hidden)
            .with(
                "gflop_per_matmul",
                format!(
                    "{:.4}",
                    2.0 * seq as f64 * hidden as f64 * hidden as f64 / 1e9
                )
            )
            .with("dtype", "f32")
            .with(
                "excludes",
                "tensor-construction,host-readback,dataset-encode-overheads",
            )
    );

    let ndarray = run_arm::<NdArray<f32>>("ndarray_f32", seq, hidden, bench)?;
    if !plausible(&ndarray.out) {
        bail!("ndarray output failed the dead-graph guard (empty, non-finite, or all-zero)");
    }
    ndarray.warn_if_suspect("ndarray_f32");

    #[cfg(feature = "flex")]
    {
        let flex = run_arm::<burn::backend::Flex>("flex_f32", seq, hidden, bench)?;
        if !plausible(&flex.out) {
            bail!("flex output failed the dead-graph guard (empty, non-finite, or all-zero)");
        }
        flex.warn_if_suspect("flex_f32");

        let (max_abs, ref_absmax, rel_to_scale) = parity(&ndarray.out, &flex.out);
        println!(
            "PARITY reference=ndarray_f32 other=flex_f32 max_abs={max_abs:.3e} \
             ref_absmax={ref_absmax:.3e} rel_to_scale={rel_to_scale:.3e}"
        );
        println!(
            "SPEEDUP flex_over_ndarray={:.3}x (ndarray {:.2} ms, flex {:.2} ms)",
            ndarray.ms / flex.ms,
            ndarray.ms,
            flex.ms
        );
    }
    #[cfg(not(feature = "flex"))]
    println!(
        "NOTE ndarray arm only ({:.2} ms) — rebuild with \
         --features flex[,flex-rayon][,flex-simd] for the comparison",
        ndarray.ms
    );

    Ok(())
}
