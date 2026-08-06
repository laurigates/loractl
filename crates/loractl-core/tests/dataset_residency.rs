//! #175: the dataset pipeline's memory footprint is O(batch), not O(dataset).
//!
//! `PreparedDataset` used to hold every example's latent + conditioning as
//! live device tensors, and the trainer additionally materialized every
//! concatenated batch up front — so peak VRAM scaled with **dataset size**.
//! Krea 2 conditioning is `[1, 512, 12, 2560]` f32 = 60 MiB per example, held
//! for the whole run, against the ~4 GB of headroom ADR-0005 Addendum 3
//! measured at 512px int4. Fifty examples was 3 GiB; a thousand was 60.
//!
//! **Why an allocator counter is a faithful proxy.** On the `NdArray`
//! backend, device memory *is* heap memory: `Tensor::from_data` allocates a
//! `Vec<f32>` on the system allocator and nothing else. A counting
//! `#[global_allocator]` therefore measures exactly the quantity a GPU would
//! be asked for, deterministically and without a GPU. The peak-VRAM number
//! itself still needs a `gh workflow run gpu.yml` dispatch against a ≥50-image
//! dataset — the 4-image `dataset-tiny` fixture is structurally too small to
//! show the bug — and nothing here claims one.
//!
//! **Why the measurement is differential.** Absolute byte counts include
//! per-`Vec` bookkeeping, `PathBuf`s, and the allocator's own rounding, none
//! of which the assertion should depend on. Every test below compares two
//! runs that differ *only* in conditioning size (S = 64 vs S = 1024, a 16×
//! payload difference) or in whether a batch is held, so the constant
//! bookkeeping cancels and only the payload can move the number.
//!
//! **`MEASURE_LOCK` is load-bearing, not decorative.** Cargo runs a test
//! binary's tests on parallel threads and the global allocator counts all of
//! them; without the lock a concurrent test's allocations land inside another's
//! measurement window and look exactly like a residency regression.

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use loractl_core::config::DatasetConfig;
use loractl_core::dataset::{PreparedDataset, plan_dataset, prepare_dataset};
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicIsize, Ordering};

type B = NdArray;

/// Live heap bytes, maintained by [`CountingAlloc`].
static LIVE: AtomicIsize = AtomicIsize::new(0);

/// High-water mark of [`LIVE`], maintained by [`CountingAlloc::alloc`].
///
/// `LIVE` sampled from a callback only sees the boundaries between units of
/// work, which is blind to anything allocated **and freed** inside one — and
/// "decode an image, then drop the buffer unread" is exactly that shape (it
/// is the regression `LatentSource`'s doc says the design exists to prevent).
/// Every peak happens at an allocation, so tracking it there sees all of it.
static PEAK: AtomicIsize = AtomicIsize::new(0);

/// Serializes every measurement in this binary — see the module docs.
static MEASURE_LOCK: Mutex<()> = Mutex::new(());

/// A pass-through allocator that tracks the live byte total.
struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let now =
                LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed) + layout.size() as isize;
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            LIVE.fetch_add(
                new_size as isize - layout.size() as isize,
                Ordering::Relaxed,
            );
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

fn live() -> isize {
    LIVE.load(Ordering::Relaxed)
}

/// Arm the high-water mark at the current live total, returning that baseline
/// for [`peak_above`]. Only meaningful while [`MEASURE_LOCK`] is held — the
/// counter is global.
fn arm_peak() -> isize {
    let before = live();
    PEAK.store(before, Ordering::Relaxed);
    before
}

/// Bytes the high-water mark rose above `baseline`.
fn peak_above(baseline: isize) -> isize {
    PEAK.load(Ordering::Relaxed) - baseline
}

/// Slack allowed on every differential assertion: the plan's own bookkeeping
/// (paths, `Vec` headers) plus allocator rounding. Two orders of magnitude
/// below the payload differences under test.
const SLACK: isize = 16 * 1024;

/// `count` identical `side × side` images in a fresh temp dir.
///
/// The trailing counter matters for the same reason [`MEASURE_LOCK`] does:
/// cargo runs this binary's tests on parallel threads of one process, so the
/// pid separates nothing and two tags differing only by a nanosecond clock
/// read can collide into one directory.
fn temp_dataset_sized(tag: &str, count: usize, side: u32) -> PathBuf {
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let dir = std::env::temp_dir().join(format!(
        "loractl-{tag}-{}-{nanos}-{seq}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    for i in 0..count {
        image::RgbImage::from_fn(side, side, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 0])
        })
        .save(dir.join(format!("img{i}.png")))
        .unwrap();
        std::fs::write(dir.join(format!("img{i}.txt")), format!("caption {i}")).unwrap();
    }
    dir
}

/// Eight identical tiny images in a fresh temp dir.
fn temp_dataset(tag: &str) -> PathBuf {
    temp_dataset_sized(tag, EXAMPLES, 32)
}

const EXAMPLES: usize = 8;
/// Latent channels/side for the mock VAE: `[1, 4, 4, 4]`.
const LATENT: [usize; 4] = [1, 4, 4, 4];
/// Conditioning trailing dims: `[1, S, 4, 8]`.
const COND_N: usize = 4;
const COND_D: usize = 8;

fn config(dir: &Path) -> DatasetConfig {
    config_at(dir, 32)
}

fn config_at(dir: &Path, resolution: u32) -> DatasetConfig {
    DatasetConfig {
        path: dir.to_path_buf(),
        resolution,
        batch_size: 1,
        ..Default::default()
    }
}

/// The mock VAE: a fixed-size latent, so the cache payload under test is the
/// conditioning and nothing else varies with it.
fn mock_latent(_: Tensor<B, 4>) -> anyhow::Result<Tensor<B, 4>> {
    Ok(Tensor::from_data(
        TensorData::new(vec![0.5f32; LATENT.iter().product()], LATENT),
        &Default::default(),
    ))
}

/// The mock text encoder at sequence length `seq`.
fn mock_cond(seq: usize) -> anyhow::Result<(Tensor<B, 4>, Tensor<B, 2, Int>)> {
    let device = Default::default();
    let cond = Tensor::from_data(
        TensorData::new(
            vec![1.0f32; seq * COND_N * COND_D],
            [1, seq, COND_N, COND_D],
        ),
        &device,
    );
    let mask: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![1i64; seq], [1, seq]), &device);
    Ok((cond, mask))
}

/// Encode `dir` cold with conditioning of sequence length `seq`, then drop
/// everything the encode produced. Returns nothing: the point is the cache on
/// disk, exactly as the real encode phase works.
fn encode(dir: &Path, seq: usize) {
    let device: burn::tensor::Device<B> = Default::default();
    prepare_dataset::<B>(
        &config(dir),
        "residency-v1",
        &device,
        mock_latent,
        |_| mock_cond(seq),
        |_| {},
    )
    .expect("cold encode");
}

/// The f32 payload of one example's conditioning at sequence length `seq`.
fn cond_bytes(seq: usize) -> isize {
    (seq * COND_N * COND_D * 4) as isize
}

/// THE test: planning a dataset costs the same whether each example's
/// conditioning is 8 KiB or 128 KiB.
///
/// Teeth: the total payload difference across the eight examples is
/// `8 × (1024 − 64) × 4 × 8 × 4 B ≈ 960 KiB`, ~60× the 16 KiB slack. The
/// pre-#175 device-resident design fails this by two orders of magnitude —
/// and so would a "fix" that merely moved the tensors to host `Vec<f32>`,
/// which is why this assertion is about the *asymptote*, not the medium.
#[test]
fn plan_footprint_is_independent_of_conditioning_size() {
    let _guard = MEASURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let small_dir = temp_dataset("residency-small");
    let big_dir = temp_dataset("residency-big");
    encode(&small_dir, 64);
    encode(&big_dir, 1024);

    let before = live();
    let small = plan_dataset(&config(&small_dir), "residency-v1", |_| {}).expect("plan small");
    let cost_small = live() - before;

    let before = live();
    let big = plan_dataset(&config(&big_dir), "residency-v1", |_| {}).expect("plan big");
    let cost_big = live() - before;

    assert_eq!(small.items.len(), EXAMPLES);
    assert_eq!(big.items.len(), EXAMPLES);
    let payload_delta = EXAMPLES as isize * (cond_bytes(1024) - cond_bytes(64));
    assert!(
        cost_big <= cost_small + SLACK,
        "planning must not scale with conditioning size: {cost_small} B for S=64 vs \
         {cost_big} B for S=1024 (the payload difference is {payload_delta} B)"
    );
    assert!(
        cost_small < 4 * SLACK,
        "a plan over {EXAMPLES} examples should be a few KiB of paths, got {cost_small} B"
    );

    drop(small);
    drop(big);
    std::fs::remove_dir_all(&small_dir).ok();
    std::fs::remove_dir_all(&big_dir).ok();
}

/// The anti-vacuity kill test. Without it, a loader that returned empty
/// tensors would satisfy every assertion above and below.
#[test]
fn one_batch_costs_one_batch() {
    let _guard = MEASURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = temp_dataset("residency-one");
    encode(&dir, 1024);
    let prepared: PreparedDataset =
        plan_dataset(&config(&dir), "residency-v1", |_| {}).expect("plan");
    let plans = prepared.batches(2);
    let expected = plans[0].items.len() as isize * cond_bytes(1024);

    let before = live();
    {
        let batch = prepared
            .load_batch::<B>(&plans[0], &Default::default())
            .expect("load batch");
        let held = live() - before;
        assert!(
            held >= expected,
            "a held batch must actually contain its {expected} B of conditioning, saw {held} B"
        );
        // 3× rather than 1.5×: `Tensor::cat` builds the per-item tensors and
        // then the concatenated copy, and the latent + mask ride along. Still
        // a bounded multiple of ONE batch, which is the claim.
        assert!(
            held <= 3 * expected,
            "a held batch must not cost {held} B against an expected ~{expected} B"
        );
        std::hint::black_box(&batch);
    }
    let after = live() - before;
    assert!(
        after.abs() <= SLACK,
        "dropping the batch must return the memory, {after} B still live"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The direct statement of "peak no longer scales with example count":
/// walking every batch of an epoch accumulates nothing.
#[test]
fn iterating_every_batch_accumulates_nothing() {
    let _guard = MEASURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = temp_dataset("residency-epoch");
    encode(&dir, 1024);
    let prepared = plan_dataset(&config(&dir), "residency-v1", |_| {}).expect("plan");
    let plans = prepared.batches(1);
    assert_eq!(plans.len(), EXAMPLES, "batch_size 1 over one bucket");

    let before = live();
    for plan in &plans {
        let batch = prepared
            .load_batch::<B>(plan, &Default::default())
            .expect("load batch");
        std::hint::black_box(&batch);
    }
    let after = live() - before;
    assert!(
        after.abs() <= SLACK,
        "an epoch's worth of batches accumulated {after} B; peak must not scale with \
         example count (the payload is {} B per example)",
        cond_bytes(1024)
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// #178 — the decode pre-pass is bounded too, in HOST memory.
// ---------------------------------------------------------------------------

/// The side of the images the pre-pass tests decode, and the resolution they
/// are bucketed at — square, so the bucket is exactly `SIDE × SIDE` and one
/// decoded buffer is exactly [`DECODED`] bytes.
const SIDE: u32 = 128;
/// One decoded example: CHW f32, `3 · SIDE² · 4` bytes (192 KiB).
const DECODED: isize = 3 * (SIDE as isize) * (SIDE as isize) * 4;
/// Threads (and therefore `decode_window()`) for the pre-pass tests. Fixing
/// it makes the bound under test a constant rather than a property of
/// whatever machine ran the suite.
const WINDOW: usize = 2;

/// A pool of exactly [`WINDOW`] threads, already spawned.
///
/// The warm-up is not decoration: rayon spawns its workers lazily, and their
/// first-touch bookkeeping would otherwise land inside a measurement window
/// and read as residency.
fn fixed_pool() -> rayon::ThreadPool {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(WINDOW)
        .build()
        .unwrap();
    pool.broadcast(|_| {});
    pool
}

/// Run a cold-or-warm `prepare_dataset` over `dir` inside `pool` and return
/// its **peak** host allocation.
///
/// Peak, not a sample from the progress sink: the sink only observes the
/// boundaries between entries, so a buffer allocated and dropped *inside* one
/// unit of work is invisible to it — verified, not assumed. Sabotaging the
/// pre-pass's cache-hit arm with a discarded
/// `let _ = decode_image_for_bucket(…)` passes a sink-sampled version of
/// [`a_warm_encode_pass_allocates_no_decode_buffer`] and fails this one.
fn peak_during_prepare(pool: &rayon::ThreadPool, dir: &Path, seq: usize) -> isize {
    let baseline = arm_peak();
    pool.install(|| {
        let device: burn::tensor::Device<B> = Default::default();
        prepare_dataset::<B>(
            &config_at(dir, SIDE),
            "prepass-v1",
            &device,
            mock_latent,
            |_| mock_cond(seq),
            |_| {},
        )
        .expect("prepare");
    });
    peak_above(baseline)
}

/// #178's own bound, which was otherwise only a comment.
///
/// `decode_window()` caps how many decoded buffers the parallel pre-pass
/// holds at once, and its doc names the regression it prevents: a
/// whole-dataset pre-pass is the #175 residency bug reintroduced in *host*
/// memory. Nothing measured it — the other tests in this file all sample
/// `plan_dataset`/`load_batch`, and the pre-pass lives in `prepare_dataset`,
/// outside every one of their measurement windows. Kill test: replacing
/// `decode_window()` with `total.max(1)` — the exact regression — fails this
/// and no other test in the suite.
#[test]
fn the_cold_encode_pre_pass_is_bounded_by_the_decode_window() {
    let _guard = MEASURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let pool = fixed_pool();

    let small_dir = temp_dataset_sized("prepass-small", 8, SIDE);
    let big_dir = temp_dataset_sized("prepass-big", 40, SIDE);
    let small = peak_during_prepare(&pool, &small_dir, 8);
    let big = peak_during_prepare(&pool, &big_dir, 8);
    std::fs::remove_dir_all(&small_dir).ok();
    std::fs::remove_dir_all(&big_dir).ok();

    // Anti-vacuity: the pre-pass really did hold decoded buffers. Without
    // this a `prepare_dataset` that decoded nothing would satisfy everything
    // below.
    assert!(
        small >= DECODED,
        "the pre-pass should hold at least one {DECODED} B decoded buffer, saw {small} B"
    );
    // O(window): a small multiple of the window, not of the dataset. The
    // multiple is ~3 rather than 1 because an in-flight decode holds more
    // than its output buffer — the decoded RGB8 source and the full-frame
    // Lanczos3 result as well (see `decode_window`'s doc, which says so).
    // Measured at 0.58 MiB here; an unbounded pre-pass is 1.6 MiB at eight
    // examples and ~8 MiB at forty.
    let bound = 2 * (WINDOW as isize + 1) * DECODED;
    for (label, peak) in [("8 examples", small), ("40 examples", big)] {
        assert!(
            peak <= bound,
            "{label}: peak host residency {peak} B exceeds {bound} B \
             ({WINDOW} windowed buffers of {DECODED} B, plus slack)"
        );
    }
    // …and it is flat in the example count, which is the claim itself.
    assert!(
        (big - small).abs() <= DECODED / 2,
        "peak moved {} B between 8 and 40 examples; the pre-pass must be O(window), \
         not O(dataset)",
        big - small
    );
}

/// The work-based half of "a warm cache never decodes the image".
///
/// `dataset_pipeline.rs::a_warm_cache_never_decodes_the_image` gets its teeth
/// from a truncated PNG making `image::open` *error*, so it only sees a
/// surplus decode whose error propagates. A decode whose `Result` is dropped
/// unread is invisible to it — and that is precisely the shape `LatentSource`
/// exists to prevent (its doc says so). This measures the work instead: a
/// warm pass that decodes nothing cannot allocate a `3 · w · h · 4` buffer,
/// discarded or not.
#[test]
fn a_warm_encode_pass_allocates_no_decode_buffer() {
    let _guard = MEASURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let pool = fixed_pool();
    let dir = temp_dataset_sized("prepass-warm", 8, SIDE);

    let cold = peak_during_prepare(&pool, &dir, 8);
    assert!(
        cold >= DECODED,
        "the cold pass must decode: expected ≥ {DECODED} B live, saw {cold} B"
    );

    let warm = peak_during_prepare(&pool, &dir, 8);
    std::fs::remove_dir_all(&dir).ok();
    // Not even one decoded buffer's worth. Measured at ~78 KiB (header reads
    // and the plan's own paths); a pre-pass that decoded and discarded is
    // ~583 KiB, i.e. the window's buffers plus their sources.
    assert!(
        warm < DECODED,
        "a warm pass allocated {warm} B — as much as a whole {DECODED} B decoded \
         buffer, so it is decoding images it will not use"
    );
}
