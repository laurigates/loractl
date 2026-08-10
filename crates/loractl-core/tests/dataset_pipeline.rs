//! The M12 (#23) dataset pipeline, end to end and offline: bucket math,
//! kohya-style folder scanning, image loading, one-time encoding, cache
//! reuse, and per-bucket batching.
//!
//! The encoders are injected mocks (deterministic functions of their input),
//! which is the module's design point: the pipeline is files → buckets →
//! cache → batches, and whether the encoder is the real frozen `QwenVae` /
//! `Qwen3VlConditioner` (M14) or a test double changes nothing about its
//! contract. The cache-reuse test passes encoders that PANIC — a second
//! `prepare_dataset` over a warm cache must never invoke them.
//!
//! The #147 (`no_upscale`) / #148 (`bucketing: grid`) knobs are covered here
//! too. Both are opt-in, so the *default* path's tests are the regression
//! evidence and are deliberately left untouched — most sharply
//! [`prepare_encodes_once_reuses_cache_and_batches_per_bucket`]'s
//! `[1, 3, 8, 8]` assertion for the 32×32 `b.png`, which only holds while the
//! default still upscales it into the 64×64 bucket.
//!
//! Since #175 the pipeline is split into a **plan** (`plan_dataset` /
//! `prepare_dataset` → paths + shapes) and a **load** (`load_item` /
//! `load_batch` → tensors), so every value assertion below reads through the
//! real cache. Two guarantees gained teeth from that split and are tested
//! here: [`warm_planning_never_needs_an_encoder`] (the training path takes no
//! encoder arguments at all, so a warm epoch cannot re-encode — the signature
//! is the proof, not a closure that happens not to fire), and the bail-on-miss
//! contract now firing at *step* granularity too
//! ([`a_cache_file_deleted_after_planning_is_a_loud_error`],
//! [`a_cache_file_reshaped_after_planning_is_a_loud_error`]). The memory
//! claim itself lives in `tests/dataset_residency.rs`.
//!
//! #178 then made the host-side decode parallel. Its tests are at the bottom
//! of this file and are about what must **not** have changed: the decoded
//! buffer is bit-identical to the pre-#178 loader (kept verbatim as
//! [`decode_serial_reference`], because cache keys are name/bucket/fingerprint
//! and never content — a shifted value would invalidate nothing on disk), the
//! encoders still see scan order, a warm cache still decodes nothing, and
//! nothing observable depends on the machine's core count.

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use loractl_core::config::{BucketMode, DatasetConfig};
use loractl_core::dataset::{
    BUCKET_ALIGN, Bucket, PreparedDataset, PreparedItem, assign_bucket, bucket_set, fit_bucket,
    generate_buckets, generate_grid_buckets, plan_dataset, prepare_dataset,
};
use std::cell::Cell;
use std::path::PathBuf;

/// `Bucket` has no literal shorthand; this keeps the golden tables readable.
fn b(width: u32, height: u32) -> Bucket {
    Bucket { width, height }
}

/// The fraction of a `width × height` source discarded by cover-cropping it
/// into its nearest bucket — `1 − min(r, r_b)/max(r, r_b)`, which is exactly
/// what log-aspect nearest-neighbour minimizes.
fn crop_loss(buckets: &[Bucket], width: f64, height: f64) -> f64 {
    let chosen = buckets[assign_bucket(buckets, width as u32, height as u32)];
    let r = width / height;
    let rb = chosen.width as f64 / chosen.height as f64;
    1.0 - r.min(rb) / r.max(rb)
}

type B = NdArray;

const RESOLUTION: u32 = 64;

#[test]
fn unaligned_resolution_is_an_error_not_a_panic() {
    // resolution arrives straight from user YAML — misconfiguration must
    // surface as an Err through the Result API.
    assert!(generate_buckets(1000).is_err(), "1000 % 16 != 0 must error");
}

#[test]
fn buckets_are_aligned_unique_and_include_square() {
    let buckets = generate_buckets(RESOLUTION).expect("aligned resolution");
    assert!(!buckets.is_empty());
    for b in &buckets {
        assert_eq!(b.width % BUCKET_ALIGN, 0, "{b:?} width unaligned");
        assert_eq!(b.height % BUCKET_ALIGN, 0, "{b:?} height unaligned");
    }
    // Deduplicated…
    for (i, a) in buckets.iter().enumerate() {
        for b in &buckets[i + 1..] {
            assert_ne!(a, b, "duplicate bucket");
        }
    }
    // …and the square target bucket is present.
    assert!(
        buckets.contains(&Bucket {
            width: RESOLUTION,
            height: RESOLUTION
        }),
        "square bucket missing from {buckets:?}"
    );
}

#[test]
fn nearest_aspect_assignment_picks_matching_bucket() {
    let buckets = generate_buckets(RESOLUTION).expect("aligned resolution");
    // A square image lands in the square bucket.
    let square = assign_bucket(&buckets, 500, 500);
    assert_eq!(
        buckets[square],
        Bucket {
            width: RESOLUTION,
            height: RESOLUTION
        }
    );
    // A wide image lands in a wide bucket, a tall image in a tall one.
    let wide = buckets[assign_bucket(&buckets, 1600, 900)];
    assert!(
        wide.width > wide.height,
        "expected wide bucket, got {wide:?}"
    );
    let tall = buckets[assign_bucket(&buckets, 900, 1600)];
    assert!(
        tall.height > tall.width,
        "expected tall bucket, got {tall:?}"
    );
}

/// A unique per-test temp dir (same convention as `checkpoint_roundtrip.rs`).
///
/// The trailing counter is not decoration: cargo runs this binary's tests on
/// parallel threads of ONE process, so the pid disambiguates nothing between
/// them and two tests sharing a `tag` were separated only by a nanosecond
/// clock read. A collision would have one test's images land in another's
/// directory and fail with assertions pointing nowhere near the cause.
fn temp_dataset_dir(tag: &str) -> PathBuf {
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let dir = std::env::temp_dir().join(format!(
        "loractl-{tag}-{}-{nanos}-{seq}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Write a deterministic gradient PNG.
fn write_png(dir: &std::path::Path, name: &str, w: u32, h: u32) {
    let img = image::RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    });
    img.save(dir.join(name)).expect("write test png");
}

/// The mock image encoder: 8× average pooling (a deterministic stand-in for
/// the f8 VAE — latent channels = 3 here, which the pipeline must not care
/// about).
fn mock_encode_image(x: Tensor<B, 4>) -> anyhow::Result<Tensor<B, 4>> {
    Ok(burn::tensor::module::avg_pool2d(
        x,
        [8, 8],
        [8, 8],
        [0, 0],
        true,
        false,
    ))
}

/// The mock caption encoder: a `[1, 4, 2, 8]` stack filled with the caption
/// length (so different captions produce different tensors) and a NON-trivial
/// mask (last position 0) — the mask round-trips the cache through an
/// f32-store → `.int()`-reload conversion, and an all-ones mask couldn't tell
/// a corrupted round-trip from a correct one.
fn mock_encode_caption(caption: &str) -> anyhow::Result<(Tensor<B, 4>, Tensor<B, 2, Int>)> {
    let device = Default::default();
    let fill = caption.len() as f32;
    let cond = Tensor::from_data(
        TensorData::new(vec![fill; 4 * 2 * 8], [1, 4, 2, 8]),
        &device,
    );
    let mask = Tensor::from_data(TensorData::new(vec![1i64, 1, 1, 0], [1, 4]), &device);
    Ok((cond, mask))
}

fn flat_mask(t: &Tensor<B, 2, Int>) -> Vec<i64> {
    t.clone()
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .unwrap()
}

fn flat(t: &Tensor<B, 4>) -> Vec<f32> {
    t.clone()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap()
}

/// Materialize one planned item (#175).
///
/// `PreparedDataset` holds paths and shapes, not tensors, so every value
/// assertion below goes through the real read path — which also exercises the
/// planned-vs-found shape check on the way.
fn item(prepared: &PreparedDataset, index: usize) -> PreparedItem<B> {
    prepared
        .load_item::<B>(index, &Default::default())
        .expect("load item")
}

/// `expect_err` for results whose `Ok` type is not `Debug` — loaded tensors
/// deliberately are not (a failed assertion would dump megabytes).
fn err_of<T>(result: anyhow::Result<T>, must_fail: &str) -> String {
    match result {
        Ok(_) => panic!("{must_fail}"),
        Err(e) => format!("{e:#}"),
    }
}

#[test]
fn prepare_encodes_once_reuses_cache_and_batches_per_bucket() {
    let dir = temp_dataset_dir("dataset");
    // Two square-ish images (one needing upscale), one 4:3 PNG, and one
    // square JPEG with an UPPERCASE extension (covering the jpeg decode
    // feature and the case-insensitive extension match); captions for two,
    // the others caption-less (unconditional examples).
    write_png(&dir, "a.png", 64, 64);
    write_png(&dir, "b.png", 32, 32);
    write_png(&dir, "c.png", 100, 75);
    let jpg = image::RgbImage::from_fn(64, 64, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, 0])
    });
    jpg.save_with_format(dir.join("d.JPG"), image::ImageFormat::Jpeg)
        .expect("write test jpeg");
    std::fs::write(dir.join("a.txt"), "a red fox\n").unwrap();
    std::fs::write(dir.join("c.txt"), "green field").unwrap();

    let config = DatasetConfig {
        path: dir.clone(),
        resolution: RESOLUTION,
        batch_size: 1,
        no_upscale: false,
        bucketing: BucketMode::Aspects,
        min_bucket_resolution: None,
    };
    let device = Default::default();

    // --- Cold pass: every encoder runs exactly once per example. ---
    let img_calls = Cell::new(0usize);
    let cap_calls = Cell::new(0usize);
    let captions_seen = std::cell::RefCell::new(Vec::<String>::new());
    type Report = (usize, usize, String, bool);
    let progress = std::cell::RefCell::new(Vec::<Report>::new());
    let prepared = prepare_dataset::<B>(
        &config,
        "mock-v1",
        &device,
        |x| {
            img_calls.set(img_calls.get() + 1);
            mock_encode_image(x)
        },
        |c| {
            cap_calls.set(cap_calls.get() + 1);
            captions_seen.borrow_mut().push(c.to_string());
            mock_encode_caption(c)
        },
        |p| {
            progress
                .borrow_mut()
                .push((p.done, p.total, p.name.to_string(), p.cached))
        },
    )
    .expect("cold prepare");

    assert_eq!(img_calls.get(), 4, "one image encode per example");
    assert_eq!(cap_calls.get(), 4, "one caption encode per example");
    // Progress: one report per entry, in scan order, counting up from 0, and
    // every cold entry reported as a MISS — the flag must track the cache,
    // not be hardcoded (the warm pass below asserts the other polarity).
    assert_eq!(
        *progress.borrow(),
        vec![
            (0, 4, "a.png".to_string(), false),
            (1, 4, "b.png".to_string(), false),
            (2, 4, "c.png".to_string(), false),
            (3, 4, "d.JPG".to_string(), false),
        ],
        "cold pass must report each entry once, uncached"
    );
    // Scan order is filename order; missing captions arrive as "".
    assert_eq!(
        *captions_seen.borrow(),
        vec![
            "a red fox".to_string(),
            String::new(),
            "green field".to_string(),
            String::new()
        ]
    );
    assert_eq!(prepared.items.len(), 4);

    // The three square images share a bucket; the 4:3 one does not.
    assert_eq!(prepared.items[0].bucket, prepared.items[1].bucket);
    assert_eq!(prepared.items[0].bucket, prepared.items[3].bucket);
    assert_ne!(prepared.items[0].bucket, prepared.items[2].bucket);

    // Latents are f8 of the bucket size: 64×64 → [1, 3, 8, 8], and the
    // non-square 80×48 bucket pins the [.., height, width] dim ORDER —
    // an H/W transposition cannot survive this assertion.
    assert_eq!(item(&prepared, 0).latent.dims(), [1, 3, 8, 8]);
    assert_eq!(item(&prepared, 1).latent.dims(), [1, 3, 8, 8]);
    assert_eq!(
        item(&prepared, 2).latent.dims(),
        [1, 3, 6, 10],
        "80×48 bucket → f8 latent [1, 3, 48/8, 80/8]"
    );
    assert_eq!(item(&prepared, 3).latent.dims(), [1, 3, 8, 8]);
    // The plan's recorded shapes must agree with what a load produces — they
    // are what `load_item` checks each read against, so a plan that recorded
    // something else would reject its own cache.
    assert_eq!(prepared.items[2].latent_shape, [1, 3, 6, 10]);

    // Batching groups the square bucket (3 examples → a 2-chunk and a
    // 1-chunk) and never mixes buckets.
    let plans = prepared.batches(2);
    assert_eq!(plans.len(), 3, "square 2+1, non-square 1");
    let sizes: Vec<usize> = plans.iter().map(|p| p.items.len()).collect();
    assert_eq!(sizes.iter().sum::<usize>(), 4);
    for plan in &plans {
        // A plan names one bucket, and every item it lists is in it.
        for &i in &plan.items {
            assert_eq!(prepared.items[i].bucket, plan.bucket, "batch mixed buckets");
        }
        let batch = prepared.load_batch::<B>(plan, &device).expect("load batch");
        let b = batch.latents.dims()[0];
        assert_eq!(b, plan.items.len());
        assert_eq!(batch.conditioning.dims()[0], b);
        assert_eq!(batch.mask.dims()[0], b);
    }
    // Row pairing: the square 2-batch's rows are items 0 and 1 IN ORDER,
    // latents and conditioning aligned (a batch that shuffled one but not
    // the other would train captions against the wrong images).
    let two_plan = plans
        .iter()
        .find(|p| p.items.len() == 2)
        .expect("a 2-batch exists");
    assert_eq!(two_plan.items, vec![0, 1]);
    let two = prepared
        .load_batch::<B>(two_plan, &device)
        .expect("load 2-batch");
    assert_eq!(
        flat(&two.latents.clone().narrow(0, 0, 1)),
        flat(&item(&prepared, 0).latent)
    );
    assert_eq!(
        flat(&two.latents.clone().narrow(0, 1, 1)),
        flat(&item(&prepared, 1).latent)
    );
    // fill = caption length: row 0 is "a red fox" (9), row 1 is "" (0).
    let cond_row0 = flat(&two.conditioning.clone().narrow(0, 0, 1));
    let cond_row1 = flat(&two.conditioning.clone().narrow(0, 1, 1));
    assert!(
        cond_row0.iter().all(|&v| v == 9.0),
        "row 0 pairs with 'a red fox'"
    );
    assert!(cond_row1.iter().all(|&v| v == 0.0), "row 1 pairs with ''");

    let loaded: Vec<PreparedItem<B>> = (0..4).map(|i| item(&prepared, i)).collect();
    let cold_latents: Vec<Vec<f32>> = loaded.iter().map(|i| flat(&i.latent)).collect();
    let cold_conds: Vec<Vec<f32>> = loaded.iter().map(|i| flat(&i.conditioning)).collect();
    let cold_masks: Vec<Vec<i64>> = loaded.iter().map(|i| flat_mask(&i.mask)).collect();
    // The mock mask is non-trivial, so the f32-store → int-reload round trip
    // below is actually exercised.
    assert_eq!(cold_masks[0], vec![1, 1, 1, 0]);

    // --- Warm pass: the cache serves everything; encoders must NOT run. ---
    let warm_progress = std::cell::RefCell::new(Vec::<Report>::new());
    let warm = prepare_dataset::<B>(
        &config,
        "mock-v1",
        &device,
        |_| panic!("image encoder must not run on a warm cache"),
        |_| panic!("caption encoder must not run on a warm cache"),
        |p| {
            warm_progress
                .borrow_mut()
                .push((p.done, p.total, p.name.to_string(), p.cached))
        },
    )
    .expect("warm prepare");
    assert!(
        warm_progress.borrow().iter().all(|r| r.3),
        "a warm pass must report every entry as cached: {:?}",
        warm_progress.borrow()
    );
    assert_eq!(warm_progress.borrow().len(), 4);
    for i in 0..warm.items.len() {
        let loaded = item(&warm, i);
        assert_eq!(
            flat(&loaded.latent),
            cold_latents[i],
            "cached latent must be bit-exact"
        );
        assert_eq!(
            flat(&loaded.conditioning),
            cold_conds[i],
            "cached conditioning must be bit-exact"
        );
        assert_eq!(
            flat_mask(&loaded.mask),
            cold_masks[i],
            "cached mask must round-trip exactly"
        );
    }

    // --- A different fingerprint must miss (no stale cross-encoder reuse). ---
    let img_calls2 = Cell::new(0usize);
    prepare_dataset::<B>(
        &config,
        "mock-v2",
        &device,
        |x| {
            img_calls2.set(img_calls2.get() + 1);
            mock_encode_image(x)
        },
        mock_encode_caption,
        |_| {},
    )
    .expect("fingerprint-miss prepare");
    assert_eq!(img_calls2.get(), 4, "a new fingerprint re-encodes");

    // --- And the two fingerprints coexist: v1's cache survived v2's pass. ---
    prepare_dataset::<B>(
        &config,
        "mock-v1",
        &device,
        |_| panic!("v1 cache must have survived the v2 pass"),
        |_| panic!("v1 cache must have survived the v2 pass"),
        |_| {},
    )
    .expect("v1 still warm after v2");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn image_loading_is_exact() {
    use loractl_core::dataset::load_image_for_bucket;

    let dir = temp_dataset_dir("dataset-pixels");
    let device: burn::tensor::Device<B> = Default::default();
    let bucket = Bucket {
        width: 64,
        height: 64,
    };

    // 1. An image already exactly bucket-sized: no resize, no crop — every
    // value must be pixel / 127.5 - 1 at exactly the CHW position the
    // gradient predicts (pins normalization AND the HWC→CHW indexing).
    write_png(&dir, "exact.png", 64, 64);
    let t = load_image_for_bucket::<B>(&dir.join("exact.png"), bucket, &device).unwrap();
    assert_eq!(t.dims(), [1, 3, 64, 64]);
    let values = flat(&t);
    for y in 0..64usize {
        for x in 0..64usize {
            let expect = [x as f32, y as f32, (x + y) as f32 % 256.0];
            for c in 0..3usize {
                let got = values[c * 64 * 64 + y * 64 + x];
                let want = expect[c] / 127.5 - 1.0;
                assert!(
                    (got - want).abs() < 1e-6,
                    "pixel ({x},{y}) channel {c}: got {got}, want {want}"
                );
            }
        }
    }
    // The extremes map exactly: 0 → -1, 255 → 1 would need a 255-wide image;
    // check the formula endpoints directly instead.
    assert_eq!(0.0f32 / 127.5 - 1.0, -1.0);
    assert_eq!(255.0f32 / 127.5 - 1.0, 1.0);

    // 2. A constant-color image through a REAL downscale (128→64): any
    // interpolation of a constant is that constant, so every output value is
    // pinned exactly, filter-independent.
    let color = [200u8, 10, 90];
    image::RgbImage::from_pixel(128, 128, image::Rgb(color))
        .save(dir.join("flat.png"))
        .unwrap();
    let t = load_image_for_bucket::<B>(&dir.join("flat.png"), bucket, &device).unwrap();
    let values = flat(&t);
    for c in 0..3usize {
        let want = color[c] as f32 / 127.5 - 1.0;
        for (i, &got) in values[c * 64 * 64..(c + 1) * 64 * 64].iter().enumerate() {
            assert!(
                (got - want).abs() < 1e-2,
                "constant downscale drifted at {i}: {got} vs {want}"
            );
        }
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn corrupted_cache_file_is_an_error_not_a_panic() {
    let dir = temp_dataset_dir("dataset-corrupt");
    write_png(&dir, "a.png", 64, 64);
    let config = DatasetConfig {
        path: dir.clone(),
        resolution: RESOLUTION,
        batch_size: 1,
        no_upscale: false,
        bucketing: BucketMode::Aspects,
        min_bucket_resolution: None,
    };
    let device = Default::default();

    prepare_dataset::<B>(
        &config,
        "mock-v1",
        &device,
        mock_encode_image,
        mock_encode_caption,
        |_| {},
    )
    .expect("cold prepare");

    // Garbage every cache file, then re-prepare: the pipeline must surface a
    // parse error, not panic and not silently re-encode.
    let cache_dir = dir.join(".loractl-cache");
    let cache_files: Vec<PathBuf> = std::fs::read_dir(&cache_dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    for path in &cache_files {
        std::fs::write(path, b"not a safetensors file").unwrap();
    }
    let result = prepare_dataset::<B>(
        &config,
        "mock-v1",
        &device,
        mock_encode_image,
        mock_encode_caption,
        |_| {},
    );
    let err = format!("{:#}", result.expect_err("corrupted cache must error"));
    assert!(
        err.contains("parsing cache file"),
        "error should localize the bad cache file: {err}"
    );
    // …and the training path fails the same way, at PLAN time — before the
    // model is loaded — because the header parse happens there (#175).
    let err = format!(
        "{:#}",
        plan_dataset(&config, "mock-v1", |_| {}).expect_err("corrupted cache must error")
    );
    assert!(
        err.contains("parsing cache file"),
        "plan_dataset should localize the bad cache file: {err}"
    );

    // A TRUNCATED file — valid header, short data — is the case a header-only
    // read could plausibly wave through. safetensors' `read_metadata`
    // validates `header + data == file length`, so it does not.
    std::fs::remove_dir_all(&cache_dir).unwrap();
    prepare_dataset::<B>(
        &config,
        "mock-v1",
        &device,
        mock_encode_image,
        mock_encode_caption,
        |_| {},
    )
    .expect("re-encode after wiping the cache");
    for path in &cache_files {
        let bytes = std::fs::read(path).unwrap();
        std::fs::write(path, &bytes[..bytes.len() - 4]).unwrap();
    }
    let err = format!(
        "{:#}",
        plan_dataset(&config, "mock-v1", |_| {}).expect_err("truncated cache must error")
    );
    assert!(
        err.contains("parsing cache file"),
        "a truncated cache file must be caught at plan time: {err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The mechanical form of the warm-cache guarantee (#175).
///
/// The panicking-closure test above proves the encoders *do not fire*; this
/// one proves the training path *has no encoders to fire*. `plan_dataset`
/// takes no encoder arguments at all, so unlike a closure-based proof it
/// cannot be weakened to "usually does not encode" — the call site is the
/// evidence.
#[test]
fn warm_planning_never_needs_an_encoder() {
    let dir = temp_dataset_dir("dataset-plan-warm");
    write_four_image_fixture(&dir);
    let config = config_for(&dir);

    prepare_dataset::<B>(
        &config,
        "mock-v1",
        &Default::default(),
        mock_encode_image,
        mock_encode_caption,
        |_| {},
    )
    .expect("cold prepare");

    let planned = plan_dataset(&config, "mock-v1", |_| {}).expect("warm plan");
    assert_eq!(planned.items.len(), 4);
    // The plan's shapes are the mocks' outputs, read back from the file
    // headers alone: the 8× pool of each bucket, and the [1, 4, 2, 8] / [1, 4]
    // conditioning pair.
    assert_eq!(planned.items[0].latent_shape, [1, 3, 8, 8]);
    assert_eq!(planned.items[1].latent_shape, [1, 3, 8, 8]);
    assert_eq!(planned.items[2].latent_shape, [1, 3, 6, 10]);
    assert_eq!(planned.items[3].latent_shape, [1, 3, 8, 8]);
    for cached in &planned.items {
        assert_eq!(cached.cond_shape, [1, 4, 2, 8]);
        assert_eq!(cached.mask_shape, [1, 4]);
    }
    // And the loaded values match what the encode pass produced.
    assert_eq!(flat_mask(&item(&planned, 0).mask), vec![1, 1, 1, 0]);

    std::fs::remove_dir_all(&dir).ok();
}

/// The bail-on-miss contract at PLAN granularity: a cold cache on the
/// training path is a hard error that names the file it wanted.
#[test]
fn planning_a_cold_dataset_names_the_missing_file() {
    let dir = temp_dataset_dir("dataset-plan-cold");
    write_png(&dir, "a.png", 64, 64);
    let config = config_for(&dir);

    let err = format!(
        "{:#}",
        plan_dataset(&config, "mock-v1", |_| {}).expect_err("a cold cache must error")
    );
    assert!(
        err.contains("cache miss after the encode phase"),
        "the bail-on-miss wording must stay recognizable: {err}"
    );
    assert!(
        err.contains("a.png.64x64."),
        "the error must name the file it wanted: {err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The bail-on-miss contract at STEP granularity (#175): now that batches are
/// read per step, a cache file that disappears *after* planning must be a
/// loud error rather than a silently skipped or empty batch.
#[test]
fn a_cache_file_deleted_after_planning_is_a_loud_error() {
    let dir = temp_dataset_dir("dataset-vanish");
    write_four_image_fixture(&dir);
    let config = config_for(&dir);

    prepare_dataset::<B>(
        &config,
        "mock-v1",
        &Default::default(),
        mock_encode_image,
        mock_encode_caption,
        |_| {},
    )
    .expect("cold prepare");
    let prepared = plan_dataset(&config, "mock-v1", |_| {}).expect("warm plan");
    let plans = prepared.batches(1);

    // Delete item 0's conditioning, then load the batch that contains it.
    std::fs::remove_file(&prepared.items[0].cond_path).unwrap();
    let plan = plans
        .iter()
        .find(|p| p.items.contains(&0))
        .expect("item 0 is in some batch");
    let err = err_of(
        prepared.load_batch::<B>(plan, &Default::default()),
        "a vanished cache file must error",
    );
    assert!(
        err.contains("disappeared mid-run") && err.contains("cond.safetensors"),
        "the error must name the path and say the dataset must not change: {err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The loaders draw **no RNG** — the load-bearing correctness claim of #175,
/// now that materialization happens inside the step loop.
///
/// burn's backend RNG is one process-global stream, and every frozen `Param`
/// in this workspace materializes lazily *out of it*
/// (`.claude/rules/burn-lazy-param-init.md`), so a single extra draw inside
/// `load_batch` would move where every base weight lands in that stream and
/// silently shift every pinned loss trajectory. The e2e test's
/// reseeded-rerun assertion cannot see that — it compares two runs of the
/// *same* code, so it detects nondeterminism but not a systematic shift, and
/// a shuffle with a fixed seed added to `load_batch` would pass it.
///
/// So the property is pinned directly, and without a float golden: seed,
/// draw, reseed, load every batch of an epoch, draw again. The two draws must
/// be bit-identical. (No other test in this binary draws from the backend
/// RNG, so the parallel test threads cannot perturb this one.)
#[test]
fn loading_a_batch_draws_no_rng() {
    use burn::tensor::Distribution;
    use burn::tensor::backend::Backend;

    let dir = temp_dataset_dir("dataset-rng");
    write_four_image_fixture(&dir);
    let config = config_for(&dir);
    let device: burn::tensor::Device<B> = Default::default();

    prepare_dataset::<B>(
        &config,
        "mock-v1",
        &device,
        mock_encode_image,
        mock_encode_caption,
        |_| {},
    )
    .expect("cold prepare");
    let prepared = plan_dataset(&config, "mock-v1", |_| {}).expect("warm plan");
    let plans = prepared.batches(2);
    assert!(plans.len() > 1, "several batches, so several loads");

    let draw = |device: &burn::tensor::Device<B>| -> Vec<f32> {
        flat(&Tensor::<B, 4>::random(
            [1, 2, 4, 4],
            Distribution::Default,
            device,
        ))
    };

    B::seed(&device, 20260806);
    let reference = draw(&device);

    B::seed(&device, 20260806);
    for plan in &plans {
        let batch = prepared.load_batch::<B>(plan, &device).expect("load batch");
        std::hint::black_box(&batch);
    }
    let after_loading = draw(&device);

    assert_eq!(
        reference,
        after_loading,
        "loading {} batches consumed backend RNG — every lazily-initialized \
         frozen Param would now materialize elsewhere in the stream",
        plans.len()
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A cache file *rewritten* between planning and loading is a different
/// training example wearing the same name. The pre-#175 design could not
/// detect this at all — it read whatever was on disk and trained on it.
#[test]
fn a_cache_file_reshaped_after_planning_is_a_loud_error() {
    let dir = temp_dataset_dir("dataset-reshape");
    write_four_image_fixture(&dir);
    let config = config_for(&dir);

    prepare_dataset::<B>(
        &config,
        "mock-v1",
        &Default::default(),
        mock_encode_image,
        mock_encode_caption,
        |_| {},
    )
    .expect("cold prepare");
    let prepared = plan_dataset(&config, "mock-v1", |_| {}).expect("warm plan");
    assert_eq!(prepared.items[0].cond_shape, [1, 4, 2, 8]);

    // Rewrite item 0's conditioning at a different sequence length — exactly
    // what a re-captioned example under a longer `max_length` would produce.
    // Written with the raw safetensors writer so the file is genuinely valid;
    // only its SHAPE disagrees with the plan.
    let device: burn::tensor::Device<B> = Default::default();
    let f32_bytes = |n: usize| -> Vec<u8> {
        std::iter::repeat_n(1.0f32, n)
            .flat_map(f32::to_le_bytes)
            .collect()
    };
    let cond_bytes = f32_bytes(8 * 2 * 8);
    let mask_bytes = f32_bytes(8);
    safetensors::serialize_to_file(
        vec![
            (
                "conditioning",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![1, 8, 2, 8],
                    &cond_bytes,
                )
                .unwrap(),
            ),
            (
                "mask",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![1, 8],
                    &mask_bytes,
                )
                .unwrap(),
            ),
        ],
        None,
        &prepared.items[0].cond_path,
    )
    .unwrap();

    let err = err_of(
        prepared.load_item::<B>(0, &device),
        "a reshaped cache file must error",
    );
    assert!(
        err.contains("changed since the run was planned")
            && err.contains("[1, 4, 2, 8]")
            && err.contains("[1, 8, 2, 8]"),
        "the error must name the path plus the planned AND found shapes: {err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn empty_dataset_folder_fails_fast() {
    let dir = temp_dataset_dir("dataset-empty");
    let config = DatasetConfig {
        path: dir.clone(),
        resolution: RESOLUTION,
        batch_size: 1,
        no_upscale: false,
        bucketing: BucketMode::Aspects,
        min_bucket_resolution: None,
    };
    let device = Default::default();
    let result = prepare_dataset::<B>(&config, "mock-v1", &device, Ok, mock_encode_caption, |_| {});
    assert!(result.is_err(), "an imageless dataset dir must error");
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// #147 (`no_upscale`) and #148 (`bucketing: grid`)
// ---------------------------------------------------------------------------

/// A defaulted `DatasetConfig` over `dir` — the shape every knob test varies
/// exactly one field of.
fn config_for(dir: &std::path::Path) -> DatasetConfig {
    DatasetConfig {
        path: dir.to_path_buf(),
        resolution: RESOLUTION,
        batch_size: 1,
        ..Default::default()
    }
}

/// The four-image fixture the cache/bucket tests share: two square PNGs (one
/// of them needing an upscale into the 64×64 bucket), a 4:3 PNG, and a square
/// JPEG with an uppercase extension.
fn write_four_image_fixture(dir: &std::path::Path) {
    write_png(dir, "a.png", 64, 64);
    write_png(dir, "b.png", 32, 32);
    write_png(dir, "c.png", 100, 75);
    let jpg = image::RgbImage::from_fn(64, 64, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, 0])
    });
    jpg.save_with_format(dir.join("d.JPG"), image::ImageFormat::Jpeg)
        .expect("write test jpeg");
    std::fs::write(dir.join("a.txt"), "a red fox\n").unwrap();
    std::fs::write(dir.join("c.txt"), "green field").unwrap();
}

/// Sorted file names in `<dir>/.loractl-cache/`.
fn cache_names(dir: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir.join(".loractl-cache"))
        .expect("cache dir")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// A value snapshot of the DEFAULT generator at four resolutions, taken from
/// today's code. `generate_buckets` is not edited by #148 — the grid is a new
/// function behind a new dispatcher — but "not edited" is a claim about a
/// diff, and this is a claim about values: any accidental perturbation of the
/// seven-ratio set, its rounding, or its **order** (bucket index 0 is the
/// square bucket everywhere downstream) fails here.
#[test]
fn default_bucket_sets_are_frozen() {
    assert_eq!(
        generate_buckets(32).unwrap(),
        vec![b(32, 32), b(48, 32), b(32, 48)]
    );
    assert_eq!(
        generate_buckets(64).unwrap(),
        vec![b(64, 64), b(80, 48), b(48, 80)]
    );
    assert_eq!(
        generate_buckets(512).unwrap(),
        vec![
            b(512, 512),
            b(592, 448),
            b(448, 592),
            b(624, 416),
            b(416, 624),
            b(688, 384),
            b(384, 688),
        ]
    );
    assert_eq!(
        generate_buckets(1024).unwrap(),
        vec![
            b(1024, 1024),
            b(1184, 880),
            b(880, 1184),
            b(1248, 832),
            b(832, 1248),
            b(1360, 768),
            b(768, 1360),
        ]
    );
}

/// The new dispatcher is the identity on the default path — the one property
/// that makes "every existing config trains exactly as before" a fact rather
/// than a reading of the diff.
#[test]
fn bucket_set_default_matches_generate_buckets() {
    for resolution in [32, 64, 512] {
        let config = DatasetConfig {
            resolution,
            ..Default::default()
        };
        assert_eq!(
            bucket_set(&config).unwrap(),
            generate_buckets(resolution).unwrap(),
            "bucket_set must be generate_buckets at resolution {resolution}"
        );
    }
}

/// The on-disk cache naming — `{file_name}.{w}x{h}.{fingerprint}.latent.…`
/// and `{stem}.{fingerprint}.cond.…` — is an **external contract**:
/// `reference/krea2_lora_train.py::load_cache` globs exactly these shapes, and
/// nothing in the tree pinned them. Freezing the full default file set pins
/// the naming scheme, the fingerprint's sanitize+FNV-1a suffix, AND the
/// default bucket geometry each latent name carries, in one snapshot.
#[test]
fn default_cache_filenames_are_frozen() {
    let dir = temp_dataset_dir("dataset-names");
    write_four_image_fixture(&dir);

    prepare_dataset::<B>(
        &config_for(&dir),
        "mock-v1",
        &Default::default(),
        mock_encode_image,
        mock_encode_caption,
        |_| {},
    )
    .expect("cold prepare");

    // "mock-v1" sanitizes to itself; the suffix is FNV-1a(64) of the RAW
    // fingerprint, so a change to either half shows up here.
    const FP: &str = "mock-v1-98a68dc3451e9947";
    assert_eq!(
        cache_names(&dir),
        vec![
            format!("a.{FP}.cond.safetensors"),
            format!("a.png.64x64.{FP}.latent.safetensors"),
            format!("b.{FP}.cond.safetensors"),
            format!("b.png.64x64.{FP}.latent.safetensors"),
            format!("c.{FP}.cond.safetensors"),
            format!("c.png.80x48.{FP}.latent.safetensors"),
            format!("d.JPG.64x64.{FP}.latent.safetensors"),
            format!("d.{FP}.cond.safetensors"),
        ]
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The discriminator for #147, at the level that matters: exactly ONE of the
/// four fixture images is treated differently. `b.png` (32×32) stops being
/// stretched into the 64×64 bucket and gets a 32×32 box of its own; `a.png`
/// and `d.JPG` already fill theirs, and `c.png` (100×75 → 80×48) was always a
/// downscale — `no_upscale` has nothing to say about any of them.
#[test]
fn no_upscale_shrinks_the_bucket_instead_of_upscaling() {
    let dir = temp_dataset_dir("dataset-noup");
    write_four_image_fixture(&dir);
    let config = DatasetConfig {
        no_upscale: true,
        ..config_for(&dir)
    };

    let prepared = prepare_dataset::<B>(
        &config,
        "mock-v1",
        &Default::default(),
        mock_encode_image,
        mock_encode_caption,
        |_| {},
    )
    .expect("cold prepare");

    assert_eq!(
        item(&prepared, 0).latent.dims(),
        [1, 3, 8, 8],
        "a.png 64×64"
    );
    assert_eq!(
        item(&prepared, 1).latent.dims(),
        [1, 3, 4, 4],
        "b.png 32×32 must get a 32×32 box, not be upscaled to 64×64"
    );
    assert_eq!(
        item(&prepared, 2).latent.dims(),
        [1, 3, 6, 10],
        "c.png was already a downscale into 80×48"
    );
    assert_eq!(
        item(&prepared, 3).latent.dims(),
        [1, 3, 8, 8],
        "d.JPG 64×64"
    );

    // The derived box is APPENDED to the generated set, never substituted for
    // a generated one — the three fixed buckets keep their indices, which is
    // what lets a mixed dataset batch normally.
    assert_eq!(
        prepared.buckets,
        vec![b(64, 64), b(80, 48), b(48, 80), b(32, 32)]
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The property behind the spot check, plus the floor: with `no_upscale`, no
/// entry's box may exceed its source in either dimension **or
/// [`BUCKET_ALIGN`], whichever is larger** — a zero-sided bucket is not a
/// bucket, so a source under 16 px on a side is the one case that still gets
/// scaled up, into a 16×16 box.
///
/// That `.max(BUCKET_ALIGN)` is the whole point of the `sub16.png` fixture
/// below. The property used to be written as the flat
/// `bucket.width <= w && bucket.height <= h`, which the implementation does
/// **not** hold and `fit_bucket_shrinks_only_when_the_source_is_smaller`
/// pins the opposite of (`fit_bucket(64×64, 4, 4) == 16×16`); the two only
/// coexisted because no fixture image was smaller than 16 px. A property test
/// that states a stronger invariant than the code has reads as evidence it
/// is not, which is worse than not having it.
#[test]
fn no_upscale_never_upscales_any_entry_and_floors_at_bucket_align() {
    let dir = temp_dataset_dir("dataset-noup-prop");
    write_four_image_fixture(&dir);
    // A deliberately tiny source: 20×20 fits a 16×16 box (a 40×40 thumbnail in
    // a 512px dataset is the real-world shape of this). Documented as a
    // near-useless training example, but it must not error or truncate to 0.
    write_png(&dir, "tiny.png", 20, 20);
    // …and one *below* the floor, which therefore IS upscaled (8 → 16). The
    // only case where `no_upscale` does not mean "never upscale", and the
    // reason the assertion below is written against `max(BUCKET_ALIGN)`.
    write_png(&dir, "sub16.png", 8, 8);
    let config = DatasetConfig {
        no_upscale: true,
        ..config_for(&dir)
    };

    let prepared = prepare_dataset::<B>(
        &config,
        "mock-v1",
        &Default::default(),
        mock_encode_image,
        mock_encode_caption,
        |_| {},
    )
    .expect("cold prepare");

    assert_no_entry_is_upscaled_past_the_floor(&prepared);
    let box_of = |name: &str| {
        let i = prepared
            .entries
            .iter()
            .position(|e| e.image_path.ends_with(name))
            .unwrap_or_else(|| panic!("{name} scanned"));
        prepared.buckets[prepared.entries[i].bucket]
    };
    assert_eq!(box_of("tiny.png"), b(16, 16));
    // The floor branch, exercised end to end rather than only in the unit
    // test: 8×8 is below one alignment step, so it is floored *up*.
    assert_eq!(box_of("sub16.png"), b(16, 16));

    std::fs::remove_dir_all(&dir).ok();
}

/// The `no_upscale` box property, shared by the two tests that assert it
/// (default and grid bucketing): every box is aligned, and no larger than its
/// source **or one alignment step**, whichever is larger.
fn assert_no_entry_is_upscaled_past_the_floor(prepared: &PreparedDataset) {
    for entry in &prepared.entries {
        let (w, h) = image::image_dimensions(&entry.image_path).unwrap();
        let bucket = prepared.buckets[entry.bucket];
        assert!(
            bucket.width <= w.max(BUCKET_ALIGN) && bucket.height <= h.max(BUCKET_ALIGN),
            "{} is {w}×{h} but got bucket {bucket:?}",
            entry.image_path.display()
        );
        assert_eq!(bucket.width % BUCKET_ALIGN, 0, "{bucket:?} width unaligned");
        assert_eq!(
            bucket.height % BUCKET_ALIGN,
            0,
            "{bucket:?} height unaligned"
        );
    }
}

/// `fit_bucket` on its own, including the cases the fixture cannot reach.
#[test]
fn fit_bucket_shrinks_only_when_the_source_is_smaller() {
    // Already covering the bucket → untouched (the common downscale case).
    assert_eq!(fit_bucket(b(512, 512), 1024, 768), b(512, 512));
    assert_eq!(fit_bucket(b(80, 48), 100, 75), b(80, 48));
    assert_eq!(fit_bucket(b(64, 64), 64, 64), b(64, 64));
    // Smaller in both dimensions → aligned down to fit.
    assert_eq!(fit_bucket(b(64, 64), 32, 32), b(32, 32));
    assert_eq!(fit_bucket(b(512, 512), 300, 300), b(288, 288));
    // Smaller in ONE dimension: the binding side sets the scale, so the aspect
    // is preserved rather than the box being squashed to the source.
    assert_eq!(fit_bucket(b(512, 512), 600, 400), b(400, 400));
    // Below one alignment step → floored, never zero.
    assert_eq!(fit_bucket(b(64, 64), 4, 4), b(16, 16));
}

/// Flipping `no_upscale` needs **no cache-fingerprint bump**, because the
/// latent key already carries the bucket box. Proven the only way that means
/// anything: the unaffected latents must be byte-identical and *un-rewritten*,
/// a new file must appear for the affected one, and the conditioning cache —
/// the expensive half, the 4B text encoder — must not be touched at all.
#[test]
fn no_upscale_reuses_the_unaffected_latents_and_leaves_conditioning_alone() {
    let dir = temp_dataset_dir("dataset-noup-cache");
    write_four_image_fixture(&dir);

    prepare_dataset::<B>(
        &config_for(&dir),
        "mock-v1",
        &Default::default(),
        mock_encode_image,
        mock_encode_caption,
        |_| {},
    )
    .expect("default prepare");

    let cache_dir = dir.join(".loractl-cache");
    let snapshot = |name: &str| -> (Vec<u8>, std::time::SystemTime) {
        let path = cache_dir.join(name);
        let meta = std::fs::metadata(&path).expect("cache file");
        (std::fs::read(&path).unwrap(), meta.modified().unwrap())
    };
    const FP: &str = "mock-v1-98a68dc3451e9947";
    let before: Vec<(String, (Vec<u8>, std::time::SystemTime))> = cache_names(&dir)
        .into_iter()
        .map(|n| {
            let s = snapshot(&n);
            (n, s)
        })
        .collect();

    // The image encoder may run at most ONCE on the second pass — only
    // `b.png`, whose box changed from 64×64 to 32×32 — and the caption encoder
    // must not run at all.
    let img_calls = Cell::new(0usize);
    prepare_dataset::<B>(
        &DatasetConfig {
            no_upscale: true,
            ..config_for(&dir)
        },
        "mock-v1",
        &Default::default(),
        |x| {
            img_calls.set(img_calls.get() + 1);
            mock_encode_image(x)
        },
        |_| panic!("flipping no_upscale must not re-encode any caption"),
        |_| {},
    )
    .expect("no_upscale prepare");
    assert_eq!(img_calls.get(), 1, "only b.png's box changed");

    // Every pre-existing file is still there, byte-identical and un-rewritten
    // (the superseded 64×64 latent for b.png is orphaned, not deleted — it is
    // correct again the moment the knob is flipped back).
    for (name, (bytes, mtime)) in &before {
        let (now_bytes, now_mtime) = snapshot(name);
        assert_eq!(&now_bytes, bytes, "{name} was rewritten");
        assert_eq!(&now_mtime, mtime, "{name}'s mtime moved");
    }
    // …and exactly one new file appeared: b.png's smaller box.
    let after = cache_names(&dir);
    let new: Vec<&String> = after
        .iter()
        .filter(|n| !before.iter().any(|(old, _)| old == *n))
        .collect();
    assert_eq!(new, vec![&format!("b.png.32x32.{FP}.latent.safetensors")]);

    std::fs::remove_dir_all(&dir).ok();
}

/// The teeth for the sort-before-assign restructure.
///
/// `no_upscale` grows the bucket set *during* assignment, so if assignment ran
/// in `read_dir` order — as it did before #147, harmlessly, because the set
/// was fixed — the derived buckets would be **appended in filesystem order**.
/// `batches()` iterates buckets by index, so batch order (and hence every
/// pinned loss trajectory in `diffusion_trainer.rs`) would become
/// filesystem-dependent.
///
/// The bite comes from asserting the exact bucket vec over eight sources whose
/// **lexicographic** order differs from the directory's enumeration order.
/// ext4's hashed directory index (and tmpfs's insertion order) scrambles these
/// names; under the pre-restructure code the first *distinct* derived box
/// would then be 48×48 or 16×16 rather than 32×32, and the vec below fails.
/// The files are therefore created in **reverse** lexicographic order, which
/// is the worst case for any filesystem that enumerates in insertion order.
///
/// That is still a fact about *this* filesystem, so the test does not rest on
/// it: `prepared.entries` being sorted by path is asserted directly, and that
/// assertion has the same teeth on every filesystem — it is the property the
/// restructure exists to provide (assignment happens in sorted order, so the
/// derived boxes are appended in sorted order, so bucket indices — and hence
/// batch order, and hence every pinned loss trajectory — are
/// filesystem-independent).
#[test]
fn bucket_indices_are_stable_regardless_of_directory_order() {
    let dir = temp_dataset_dir("dataset-bucket-order");
    // All square, so all attracted to the 64×64 bucket; the derived boxes are
    // what differ. Lexicographic order fixes the append order as
    // 32×32 (b), 48×48 (c), 16×16 (d) — every later image dedups into one of
    // them, and `e` needs no derived box at all.
    let fixture = [
        ("a.png", 64u32),
        ("b.png", 32),
        ("c.png", 48),
        ("d.png", 20),
        ("e.png", 100),
        ("f.png", 40),
        ("g.png", 56),
        ("h.png", 24),
    ];
    for (name, size) in fixture.iter().rev() {
        write_png(&dir, name, *size, *size);
    }

    let enumerated: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    let mut sorted = enumerated.clone();
    sorted.sort();
    if enumerated == sorted {
        eprintln!(
            "note: this filesystem enumerates in sorted order despite reverse-order \
             creation, so the bucket-vec assertion below is weak here — the \
             entries-are-sorted assertion is what carries the test on such a host"
        );
    }

    let prepared = prepare_dataset::<B>(
        &DatasetConfig {
            no_upscale: true,
            ..config_for(&dir)
        },
        "mock-v1",
        &Default::default(),
        mock_encode_image,
        mock_encode_caption,
        |_| {},
    )
    .expect("prepare");

    // The filesystem-INDEPENDENT half: assignment runs over a path-sorted
    // scan, so this holds whatever `read_dir` returned. Everything below is a
    // consequence of it.
    let paths: Vec<&PathBuf> = prepared.entries.iter().map(|e| &e.image_path).collect();
    let mut want = paths.clone();
    want.sort();
    assert_eq!(paths, want, "entries must be scanned in sorted path order");

    assert_eq!(
        prepared.buckets,
        vec![
            b(64, 64),
            b(80, 48),
            b(48, 80),
            b(32, 32), // b.png
            b(48, 48), // c.png
            b(16, 16), // d.png
        ],
        "derived buckets must be appended in SORTED scan order, not {enumerated:?}"
    );
    let assigned: Vec<(String, usize)> = prepared
        .entries
        .iter()
        .map(|e| {
            (
                e.image_path.file_name().unwrap().to_string_lossy().into(),
                e.bucket,
            )
        })
        .collect();
    assert_eq!(
        assigned,
        vec![
            ("a.png".to_string(), 0), // 64×64, already covers its bucket
            ("b.png".to_string(), 3), // 32×32
            ("c.png".to_string(), 4), // 48×48
            ("d.png".to_string(), 5), // 20×20 → floored to 16×16
            ("e.png".to_string(), 0), // 100×100 → a plain downscale
            ("f.png".to_string(), 3), // 40×40 → 32×32, dedups into b's box
            ("g.png".to_string(), 4), // 56×56 → 48×48, dedups into c's box
            ("h.png".to_string(), 5), // 24×24 → 16×16, dedups into d's box
        ]
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The grid generator's own contract (#148). The area bound is asserted
/// **strictly**, which the fixed set does not satisfy — 688×384 = 264 192 >
/// 512² — so this is a real difference between the two generators, not a
/// restatement of "buckets are about the right size".
///
/// The last three rows sit at the `min_side = resolution / 4` floor, and they
/// are here to pin the *count*: `generate_grid_buckets`' doc used to claim
/// that floor "caps the bucket count at ~200", which is only true at 512px —
/// the count is `≈ 0.23 · resolution` and scales with it. Since the whole
/// documented cost of grid mode is the partial-batch degradation, a count
/// understated 4× at a shipped resolution is exactly the wrong direction, so
/// the doc's numbers are asserted rather than asserted-about.
#[test]
fn grid_buckets_are_aligned_symmetric_and_area_bounded() {
    for (resolution, min_side, count) in [
        (512u32, 256u32, 65usize),
        (64, 32, 9),
        (32, 16, 5),
        // At the resolution/4 floor — the doc's own numbers.
        (512, 128, 193),
        (1024, 256, 385),
        (2048, 512, 769),
    ] {
        let buckets = generate_grid_buckets(resolution, min_side).expect("valid grid");
        assert_eq!(buckets.len(), count, "grid({resolution}, {min_side}) size");
        let area = (resolution as u64).pow(2);
        for bucket in &buckets {
            assert_eq!(bucket.width % BUCKET_ALIGN, 0, "{bucket:?} width unaligned");
            assert_eq!(
                bucket.height % BUCKET_ALIGN,
                0,
                "{bucket:?} height unaligned"
            );
            assert!(
                bucket.width as u64 * bucket.height as u64 <= area,
                "{bucket:?} exceeds the {resolution}² area budget"
            );
            assert!(
                bucket.width.min(bucket.height) >= min_side,
                "{bucket:?} is thinner than min_side {min_side}"
            );
            assert!(
                buckets.contains(&b(bucket.height, bucket.width)),
                "{bucket:?} has no transpose — the set is not symmetric"
            );
        }
        for (i, x) in buckets.iter().enumerate() {
            for y in &buckets[i + 1..] {
                assert_ne!(x, y, "duplicate bucket in grid({resolution}, {min_side})");
            }
        }
        assert!(
            buckets.contains(&b(resolution, resolution)),
            "the square bucket must survive grid generation"
        );
        let aspects: Vec<f64> = buckets
            .iter()
            .map(|x| x.width as f64 / x.height as f64)
            .collect();
        let lo = aspects.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = aspects.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // The extreme aspect is `(resolution / min_side)²` — the derivation
        // the `min_side ≥ resolution/4` rule quotes, and the half of that
        // rule's rationale that really is resolution-independent.
        let extreme = (resolution as f64 / min_side as f64).powi(2);
        assert!(
            (lo - 1.0 / extreme).abs() < 1e-9 && (hi - extreme).abs() < 1e-9,
            "grid({resolution}, {min_side}) spans {lo}..{hi}, expected \
             {}..{extreme}",
            1.0 / extreme
        );
    }

    // The contrast the strict area bound is there for.
    assert!(
        generate_buckets(512)
            .unwrap()
            .iter()
            .any(|x| x.width as u64 * x.height as u64 > 512 * 512),
        "the fixed set overshoots resolution² — that is what the grid fixes"
    );
}

/// The point of #148, as a number rather than a claim. Crop loss depends only
/// on aspect mismatch, so this is measurable offline and needs no GPU.
///
/// It is also the kill-test for the generator: if `generate_grid_buckets`
/// degenerated into the fixed set (or into anything with comparably sparse
/// aspect coverage), the worst-case bound below fails.
#[test]
fn grid_bucketing_reduces_worst_case_crop_loss() {
    let aspects = generate_buckets(512).unwrap();
    let grid = generate_grid_buckets(512, 256).unwrap();

    let worst = |buckets: &[Bucket]| -> f64 {
        (0..=3750)
            .map(|i| {
                let r = 0.25 + i as f64 * 0.001;
                crop_loss(buckets, r * 1000.0, 1000.0)
            })
            .fold(0.0f64, f64::max)
    };
    let worst_aspects = worst(&aspects);
    let worst_grid = worst(&grid);
    assert!(
        worst_aspects > 0.5,
        "the fixed set is expected to discard >50% somewhere in 0.25..4.0, got {worst_aspects}"
    );
    assert!(
        worst_grid < 0.05,
        "grid mode must cap the worst case near 4%, got {worst_grid}"
    );

    // The two shapes the issue names, exactly.
    for (w, h) in [(1600.0, 400.0), (400.0, 1600.0)] {
        assert!(
            crop_loss(&aspects, w, h) > 0.5,
            "{w}×{h} should be a disaster under the fixed set"
        );
        assert_eq!(
            crop_loss(&grid, w, h),
            0.0,
            "{w}×{h} is exactly a grid bucket's aspect"
        );
    }

    // Honest about the trade: for the seven ratios the fixed list was
    // hand-picked for, the 16px grid step is slightly WORSE. Documented in the
    // README and in `BucketMode::Grid`'s doc — pinned here so the docs cannot
    // quietly become a one-sided sales pitch.
    assert!(
        crop_loss(&grid, 1024.0, 768.0) > crop_loss(&aspects, 1024.0, 768.0),
        "4:3 is the fixed set's home turf"
    );
}

/// Every grid/`min_bucket_resolution` misconfiguration arrives from user YAML,
/// so every one of them is an `Err` naming the field — never a panic, and
/// never (rule 5) a knob that is silently ignored.
#[test]
fn grid_mode_config_errors_are_errors_not_panics() {
    let case = |bucketing, resolution, min_bucket_resolution| DatasetConfig {
        resolution,
        bucketing,
        min_bucket_resolution,
        ..Default::default()
    };
    for (config, needle) in [
        (
            case(BucketMode::Grid, 512, Some(250)),
            "must be a non-zero multiple",
        ),
        (case(BucketMode::Grid, 512, Some(0)), "must be a non-zero"),
        (case(BucketMode::Grid, 512, Some(768)), "exceeds"),
        // `min_side == resolution` passes every *other* rule and yields a
        // one-bucket set: every image center-cropped square, i.e. strictly
        // worse than the `aspects` default the user opted out of, with no
        // signal but the encode phase's bucket count.
        (
            case(BucketMode::Grid, 512, Some(512)),
            "collapses to the single 512×512 square bucket",
        ),
        (
            case(BucketMode::Grid, 512, Some(64)),
            "below dataset.resolution / 4",
        ),
        (
            case(BucketMode::Aspects, 512, Some(256)),
            "only meaningful with `bucketing: grid`",
        ),
        (case(BucketMode::Grid, 1000, None), "multiple of 16"),
        (case(BucketMode::Aspects, 1000, None), "multiple of 16"),
    ] {
        let err = bucket_set(&config).expect_err("misconfiguration must error");
        let text = format!("{err:#}");
        assert!(
            text.contains(needle),
            "expected {needle:?} in the error, got: {text}"
        );
    }

    // …and the valid shapes really do resolve, so the table above is not just
    // asserting that everything fails.
    assert_eq!(
        bucket_set(&case(BucketMode::Grid, 512, Some(256))).unwrap(),
        generate_grid_buckets(512, 256).unwrap()
    );
    // `None` → resolution/2, aligned down: aspects from 1:4 to 4:1.
    assert_eq!(
        bucket_set(&case(BucketMode::Grid, 512, None)).unwrap(),
        generate_grid_buckets(512, 256).unwrap()
    );
}

/// The two knobs are orthogonal, not accidentally coupled: `no_upscale`'s
/// `fit_bucket` runs against whatever set the mode produced, and every
/// resulting box is still aligned and still no larger than its source.
#[test]
fn no_upscale_composes_with_grid() {
    let dir = temp_dataset_dir("dataset-grid-noup");
    write_four_image_fixture(&dir);
    let config = DatasetConfig {
        bucketing: BucketMode::Grid,
        no_upscale: true,
        ..config_for(&dir)
    };

    let prepared = prepare_dataset::<B>(
        &config,
        "mock-v1",
        &Default::default(),
        mock_encode_image,
        mock_encode_caption,
        |_| {},
    )
    .expect("grid + no_upscale prepare");

    assert_no_entry_is_upscaled_past_the_floor(&prepared);

    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// #178 — parallel decode, and the bit-identity it must not buy at the cost of.
// ---------------------------------------------------------------------------

/// The **pre-#178 image loader, verbatim**: `image::open` → `to_rgb8` →
/// Lanczos3 cover-resize → `crop_imm` → `to_image()` (the second full-frame
/// allocation + copy #178 removes) → a per-pixel
/// `data[c * bh * bw + y * bw + x]` write.
///
/// This oracle exists because the cache keys are name/bucket/fingerprint and
/// **never content**: a resize or transpose change invalidates nothing on
/// disk, so an optimization that shifted a value would silently mix
/// old-algorithm and new-algorithm latents inside one adapter, with no error
/// anywhere. "Bit-identical to the serial path" is therefore the acceptance
/// criterion, and this function is what it is measured against — so it is
/// copied here rather than deleted, and must not be "tidied" to track the
/// implementation.
fn decode_serial_reference(path: &std::path::Path, bucket: Bucket) -> Vec<f32> {
    let img = image::open(path).expect("decode").to_rgb8();
    let (w, h) = (img.width(), img.height());
    let (bw, bh) = (bucket.width, bucket.height);

    let scale = f64::max(bw as f64 / w as f64, bh as f64 / h as f64);
    let rw = (w as f64 * scale).ceil() as u32;
    let rh = (h as f64 * scale).ceil() as u32;
    let resized = image::imageops::resize(&img, rw, rh, image::imageops::FilterType::Lanczos3);
    let cropped = image::imageops::crop_imm(&resized, (rw - bw) / 2, (rh - bh) / 2, bw, bh);

    let (bw, bh) = (bw as usize, bh as usize);
    let mut data = vec![0.0f32; 3 * bh * bw];
    for (x, y, pixel) in cropped.to_image().enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        for c in 0..3 {
            data[c * bh * bw + y * bw + x] = pixel.0[c] as f32 / 127.5 - 1.0;
        }
    }
    data
}

/// Compare two decodes **bit pattern by bit pattern**, not within a
/// tolerance: the claim under test is exact equality, and `1e-6` would pass
/// for a resize that quietly changed filter or crop origin.
fn assert_bit_identical(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length differs");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}: value {i} differs — got {g}, want {w}"
        );
    }
}

#[test]
fn decode_is_bit_identical_to_the_serial_reference() {
    use loractl_core::dataset::decode_image_for_bucket;

    let dir = temp_dataset_dir("dataset-bitident");

    // A gradient PNG in several shapes, plus a JPEG (a different decoder
    // path). The sizes are chosen so the cases below hit every branch of the
    // cover-resize: exact fit, pure downscale, pure upscale, and a crop that
    // lands on an ODD overflow in x and in y separately (so an off-by-one in
    // the `(rw - bw) / 2` origin cannot hide behind symmetry).
    write_png(&dir, "square.png", 64, 64);
    write_png(&dir, "wide.png", 200, 71);
    write_png(&dir, "tall.png", 71, 200);
    write_png(&dir, "small.png", 21, 17);
    let jpg = image::RgbImage::from_fn(97, 63, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x * y) % 256) as u8])
    });
    jpg.save_with_format(dir.join("photo.JPG"), image::ImageFormat::Jpeg)
        .expect("write test jpeg");

    // Every image against every bucket shape — square, landscape, portrait —
    // which is what makes an H/W transposition in the new plane-wise write
    // impossible to miss.
    let buckets = [b(64, 64), b(80, 48), b(48, 80), b(16, 16)];
    for name in [
        "square.png",
        "wide.png",
        "tall.png",
        "small.png",
        "photo.JPG",
    ] {
        for bucket in buckets {
            let path = dir.join(name);
            let want = decode_serial_reference(&path, bucket);
            let got = decode_image_for_bucket(&path, bucket).expect("decode");
            assert_bit_identical(&got, &want, &format!("{name} into {bucket:?}"));
        }
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_tensor_loader_still_matches_the_serial_reference() {
    use loractl_core::dataset::load_image_for_bucket;

    // `decode_image_for_bucket` is the host-side seam; this pins that the
    // tensor wrapper over it is still the same values in the same
    // `[1, 3, h, w]` layout — i.e. that the split did not move the transpose
    // into the tensor constructor.
    let dir = temp_dataset_dir("dataset-bitident-tensor");
    write_png(&dir, "wide.png", 200, 71);
    let bucket = b(80, 48);
    let device: burn::tensor::Device<B> = Default::default();

    let t = load_image_for_bucket::<B>(&dir.join("wide.png"), bucket, &device).unwrap();
    assert_eq!(t.dims(), [1, 3, 48, 80]);
    assert_bit_identical(
        &flat(&t),
        &decode_serial_reference(&dir.join("wide.png"), bucket),
        "load_image_for_bucket",
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn parallel_decode_feeds_the_encoder_in_scan_order() {
    // The design constraint of #178: decode is parallel, **encode is not**.
    // The encoders are `FnMut` and GPU-bound, and burn device tensors are not
    // safely shareable across threads — so the pre-pass must reorder nothing
    // that reaches them.
    //
    // Each image is a distinct flat colour, so the first channel value of the
    // tensor the encoder receives identifies exactly which file produced it.
    // Enough images (12) to span several decode windows on any core count.
    let dir = temp_dataset_dir("dataset-decode-order");
    let count = 12u32;
    for i in 0..count {
        image::RgbImage::from_pixel(64, 64, image::Rgb([(i * 20) as u8, 0, 0]))
            .save(dir.join(format!("{i:02}.png")))
            .unwrap();
    }

    let config = config_for(&dir);
    let device = Default::default();
    let seen = std::cell::RefCell::new(Vec::new());
    let names = std::cell::RefCell::new(Vec::new());

    prepare_dataset::<B>(
        &config,
        "order-v1",
        &device,
        |x| {
            let first = flat(&x)[0];
            // Invert the encoder's own normalization to recover the red byte.
            seen.borrow_mut()
                .push(((first + 1.0) * 127.5).round() as u32);
            mock_encode_image(x)
        },
        mock_encode_caption,
        |p| names.borrow_mut().push(p.name.to_string()),
    )
    .expect("prepare");

    let expected: Vec<u32> = (0..count).map(|i| i * 20).collect();
    assert_eq!(
        seen.into_inner(),
        expected,
        "the image encoder must see the scan order, whatever order the decodes finished in"
    );
    let expected_names: Vec<String> = (0..count).map(|i| format!("{i:02}.png")).collect();
    assert_eq!(
        names.into_inner(),
        expected_names,
        "progress must still be reported once per entry, in scan order"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Truncate a PNG in place so its **header still parses** (`image_dimensions`,
/// which is all `scan_dataset` needs, reads only IHDR) while its **pixel data
/// does not** — turning any decode into a loud failure.
///
/// The cut point is searched rather than hard-coded at the IDAT offset,
/// because exactly how far past IHDR the reader must get before it will report
/// dimensions is a decoder implementation detail, not a format guarantee. The
/// two conditions are asserted on the way out, so a future `image` release
/// that makes this impossible fails here with a readable message instead of
/// making the test that uses it vacuous.
fn truncate_to_header_only(path: &std::path::Path) {
    let bytes = std::fs::read(path).unwrap();
    let idat = bytes
        .windows(4)
        .position(|w| w == b"IDAT")
        .expect("png has an IDAT chunk");
    for keep in idat..bytes.len() {
        std::fs::write(path, &bytes[..keep]).unwrap();
        if image::image_dimensions(path).is_ok() && image::open(path).is_err() {
            return;
        }
    }
    panic!(
        "no prefix of {} reads as header-only-but-undecodable — this test has no teeth",
        path.display()
    );
}

#[test]
fn a_warm_cache_never_decodes_the_image() {
    // The warm-epoch guarantee one layer below the encoders: on a hit the
    // pipeline must not decode the image either — not to "validate" it, and
    // not because a parallel pre-pass decodes eagerly before checking the
    // cache. Truncating each PNG just before its first `IDAT` chunk leaves
    // `image::image_dimensions` (which `scan_dataset` needs, and which reads
    // only IHDR) working while `image::open` fails, so any decode at all is a
    // loud failure rather than a silent cost.
    let dir = temp_dataset_dir("dataset-nodecode");
    write_png(&dir, "a.png", 64, 64);
    write_png(&dir, "b.png", 100, 75);
    let config = config_for(&dir);
    let device = Default::default();

    prepare_dataset::<B>(
        &config,
        "nodecode-v1",
        &device,
        mock_encode_image,
        mock_encode_caption,
        |_| {},
    )
    .expect("cold pass");

    for name in ["a.png", "b.png"] {
        truncate_to_header_only(&dir.join(name));
    }

    prepare_dataset::<B>(
        &config,
        "nodecode-v1",
        &device,
        |_| panic!("image encoder must not run on a warm cache"),
        |_| panic!("caption encoder must not run on a warm cache"),
        |_| {},
    )
    .expect("a warm pass must not decode the images");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_decode_window_size_changes_nothing_observable() {
    // The determinism half of #178's acceptance: parallel iteration must not
    // reorder anything that feeds the cache keys or the encoders. The window
    // is `rayon::current_num_threads()`, so a machine's core count is a free
    // variable — and this pins that it is not an *observable* one, by running
    // the identical dataset under a 1-thread pool (which collapses the window
    // to exactly the pre-#178 serial interleaving) and a 7-thread pool, then
    // comparing every cached byte and every encoder call.
    //
    // 7 is deliberately coprime with 12 so the windows straddle entries
    // rather than tiling them evenly.
    fn run(tag: &str, threads: usize) -> (Vec<String>, Vec<Vec<u8>>, Vec<u32>, Vec<String>) {
        let dir = temp_dataset_dir(tag);
        for i in 0..12u32 {
            image::RgbImage::from_fn(64, 64, |x, y| {
                image::Rgb([(i * 20) as u8, (x % 256) as u8, (y % 256) as u8])
            })
            .save(dir.join(format!("{i:02}.png")))
            .unwrap();
            std::fs::write(dir.join(format!("{i:02}.txt")), format!("caption {i}")).unwrap();
        }
        let config = config_for(&dir);
        // `Mutex`, not `Cell`: `ThreadPool::install` needs a `Send` closure.
        // It is never contended — `prepare_dataset` calls the encoders and the
        // progress sink from the calling thread only, which is exactly the
        // property under test.
        let seen = std::sync::Mutex::new(Vec::new());
        let names = std::sync::Mutex::new(Vec::new());

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            prepare_dataset::<B>(
                &config,
                "window-v1",
                &Default::default(),
                |x| {
                    let first = flat(&x)[0];
                    seen.lock()
                        .unwrap()
                        .push(((first + 1.0) * 127.5).round() as u32);
                    mock_encode_image(x)
                },
                mock_encode_caption,
                |p| names.lock().unwrap().push(p.name.to_string()),
            )
            .expect("prepare");
        });

        let cache_files = cache_names(&dir);
        let bytes: Vec<Vec<u8>> = cache_files
            .iter()
            .map(|n| std::fs::read(dir.join(".loractl-cache").join(n)).unwrap())
            .collect();
        std::fs::remove_dir_all(&dir).ok();
        (
            cache_files,
            bytes,
            seen.into_inner().unwrap(),
            names.into_inner().unwrap(),
        )
    }

    let (serial_names, serial_bytes, serial_seen, serial_progress) = run("dataset-w1", 1);
    let (par_names, par_bytes, par_seen, par_progress) = run("dataset-w7", 7);

    assert_eq!(
        serial_names, par_names,
        "cache file names must not depend on the window size"
    );
    assert_eq!(
        serial_bytes, par_bytes,
        "cached bytes must not depend on the window size"
    );
    assert_eq!(
        serial_seen, par_seen,
        "the image encoder must see the same order regardless of the window size"
    );
    assert_eq!(
        serial_progress, par_progress,
        "progress must be reported in the same order regardless of the window size"
    );
    // Anti-vacuity: the run really did encode 12 distinct images in scan order.
    assert_eq!(par_seen, (0..12u32).map(|i| i * 20).collect::<Vec<_>>());
}
