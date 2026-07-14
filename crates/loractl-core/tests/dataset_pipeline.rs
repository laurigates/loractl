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

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use loractl_core::config::DatasetConfig;
use loractl_core::dataset::{
    BUCKET_ALIGN, Bucket, assign_bucket, generate_buckets, prepare_dataset,
};
use std::cell::Cell;
use std::path::PathBuf;

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
fn temp_dataset_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let dir = std::env::temp_dir().join(format!("loractl-{tag}-{}-{nanos}", std::process::id()));
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
    };
    let device = Default::default();

    // --- Cold pass: every encoder runs exactly once per example. ---
    let img_calls = Cell::new(0usize);
    let cap_calls = Cell::new(0usize);
    let captions_seen = std::cell::RefCell::new(Vec::<String>::new());
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
    )
    .expect("cold prepare");

    assert_eq!(img_calls.get(), 4, "one image encode per example");
    assert_eq!(cap_calls.get(), 4, "one caption encode per example");
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
    assert_eq!(prepared.items[0].latent.dims(), [1, 3, 8, 8]);
    assert_eq!(prepared.items[1].latent.dims(), [1, 3, 8, 8]);
    assert_eq!(
        prepared.items[2].latent.dims(),
        [1, 3, 6, 10],
        "80×48 bucket → f8 latent [1, 3, 48/8, 80/8]"
    );
    assert_eq!(prepared.items[3].latent.dims(), [1, 3, 8, 8]);

    // Batching groups the square bucket (3 examples → a 2-chunk and a
    // 1-chunk) and never mixes buckets.
    let batches = prepared.batches(2);
    assert_eq!(batches.len(), 3, "square 2+1, non-square 1");
    let sizes: Vec<usize> = batches.iter().map(|b| b.latents.dims()[0]).collect();
    assert_eq!(sizes.iter().sum::<usize>(), 4);
    for batch in &batches {
        let b = batch.latents.dims()[0];
        assert_eq!(batch.conditioning.dims()[0], b);
        assert_eq!(batch.mask.dims()[0], b);
    }
    // Row pairing: the square 2-batch's rows are items 0 and 1 IN ORDER,
    // latents and conditioning aligned (a batch that shuffled one but not
    // the other would train captions against the wrong images).
    let two = batches
        .iter()
        .find(|b| b.latents.dims()[0] == 2)
        .expect("a 2-batch exists");
    assert_eq!(
        flat(&two.latents.clone().narrow(0, 0, 1)),
        flat(&prepared.items[0].latent)
    );
    assert_eq!(
        flat(&two.latents.clone().narrow(0, 1, 1)),
        flat(&prepared.items[1].latent)
    );
    // fill = caption length: row 0 is "a red fox" (9), row 1 is "" (0).
    let cond_row0 = flat(&two.conditioning.clone().narrow(0, 0, 1));
    let cond_row1 = flat(&two.conditioning.clone().narrow(0, 1, 1));
    assert!(
        cond_row0.iter().all(|&v| v == 9.0),
        "row 0 pairs with 'a red fox'"
    );
    assert!(cond_row1.iter().all(|&v| v == 0.0), "row 1 pairs with ''");

    let cold_latents: Vec<Vec<f32>> = prepared.items.iter().map(|i| flat(&i.latent)).collect();
    let cold_conds: Vec<Vec<f32>> = prepared
        .items
        .iter()
        .map(|i| flat(&i.conditioning))
        .collect();
    let cold_masks: Vec<Vec<i64>> = prepared.items.iter().map(|i| flat_mask(&i.mask)).collect();
    // The mock mask is non-trivial, so the f32-store → int-reload round trip
    // below is actually exercised.
    assert_eq!(cold_masks[0], vec![1, 1, 1, 0]);

    // --- Warm pass: the cache serves everything; encoders must NOT run. ---
    let warm = prepare_dataset::<B>(
        &config,
        "mock-v1",
        &device,
        |_| panic!("image encoder must not run on a warm cache"),
        |_| panic!("caption encoder must not run on a warm cache"),
    )
    .expect("warm prepare");
    for (i, item) in warm.items.iter().enumerate() {
        assert_eq!(
            flat(&item.latent),
            cold_latents[i],
            "cached latent must be bit-exact"
        );
        assert_eq!(
            flat(&item.conditioning),
            cold_conds[i],
            "cached conditioning must be bit-exact"
        );
        assert_eq!(
            flat_mask(&item.mask),
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
    };
    let device = Default::default();

    prepare_dataset::<B>(&config, "mock-v1", &device, mock_encode_image, |c| {
        mock_encode_caption(c)
    })
    .expect("cold prepare");

    // Garbage every cache file, then re-prepare: the pipeline must surface a
    // parse error, not panic and not silently re-encode.
    let cache_dir = dir.join(".loractl-cache");
    for entry in std::fs::read_dir(&cache_dir).unwrap() {
        std::fs::write(entry.unwrap().path(), b"not a safetensors file").unwrap();
    }
    let result = prepare_dataset::<B>(&config, "mock-v1", &device, mock_encode_image, |c| {
        mock_encode_caption(c)
    });
    let err = format!("{:#}", result.err().expect("corrupted cache must error"));
    assert!(
        err.contains("parsing cache file"),
        "error should localize the bad cache file: {err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn empty_dataset_folder_fails_fast() {
    let dir = temp_dataset_dir("dataset-empty");
    let config = DatasetConfig {
        path: dir.clone(),
        resolution: RESOLUTION,
    };
    let device = Default::default();
    let result = prepare_dataset::<B>(&config, "mock-v1", &device, Ok, mock_encode_caption);
    assert!(result.is_err(), "an imageless dataset dir must error");
    std::fs::remove_dir_all(&dir).ok();
}
