//! The image dataset pipeline — aspect-ratio bucketing + latent/conditioning
//! caching (M12, #23).
//!
//! Implements the kohya/ai-toolkit folder convention the roadmap targets: a
//! directory of images with same-named `.txt` caption files. Images are
//! grouped into **aspect-ratio buckets** (each dimension a multiple of
//! [`BUCKET_ALIGN`] = 16 — the Krea 2 constraint `ae.compression · patch`),
//! resized cover-style + center-cropped to their bucket, VAE-encoded once,
//! and the latents cached to disk; captions are conditioning-encoded once and
//! cached the same way. Caching is what makes per-step cost tractable at 12B:
//! after the first pass, training never touches the image decoder, the VAE,
//! or the text encoder again.
//!
//! ## Decoupled from the concrete models
//!
//! [`prepare_dataset`] takes the two encoders as **closures** rather than
//! depending on [`QwenVae`](crate::QwenVae)/[`Qwen3VlConditioner`](crate::Qwen3VlConditioner)
//! directly: the pipeline's job is files → buckets → cache → batches, and the
//! encode step is whatever the trainer wires in (M14 passes the real frozen
//! models; the offline tests pass mocks). This keeps the pipeline fully
//! testable without checkpoints and keeps model choices out of the data
//! layer.
//!
//! ## Cache layout
//!
//! Under `<dataset>/.loractl-cache/`, keyed by the image file name (latents)
//! or stem (conditioning — captions are stem-keyed by convention, so images
//! sharing a stem share a caption), the bucket shape, and a caller-supplied
//! **fingerprint** (encoder identity — e.g. `"qwen-vae-f8x16+krea2-4b-ml512"`);
//! change the fingerprint — by *any* character — and the cache misses rather
//! than serving stale tensors from a different encoder setup (the filename
//! carries a sanitized prefix plus an FNV-1a hash of the raw string, so
//! sanitization cannot alias two fingerprints):
//!
//! ```text
//! {file_name}.{w}x{h}.{fingerprint}.latent.safetensors  "latent"        [1, z, h', w']
//! {stem}.{fingerprint}.cond.safetensors                 "conditioning"  [1, s, n, d]
//!                                                       "mask"          [1, s] (f32 0/1)
//! ```
//!
//! Cache keys deliberately do **not** hash file contents: an image or
//! caption edited *in place* under the same name serves the stale cache
//! until `.loractl-cache/` is deleted. Content-hash invalidation can come
//! later if it earns its cost; delete the cache dir after editing a dataset
//! in place.
//!
//! Like the rest of `loractl-core`, this module emits no output and imports
//! no CLI.

use crate::export::{OwnedF32Tensor, to_owned_f32};
use anyhow::{Context, Result, bail};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use std::path::{Path, PathBuf};

/// Every bucket dimension is a multiple of this: Krea 2's
/// `ae.compression (8) · patch (2)` — the resolution granularity the latent
/// patch grid supports.
pub const BUCKET_ALIGN: u32 = 16;

/// The aspect ratios buckets are generated for (width : height).
const ASPECTS: [(u32, u32); 7] = [(1, 1), (4, 3), (3, 4), (3, 2), (2, 3), (16, 9), (9, 16)];

/// One resolution bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bucket {
    /// Pixel width (multiple of [`BUCKET_ALIGN`]).
    pub width: u32,
    /// Pixel height (multiple of [`BUCKET_ALIGN`]).
    pub height: u32,
}

impl Bucket {
    fn aspect(&self) -> f64 {
        self.width as f64 / self.height as f64
    }
}

/// Generate the bucket set for a target `resolution`: for each aspect ratio
/// in [`ASPECTS`], the [`BUCKET_ALIGN`]-aligned box with roughly
/// `resolution²` pixels. Deduplicated; the square `resolution × resolution`
/// bucket is always present. Errors on an unaligned `resolution` —
/// this value arrives straight from user YAML, so misconfiguration must
/// surface as an error, not a panic.
pub fn generate_buckets(resolution: u32) -> Result<Vec<Bucket>> {
    if !resolution.is_multiple_of(BUCKET_ALIGN) {
        bail!(
            "dataset.resolution = {resolution} must be a multiple of {BUCKET_ALIGN} \
             (Krea 2's compression × patch grid)"
        );
    }
    let align = |v: f64| -> u32 {
        let stepped = (v / BUCKET_ALIGN as f64).round().max(1.0) as u32;
        stepped * BUCKET_ALIGN
    };
    let mut buckets = Vec::new();
    for (aw, ah) in ASPECTS {
        let aspect = aw as f64 / ah as f64;
        let w = align(resolution as f64 * aspect.sqrt());
        let h = align(resolution as f64 / aspect.sqrt());
        let bucket = Bucket {
            width: w,
            height: h,
        };
        if !buckets.contains(&bucket) {
            buckets.push(bucket);
        }
    }
    Ok(buckets)
}

/// The bucket whose aspect ratio is nearest (in log space) to
/// `width × height`'s.
pub fn assign_bucket(buckets: &[Bucket], width: u32, height: u32) -> usize {
    let aspect = (width as f64 / height as f64).ln();
    buckets
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (a.aspect().ln() - aspect).abs();
            let db = (b.aspect().ln() - aspect).abs();
            da.partial_cmp(&db).expect("finite aspects")
        })
        .expect("bucket set is non-empty")
        .0
}

/// One dataset entry: an image, its caption, and its assigned bucket.
#[derive(Debug, Clone)]
pub struct DatasetEntry {
    /// The image file.
    pub image_path: PathBuf,
    /// The caption (contents of the same-stem `.txt`, trimmed; empty when no
    /// caption file exists — an unconditional example).
    pub caption: String,
    /// Index into the bucket set.
    pub bucket: usize,
}

/// Scan a kohya-style dataset folder: every `.png`/`.jpg`/`.jpeg` image (with
/// an optional same-stem `.txt` caption), each assigned to its nearest
/// bucket. Sorted by filename for determinism. Errors when the folder holds
/// no images (fail fast — an empty dataset is a configuration mistake).
pub fn scan_dataset(dir: &Path, buckets: &[Bucket]) -> Result<Vec<DatasetEntry>> {
    let mut entries = Vec::new();
    let read = std::fs::read_dir(dir)
        .with_context(|| format!("reading dataset directory {}", dir.display()))?;
    for entry in read {
        let path = entry?.path();
        let is_image = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"));
        if !is_image {
            continue;
        }
        let (width, height) = image::image_dimensions(&path)
            .with_context(|| format!("reading dimensions of {}", path.display()))?;
        let caption_path = path.with_extension("txt");
        let caption = match std::fs::read_to_string(&caption_path) {
            Ok(text) => text.trim().to_string(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => {
                return Err(e)
                    .with_context(|| format!("reading caption {}", caption_path.display()));
            }
        };
        entries.push(DatasetEntry {
            image_path: path,
            caption,
            bucket: assign_bucket(buckets, width, height),
        });
    }
    if entries.is_empty() {
        bail!("no .png/.jpg/.jpeg images found in {}", dir.display());
    }
    entries.sort_by(|a, b| a.image_path.cmp(&b.image_path));
    Ok(entries)
}

/// Decode an image, resize it cover-style to its bucket (preserving aspect,
/// so the shorter relative side fits exactly), center-crop the overflow, and
/// return it as a `[1, 3, height, width]` tensor scaled to `[-1, 1]` (the
/// VAE's input range).
pub fn load_image_for_bucket<B: Backend>(
    path: &Path,
    bucket: Bucket,
    device: &B::Device,
) -> Result<Tensor<B, 4>> {
    let img = image::open(path)
        .with_context(|| format!("decoding {}", path.display()))?
        .to_rgb8();
    let (w, h) = (img.width(), img.height());
    let (bw, bh) = (bucket.width, bucket.height);

    // Cover: scale so both dimensions reach the bucket, then center-crop.
    let scale = f64::max(bw as f64 / w as f64, bh as f64 / h as f64);
    let rw = (w as f64 * scale).ceil() as u32;
    let rh = (h as f64 * scale).ceil() as u32;
    let resized = image::imageops::resize(&img, rw, rh, image::imageops::FilterType::Lanczos3);
    let cropped = image::imageops::crop_imm(&resized, (rw - bw) / 2, (rh - bh) / 2, bw, bh);

    // HWC u8 → CHW f32 in [-1, 1].
    let (bw, bh) = (bw as usize, bh as usize);
    let mut data = vec![0.0f32; 3 * bh * bw];
    for (x, y, pixel) in cropped.to_image().enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        for c in 0..3 {
            data[c * bh * bw + y * bw + x] = pixel.0[c] as f32 / 127.5 - 1.0;
        }
    }
    Ok(Tensor::from_data(
        TensorData::new(data, [1, 3, bh, bw]),
        device,
    ))
}

/// A cache-file tensor as raw values + shape.
type CachedTensor = (Vec<f32>, Vec<usize>);

/// FNV-1a (64-bit) over a string — the injective-enough suffix that keeps
/// distinct fingerprints from colliding after filename sanitization.
fn fnv1a64(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// The on-disk latent/conditioning cache (see the module docs for layout).
struct DatasetCache {
    dir: PathBuf,
    fingerprint: String,
}

impl DatasetCache {
    fn new(dataset_dir: &Path, fingerprint: &str) -> Result<Self> {
        let dir = dataset_dir.join(".loractl-cache");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating cache dir {}", dir.display()))?;
        // Filename-safe AND injective: a readable sanitized prefix plus an
        // FNV-1a hash of the RAW fingerprint, so fingerprints differing only
        // in sanitized-away characters ("qwen_vae.f8" vs "qwen-vae+f8", both
        // sanitizing to "qwen-vae-f8") cannot serve each other's cache.
        let sanitized: String = fingerprint
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' {
                    c
                } else {
                    '-'
                }
            })
            .collect();
        let fingerprint = format!("{sanitized}-{:016x}", fnv1a64(fingerprint));
        Ok(Self { dir, fingerprint })
    }

    fn latent_path(&self, file_name: &str, bucket: Bucket) -> PathBuf {
        self.dir.join(format!(
            "{file_name}.{}x{}.{}.latent.safetensors",
            bucket.width, bucket.height, self.fingerprint
        ))
    }

    fn cond_path(&self, stem: &str) -> PathBuf {
        self.dir
            .join(format!("{stem}.{}.cond.safetensors", self.fingerprint))
    }

    /// Read named f32 tensors from a cache file; `None` on a miss.
    fn read(&self, path: &Path, names: &[&str]) -> Result<Option<Vec<CachedTensor>>> {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parsing cache file {}", path.display()))?;
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            let view = st
                .tensor(name)
                .with_context(|| format!("cache file {} lacks '{name}'", path.display()))?;
            if view.dtype() != safetensors::Dtype::F32 {
                bail!("cache tensor '{name}' in {} is not F32", path.display());
            }
            let values: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            out.push((values, view.shape().to_vec()));
        }
        Ok(Some(out))
    }

    fn write(&self, path: &Path, tensors: Vec<(&str, OwnedF32Tensor)>) -> Result<()> {
        let views: Vec<(&str, &OwnedF32Tensor)> = tensors.iter().map(|(k, t)| (*k, t)).collect();
        safetensors::serialize_to_file(views, None, path)
            .with_context(|| format!("writing cache file {}", path.display()))
    }
}

/// One prepared example: cached tensors plus its bucket, batchable with any
/// other example from the same bucket.
pub struct PreparedItem<B: Backend> {
    /// Normalized VAE latent `[1, z, h', w']`.
    pub latent: Tensor<B, 4>,
    /// Conditioning stack `[1, s, n, d]`.
    pub conditioning: Tensor<B, 4>,
    /// The conditioning key mask `[1, s]` (0/1).
    pub mask: Tensor<B, 2, Int>,
    /// Index into [`PreparedDataset::buckets`].
    pub bucket: usize,
}

/// The prepared (fully cached) dataset.
pub struct PreparedDataset<B: Backend> {
    /// One entry per training image, in scan order.
    pub items: Vec<PreparedItem<B>>,
    /// The bucket set the items reference.
    pub buckets: Vec<Bucket>,
}

/// One training batch: same-bucket examples concatenated on the batch dim.
pub struct DatasetBatch<B: Backend> {
    /// Latents `[b, z, h', w']`.
    pub latents: Tensor<B, 4>,
    /// Conditioning `[b, s, n, d]`.
    pub conditioning: Tensor<B, 4>,
    /// Conditioning mask `[b, s]`.
    pub mask: Tensor<B, 2, Int>,
}

impl<B: Backend> PreparedDataset<B> {
    /// Group items per bucket and chunk into batches of at most
    /// `batch_size`. Batches never mix buckets (shapes differ across
    /// buckets); the final chunk of a bucket may be smaller.
    pub fn batches(&self, batch_size: usize) -> Vec<DatasetBatch<B>> {
        assert!(batch_size > 0, "batch_size must be positive");
        let mut batches = Vec::new();
        for bucket in 0..self.buckets.len() {
            let indices: Vec<usize> = (0..self.items.len())
                .filter(|&i| self.items[i].bucket == bucket)
                .collect();
            for chunk in indices.chunks(batch_size) {
                let latents = Tensor::cat(
                    chunk
                        .iter()
                        .map(|&i| self.items[i].latent.clone())
                        .collect(),
                    0,
                );
                let conditioning = Tensor::cat(
                    chunk
                        .iter()
                        .map(|&i| self.items[i].conditioning.clone())
                        .collect(),
                    0,
                );
                let mask = Tensor::cat(
                    chunk.iter().map(|&i| self.items[i].mask.clone()).collect(),
                    0,
                );
                batches.push(DatasetBatch {
                    latents,
                    conditioning,
                    mask,
                });
            }
        }
        batches
    }
}

/// Scan, bucket, encode-or-load-from-cache every example of the dataset at
/// `config.path` (see the module docs).
///
/// - `fingerprint` names the encoder setup for cache keying.
/// - `encode_image` maps a `[1, 3, h, w]` image in `[-1, 1]` to its latent
///   `[1, z, h', w']` (M14 wires [`QwenVae::encode`](crate::QwenVae::encode)).
/// - `encode_caption` maps a caption to its conditioning stack + mask (M14
///   wires [`Qwen3VlConditioner::encode_captions`](crate::Qwen3VlConditioner::encode_captions)).
///
/// Both closures run **once per example on a cache miss and never on a
/// hit** — after the first pass, epochs re-read pure tensor files.
pub fn prepare_dataset<B: Backend>(
    config: &crate::config::DatasetConfig,
    fingerprint: &str,
    device: &B::Device,
    mut encode_image: impl FnMut(Tensor<B, 4>) -> Result<Tensor<B, 4>>,
    mut encode_caption: impl FnMut(&str) -> Result<(Tensor<B, 4>, Tensor<B, 2, Int>)>,
) -> Result<PreparedDataset<B>> {
    let buckets = generate_buckets(config.resolution)?;
    let entries = scan_dataset(&config.path, &buckets)?;
    let cache = DatasetCache::new(&config.path, fingerprint)?;

    let mut items = Vec::with_capacity(entries.len());
    for entry in &entries {
        // The latent cache keys on the FULL file name: `a.png` and `a.jpg`
        // share a stem (and thus a caption — the kohya convention keys
        // captions by stem, so the conditioning cache below sharing is
        // correct) but are different pixels, so their latents must not
        // collide.
        let file_name = entry
            .image_path
            .file_name()
            .and_then(|s| s.to_str())
            .with_context(|| format!("non-UTF-8 image name {}", entry.image_path.display()))?;
        let stem = entry
            .image_path
            .file_stem()
            .and_then(|s| s.to_str())
            .with_context(|| format!("non-UTF-8 image name {}", entry.image_path.display()))?;
        let bucket = buckets[entry.bucket];

        // Latent: cache hit → rebuild the tensor; miss → decode/encode/store.
        let latent_path = cache.latent_path(file_name, bucket);
        let latent = match cache.read(&latent_path, &["latent"])? {
            Some(mut tensors) => {
                let (values, shape) = tensors.remove(0);
                if shape.len() != 4 {
                    bail!("cached latent {} is not rank-4", latent_path.display());
                }
                Tensor::from_data(TensorData::new(values, shape), device)
            }
            None => {
                let image = load_image_for_bucket::<B>(&entry.image_path, bucket, device)?;
                let latent = encode_image(image)
                    .with_context(|| format!("encoding {}", entry.image_path.display()))?;
                cache.write(&latent_path, vec![("latent", to_owned_f32(latent.clone()))])?;
                latent
            }
        };

        // Conditioning: same shape of hit/miss, two tensors per file. The
        // mask is stored as f32 0/1 (the cache is a single-dtype format
        // here) and converted back to Int on load.
        let cond_path = cache.cond_path(stem);
        let (conditioning, mask) = match cache.read(&cond_path, &["conditioning", "mask"])? {
            Some(mut tensors) => {
                let (mask_values, mask_shape) = tensors.remove(1);
                let (cond_values, cond_shape) = tensors.remove(0);
                if cond_shape.len() != 4 || mask_shape.len() != 2 {
                    bail!(
                        "cached conditioning {} has wrong ranks",
                        cond_path.display()
                    );
                }
                let conditioning =
                    Tensor::from_data(TensorData::new(cond_values, cond_shape), device);
                let mask =
                    Tensor::<B, 2>::from_data(TensorData::new(mask_values, mask_shape), device)
                        .int();
                (conditioning, mask)
            }
            None => {
                let (conditioning, mask) = encode_caption(&entry.caption)
                    .with_context(|| format!("encoding caption for {stem}"))?;
                cache.write(
                    &cond_path,
                    vec![
                        ("conditioning", to_owned_f32(conditioning.clone())),
                        ("mask", to_owned_f32(mask.clone().float())),
                    ],
                )?;
                (conditioning, mask)
            }
        };

        items.push(PreparedItem {
            latent,
            conditioning,
            mask,
            bucket: entry.bucket,
        });
    }

    Ok(PreparedDataset { items, buckets })
}
