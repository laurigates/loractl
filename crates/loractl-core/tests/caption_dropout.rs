//! Caption dropout (`dataset.caption_dropout_rate`, #219) — offline, ndarray.
//!
//! The knob's failure mode is *silence*: a rate that is parsed and ignored, or
//! applied once at encode time and frozen into the cache, still trains and
//! still converges. So every claim here is about observable behaviour, never
//! about a field existing:
//!
//! - **the substitution policy** ([`CaptionDropout`]) is a pure function of
//!   `(seed, global step, row)` — rate 0 never drops, rate 1 always drops, the
//!   empirical rate tracks the configured one, and the pattern varies across
//!   steps and seeds, which is exactly what a resumed run needs to reproduce
//!   and what a cache-baked decision could not do;
//! - **the cache** gains one shared, fingerprint-keyed empty-caption entry
//!   when (and only when) the rate is non-zero, and the per-stem entries stay
//!   caption-only;
//! - **batch assembly** swaps the conditioning of exactly the rows the policy
//!   names, leaves latents alone, and at rate 0 is bit-identical to
//!   [`PreparedDataset::load_batch`];
//! - **end to end** a non-zero rate changes the loss trajectory of the real
//!   diffusion trainer (the `tests/dropout.rs` shape, because `lora.dropout`
//!   once shipped dead), while the default stays deterministic.

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use loractl_core::config::{
    DatasetConfig, LoraConfig, ModelConfig, ModelVariant, OptimConfig, OutputConfig, TargetSpec,
    TaskKind,
};
use loractl_core::dataset::{
    CaptionDropout, ConditioningChoice, ConditioningPolicy, PreparedDataset, plan_dataset,
    prepare_dataset,
};
use loractl_core::{
    BurnTrainer, DiffusionTrainer, TrainConfig, TrainEvent, Trainer, read_metadata,
};
use std::cell::RefCell;
use std::path::{Path, PathBuf};

type B = NdArray;

const FINGERPRINT: &str = "mock-v1";

/// burn's backend RNG is process-global, so trainings must not interleave
/// (mirrors `tests/diffusion_trainer.rs`).
static TRAIN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "loractl-{tag}-{}-{nanos}-{seq}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------------------
// The policy
// ---------------------------------------------------------------------------

fn pattern(
    policy: &CaptionDropout,
    steps: std::ops::RangeInclusive<u64>,
    rows: usize,
) -> Vec<bool> {
    steps
        .flat_map(|step| (0..rows).map(move |row| (step, row)))
        .map(|(step, row)| policy.choose(step, row) == ConditioningChoice::Empty)
        .collect()
}

#[test]
fn rate_zero_never_drops_and_rate_one_always_drops() {
    let never = CaptionDropout::new(0.0, 7).unwrap();
    let always = CaptionDropout::new(1.0, 7).unwrap();
    assert!(
        pattern(&never, 1..=500, 4).iter().all(|&d| !d),
        "rate 0 must never drop"
    );
    assert!(
        pattern(&always, 1..=500, 4).iter().all(|&d| d),
        "rate 1 must always drop (the draw is in [0, 1), so `< 1` is certain)"
    );
}

#[test]
fn the_empirical_drop_rate_tracks_the_configured_one() {
    for rate in [0.1f32, 0.3, 0.5, 0.9] {
        let policy = CaptionDropout::new(rate, 42).unwrap();
        let draws = pattern(&policy, 1..=2_000, 4);
        let observed = draws.iter().filter(|&&d| d).count() as f64 / draws.len() as f64;
        // 8000 Bernoulli draws: sd <= 0.0056, so 0.03 is > 5 sigma.
        assert!(
            (observed - f64::from(rate)).abs() < 0.03,
            "configured {rate}, observed {observed}"
        );
    }
}

#[test]
fn the_pattern_is_a_pure_function_of_seed_step_and_row() {
    // Two independently built policies agree everywhere: there is no hidden
    // state for a resumed run to have lost.
    let a = CaptionDropout::new(0.4, 42).unwrap();
    let b = CaptionDropout::new(0.4, 42).unwrap();
    assert_eq!(pattern(&a, 1..=200, 3), pattern(&b, 1..=200, 3));
    // ...and call order is irrelevant: asking for step 150 first, cold, gives
    // the same answer as walking up to it.
    let walked = pattern(&a, 1..=150, 3);
    let cold = pattern(&b, 150..=150, 3);
    assert_eq!(&walked[149 * 3..], &cold[..]);
}

#[test]
fn the_pattern_varies_across_steps_rows_and_seeds() {
    let policy = CaptionDropout::new(0.5, 42).unwrap();
    // Per epoch: with one batch per epoch the step IS the epoch, so a pattern
    // that did not vary across steps would be a decision frozen per example.
    let row0: Vec<bool> = (1..=64)
        .map(|s| policy.choose(s, 0) == ConditioningChoice::Empty)
        .collect();
    assert!(row0.iter().any(|&d| d) && row0.iter().any(|&d| !d));
    // Rows within one step are independent draws.
    let step: Vec<bool> = (0..64)
        .map(|r| policy.choose(5, r) == ConditioningChoice::Empty)
        .collect();
    assert!(step.iter().any(|&d| d) && step.iter().any(|&d| !d));
    // A different seed is a different stream.
    let other = CaptionDropout::new(0.5, 43).unwrap();
    assert_ne!(pattern(&policy, 1..=64, 2), pattern(&other, 1..=64, 2));
}

#[test]
fn invalid_rates_are_rejected_everywhere() {
    for bad in [f32::NAN, -0.1, 1.0001, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(CaptionDropout::new(bad, 0).is_err(), "new({bad})");
        let config = DatasetConfig {
            caption_dropout_rate: bad,
            ..Default::default()
        };
        let err = config.validate().expect_err("validate must refuse");
        assert!(
            format!("{err:#}").contains("caption_dropout_rate"),
            "the error names the field: {err:#}"
        );
        assert!(CaptionDropout::from_config(&config, 0).is_err());
    }
    for good in [0.0f32, 0.25, 1.0] {
        let config = DatasetConfig {
            caption_dropout_rate: good,
            ..Default::default()
        };
        config.validate().expect("in range");
    }
    // Rate 0 builds no policy at all — the caller takes the plain path.
    assert!(
        CaptionDropout::from_config(&DatasetConfig::default(), 0)
            .unwrap()
            .is_none()
    );
}

#[test]
fn the_synthetic_trainer_refuses_a_knob_it_would_ignore() {
    let out = TempDir::new("dropout-synthetic");
    let config = TrainConfig {
        steps: 2,
        dataset: DatasetConfig {
            caption_dropout_rate: 0.5,
            ..Default::default()
        },
        output: OutputConfig {
            dir: out.0.clone(),
            ..OutputConfig::default()
        },
        ..Default::default()
    };
    let err = BurnTrainer
        .train(&config, &mut |_| {})
        .expect_err("a no-caption trainer must not silently ignore the rate");
    assert!(
        format!("{err:#}").contains("caption_dropout_rate"),
        "{err:#}"
    );

    let nan = TrainConfig {
        dataset: DatasetConfig {
            caption_dropout_rate: f32::NAN,
            ..Default::default()
        },
        ..config
    };
    assert!(
        BurnTrainer.train(&nan, &mut |_| {}).is_err(),
        "NaN rejected"
    );
}

// ---------------------------------------------------------------------------
// The cache and batch assembly (mock encoders)
// ---------------------------------------------------------------------------

/// A distinct, deterministic gradient PNG per `seed`, so two images have
/// different latents and a swapped latent is detectable.
fn write_png(dir: &Path, name: &str, seed: u32) {
    let img = image::RgbImage::from_fn(64, 64, |x, y| {
        image::Rgb([
            ((x * (seed + 1)) % 256) as u8,
            ((y + seed * 40) % 256) as u8,
            ((x + y) % 256) as u8,
        ])
    });
    img.save(dir.join(name)).expect("write test png");
}

/// Four captioned square images in one bucket (so a `batch_size: 4` plan is a
/// single batch of four rows).
fn fixture(tag: &str) -> TempDir {
    let dir = TempDir::new(tag);
    for (i, (name, caption)) in [
        ("a", "a red fox"),
        ("b", "green field"),
        ("c", "sand dunes at noon"),
        ("d", "sky"),
    ]
    .into_iter()
    .enumerate()
    {
        write_png(&dir.0, &format!("{name}.png"), i as u32);
        std::fs::write(dir.0.join(format!("{name}.txt")), caption).unwrap();
    }
    dir
}

fn config(dir: &Path, rate: f32) -> DatasetConfig {
    DatasetConfig {
        path: dir.to_path_buf(),
        resolution: 64,
        batch_size: 4,
        caption_dropout_rate: rate,
        ..Default::default()
    }
}

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

/// Conditioning filled with `caption.len() + 1` — so the empty caption is
/// `1.0` and no real caption is — and a mask whose number of ones also depends
/// on the caption, so a swapped mask is as visible as a swapped stack.
fn mock_encode_caption(caption: &str) -> anyhow::Result<(Tensor<B, 4>, Tensor<B, 2, Int>)> {
    let device = Default::default();
    let cond = Tensor::from_data(
        TensorData::new(vec![caption.len() as f32 + 1.0; 4 * 2 * 8], [1, 4, 2, 8]),
        &device,
    );
    let ones = 1 + caption.len().min(3);
    let mask: Vec<i64> = (0..4).map(|i| i64::from(i < ones)).collect();
    Ok((
        cond,
        Tensor::from_data(TensorData::new(mask, [1, 4]), &device),
    ))
}

fn prepare(config: &DatasetConfig) -> (PreparedDataset, Vec<String>) {
    let seen = RefCell::new(Vec::new());
    let prepared = prepare_dataset::<B>(
        config,
        FINGERPRINT,
        &Default::default(),
        mock_encode_image,
        |c| {
            seen.borrow_mut().push(c.to_string());
            mock_encode_caption(c)
        },
        |_| {},
    )
    .expect("prepare");
    (prepared, seen.into_inner())
}

fn cache_files(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir.join(".loractl-cache"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn flat(t: &Tensor<B, 4>) -> Vec<f32> {
    t.clone()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap()
}

fn flat_mask(t: &Tensor<B, 2, Int>) -> Vec<i64> {
    t.clone()
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .unwrap()
}

/// The per-row conditioning fill of a batch (`[b, 4, 2, 8]`).
fn row_fills(conditioning: &Tensor<B, 4>) -> Vec<f32> {
    flat(conditioning).chunks(4 * 2 * 8).map(|c| c[0]).collect()
}

#[test]
fn rate_zero_encodes_no_empty_caption_and_writes_no_extra_file() {
    let dir = fixture("dropout-off");
    let (prepared, seen) = prepare(&config(&dir.0, 0.0));
    assert_eq!(
        seen.len(),
        4,
        "one encode per captioned example, nothing more"
    );
    assert!(
        !seen.iter().any(String::is_empty),
        "no empty-caption encode at rate 0: {seen:?}"
    );
    assert!(prepared.empty_conditioning.is_none());
    assert!(
        !cache_files(&dir.0).iter().any(|n| n.contains(".empty.")),
        "no empty-caption cache file at rate 0: {:?}",
        cache_files(&dir.0)
    );
}

#[test]
fn a_positive_rate_caches_one_shared_empty_entry_and_leaves_per_stem_entries_caption_only() {
    let dir = fixture("dropout-cache");
    let (prepared, seen) = prepare(&config(&dir.0, 0.5));
    assert_eq!(
        seen.iter().filter(|c| c.is_empty()).count(),
        1,
        "\"\" is encoded exactly once, however many examples there are: {seen:?}"
    );
    assert_eq!(seen.len(), 5);
    let empties: Vec<String> = cache_files(&dir.0)
        .into_iter()
        .filter(|n| n.contains(".empty."))
        .collect();
    assert_eq!(empties.len(), 1, "one shared entry, not one per example");
    assert!(empties[0].ends_with(".cond.safetensors"));

    // The per-stem entries still hold their own caption — dropout is NOT
    // baked into them. Row 0 (`a.png`, "a red fox") reads back as its caption.
    let item = prepared.load_item::<B>(0, &Default::default()).unwrap();
    assert_eq!(
        row_fills(&item.conditioning),
        vec!["a red fox".len() as f32 + 1.0]
    );

    // A fully warm rerun encodes nothing — the empty entry is a cache hit too.
    let warm = prepare_dataset::<B>(
        &config(&dir.0, 0.5),
        FINGERPRINT,
        &Default::default(),
        |_| panic!("warm cache must not re-encode an image"),
        |c| panic!("warm cache must not re-encode {c:?}"),
        |_| {},
    )
    .expect("warm prepare");
    assert!(warm.empty_conditioning.is_some());
}

#[test]
fn planning_a_positive_rate_over_a_cache_without_the_empty_entry_is_a_loud_error() {
    let dir = fixture("dropout-miss");
    prepare(&config(&dir.0, 0.0));
    let err = plan_dataset(&config(&dir.0, 0.5), FINGERPRINT, |_| {})
        .expect_err("the training phase cannot encode");
    assert!(
        format!("{err:#}").contains("empty-caption conditioning cache miss"),
        "{err:#}"
    );
    // ...whereas the same cache plans fine at rate 0.
    plan_dataset(&config(&dir.0, 0.0), FINGERPRINT, |_| {}).expect("rate 0 needs no empty entry");
}

#[test]
fn an_empty_entry_that_cannot_batch_with_the_items_is_refused_at_plan_time() {
    let dir = fixture("dropout-shape");
    let err = prepare_dataset::<B>(
        &config(&dir.0, 0.5),
        FINGERPRINT,
        &Default::default(),
        mock_encode_image,
        |c| {
            // The empty caption comes back a different length: a mixed batch
            // would panic in `Tensor::cat` at whichever step first mixes them.
            if c.is_empty() {
                let device = Default::default();
                Ok((
                    Tensor::from_data(
                        TensorData::new(vec![1.0f32; 2 * 2 * 8], [1, 2, 2, 8]),
                        &device,
                    ),
                    Tensor::from_data(TensorData::new(vec![1i64, 0], [1, 2]), &device),
                ))
            } else {
                mock_encode_caption(c)
            }
        },
        |_| {},
    )
    .expect_err("a mismatched empty entry must not be accepted");
    assert!(format!("{err:#}").contains("would mix"), "{err:#}");
}

#[test]
fn rate_one_makes_every_row_the_empty_conditioning_and_leaves_latents_alone() {
    let dir = fixture("dropout-all");
    let (prepared, _) = prepare(&config(&dir.0, 1.0));
    let plans = prepared.batches(4);
    assert_eq!(plans.len(), 1);
    let device = Default::default();

    let plain = prepared.load_batch::<B>(&plans[0], &device).unwrap();
    let policy = CaptionDropout::new(1.0, 42).unwrap();
    let dropped = prepared
        .load_batch_with::<B>(&plans[0], 3, &policy, &device)
        .unwrap();

    assert_eq!(
        row_fills(&dropped.conditioning),
        vec![1.0; 4],
        "all rows are the empty stack"
    );
    let (_, empty_mask) = mock_encode_caption("").unwrap();
    let empty_mask = flat_mask(&empty_mask);
    for row in flat_mask(&dropped.mask).chunks(4) {
        assert_eq!(
            row,
            &empty_mask[..],
            "the mask travels with the conditioning"
        );
    }
    assert_eq!(
        flat(&dropped.latents),
        flat(&plain.latents),
        "dropout touches conditioning only"
    );
    // The un-dropped batch is untouched by any of this.
    assert_eq!(
        row_fills(&plain.conditioning),
        vec![10.0, 12.0, 19.0, 4.0],
        "captions \"a red fox\", \"green field\", \"sand dunes at noon\", \"sky\" (+1)"
    );
}

#[test]
fn rate_zero_policy_is_bit_identical_to_load_batch() {
    let dir = fixture("dropout-identity");
    let (prepared, _) = prepare(&config(&dir.0, 0.0));
    let plans = prepared.batches(4);
    let device = Default::default();
    let plain = prepared.load_batch::<B>(&plans[0], &device).unwrap();
    let policy = CaptionDropout::new(0.0, 42).unwrap();
    // Even routed through the policy path (which the trainer skips at rate 0),
    // nothing changes, and no empty entry is needed.
    let via = prepared
        .load_batch_with::<B>(&plans[0], 9, &policy, &device)
        .unwrap();
    assert_eq!(flat(&via.latents), flat(&plain.latents));
    assert_eq!(flat(&via.conditioning), flat(&plain.conditioning));
    assert_eq!(flat_mask(&via.mask), flat_mask(&plain.mask));
}

#[test]
fn asking_for_an_empty_row_without_a_planned_empty_entry_is_an_error() {
    let dir = fixture("dropout-unplanned");
    let (prepared, _) = prepare(&config(&dir.0, 0.0));
    let plans = prepared.batches(4);
    let policy = CaptionDropout::new(1.0, 1).unwrap();
    // `DatasetBatch` is deliberately not `Debug`, so no `expect_err`.
    let err = match prepared.load_batch_with::<B>(&plans[0], 1, &policy, &Default::default()) {
        Ok(_) => panic!("no empty entry was planned"),
        Err(e) => e,
    };
    assert!(
        format!("{err:#}").contains("planned without one"),
        "{err:#}"
    );
}

#[test]
fn a_mid_rate_batch_substitutes_exactly_the_rows_the_policy_names() {
    let dir = fixture("dropout-mixed");
    let (prepared, _) = prepare(&config(&dir.0, 0.5));
    let plans = prepared.batches(4);
    let device = Default::default();
    let policy = CaptionDropout::new(0.5, 42).unwrap();
    let own = [10.0f32, 12.0, 19.0, 4.0];

    let (mut dropped_rows, mut kept_rows) = (0, 0);
    for step in 1..=40u64 {
        let batch = prepared
            .load_batch_with::<B>(&plans[0], step, &policy, &device)
            .unwrap();
        let fills = row_fills(&batch.conditioning);
        for (row, fill) in fills.iter().enumerate() {
            match policy.choose(step, row) {
                ConditioningChoice::Empty => {
                    assert_eq!(*fill, 1.0, "step {step} row {row} should be empty");
                    dropped_rows += 1;
                }
                _ => {
                    assert_eq!(*fill, own[row], "step {step} row {row} keeps its caption");
                    kept_rows += 1;
                }
            }
        }
    }
    assert!(
        dropped_rows > 0 && kept_rows > 0,
        "{dropped_rows}/{kept_rows}"
    );
}

#[test]
fn a_resumed_run_reproduces_the_original_drop_pattern() {
    // Run one: plan, then walk steps 1..=12 recording the conditioning each
    // step saw. Run two is "the resume": a freshly planned dataset and a
    // freshly built policy (nothing carried over from run one) that starts at
    // step 7, as `plan.start_step()` does. Its steps must equal run one's.
    let dir = fixture("dropout-resume");
    let (first, _) = prepare(&config(&dir.0, 0.5));
    let device = Default::default();
    let policy_one = CaptionDropout::new(0.5, 42).unwrap();
    let plans = first.batches(4);
    let original: Vec<Vec<f32>> = (1..=12u64)
        .map(|step| {
            row_fills(
                &first
                    .load_batch_with::<B>(&plans[0], step, &policy_one, &device)
                    .unwrap()
                    .conditioning,
            )
        })
        .collect();

    let second = plan_dataset(&config(&dir.0, 0.5), FINGERPRINT, |_| {}).unwrap();
    let policy_two = CaptionDropout::new(0.5, 42).unwrap();
    let plans = second.batches(4);
    for step in 7..=12u64 {
        let resumed = row_fills(
            &second
                .load_batch_with::<B>(&plans[0], step, &policy_two, &device)
                .unwrap()
                .conditioning,
        );
        assert_eq!(resumed, original[(step - 1) as usize], "step {step}");
    }
    // And the pattern is not trivially constant, or the above proves nothing.
    assert!(original.windows(2).any(|w| w[0] != w[1]));
}

// ---------------------------------------------------------------------------
// End to end on the tiny Krea 2 bundle
// ---------------------------------------------------------------------------

const BUNDLE: &str = "tests/fixtures/tiny-krea2";
const DATASET: &str = "tests/fixtures/dataset-tiny";
const STEPS: u64 = 6;

fn staged_dataset(out: &TempDir) -> PathBuf {
    let dst = out.0.join("dataset");
    std::fs::create_dir_all(&dst).unwrap();
    for entry in std::fs::read_dir(DATASET).expect("checked-in dataset present") {
        let path = entry.unwrap().path();
        if path.is_file() {
            std::fs::copy(&path, dst.join(path.file_name().unwrap())).unwrap();
        }
    }
    dst
}

/// Every run gets its own output dir: `resume.auto` (the default) would
/// otherwise find the previous run's finished artifact and continue it for
/// zero steps.
static RUN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn train_config(out: &TempDir, dataset: &Path, rate: f32) -> TrainConfig {
    TrainConfig {
        steps: STEPS,
        seed: 42,
        task: TaskKind::FlowMatching,
        model: ModelConfig {
            base: BUNDLE.into(),
            variant: ModelVariant::TinyKrea2,
            ..Default::default()
        },
        lora: LoraConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            targets: vec![TargetSpec {
                pattern: r"blocks\.".into(),
                rank: None,
                alpha: None,
            }],
        },
        dataset: DatasetConfig {
            path: dataset.to_path_buf(),
            resolution: 32,
            batch_size: 2,
            caption_dropout_rate: rate,
            ..Default::default()
        },
        optim: OptimConfig {
            lr: 0.01,
            weight_decay: 0.0,
            ..OptimConfig::default()
        },
        output: OutputConfig {
            dir: out.0.join(format!(
                "out-{}",
                RUN.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            )),
            name: "krea2-lora".into(),
            checkpoint_every: 10_000,
            sample_every: 0,
        },
        ..Default::default()
    }
}

fn losses(config: &TrainConfig) -> (Vec<f32>, PathBuf) {
    let mut losses = Vec::new();
    let adapter = DiffusionTrainer
        .train(config, &mut |event| {
            if let TrainEvent::Step { loss, .. } = event {
                losses.push(loss);
            }
        })
        .expect("the run completes");
    (losses, adapter)
}

#[test]
fn a_positive_rate_changes_the_diffusion_loss_trajectory_and_the_default_does_not() {
    let _lock = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("dropout-e2e");
    let dataset = staged_dataset(&out);

    let (off_a, off_adapter) = losses(&train_config(&out, &dataset, 0.0));
    let (off_b, _) = losses(&train_config(&out, &dataset, 0.0));
    let (half, half_adapter) = losses(&train_config(&out, &dataset, 0.5));
    let (half_again, _) = losses(&train_config(&out, &dataset, 0.5));
    let (all, _) = losses(&train_config(&out, &dataset, 1.0));

    for run in [&off_a, &half, &all] {
        assert_eq!(run.len(), STEPS as usize);
        assert!(run.iter().all(|l| l.is_finite()), "{run:?}");
    }
    // Determinism first, or "differs" below could be mere nondeterminism: the
    // default reproduces itself, and so does a seeded non-zero rate.
    assert_eq!(off_a, off_b, "rate 0 is deterministic");
    assert_eq!(half, half_again, "a seeded non-zero rate is reproducible");

    let differs = |a: &[f32], b: &[f32]| a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-6);
    assert!(
        differs(&off_a, &half),
        "rate 0.5 must change the loss trajectory — identical streams mean the \
         knob is being ignored.\n off:  {off_a:?}\n half: {half:?}"
    );
    assert!(
        differs(&half, &all),
        "rate 1.0 must differ from 0.5.\n half: {half:?}\n all: {all:?}"
    );

    // The recorded rate is the configured one (the header describes the run).
    assert_eq!(
        read_metadata(&off_adapter)
            .unwrap()
            .get("ss_caption_dropout_rate"),
        Some("0")
    );
    assert_eq!(
        read_metadata(&half_adapter)
            .unwrap()
            .get("ss_caption_dropout_rate"),
        Some("0.5")
    );

    // One shared empty entry in the dataset's cache; none was needed by the
    // rate-0 runs (which went first) — they would have left no `.empty.` file.
    let empties = cache_files(&dataset)
        .into_iter()
        .filter(|n| n.contains(".empty."))
        .count();
    assert_eq!(empties, 1);
}

#[test]
fn the_empty_entry_is_not_written_by_a_rate_zero_diffusion_run() {
    let _lock = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("dropout-e2e-off");
    let dataset = staged_dataset(&out);
    let (run, _) = losses(&train_config(&out, &dataset, 0.0));
    assert_eq!(run.len(), STEPS as usize);
    assert!(
        !cache_files(&dataset).iter().any(|n| n.contains(".empty.")),
        "a default run must leave the cache exactly as it was: {:?}",
        cache_files(&dataset)
    );
}
