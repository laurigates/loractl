//! Loss / learning-rate curve PNG (`plot` feature, #215).
//!
//! A CLI-side renderer over the [`TrainEvent`] stream, exactly like the
//! `indicatif` bar: [`CurveRecorder`] accumulates `(step, loss, lr)` from
//! [`TrainEvent::Step`] and, on [`TrainEvent::Finished`], writes
//! `<adapter stem>.curve.png` next to the adapter (the path comes from the
//! `Finished` event, so it can never disagree with where the adapter went).
//! `loractl-core` knows nothing about it — core emits events, front-ends render.
//!
//! The PNG is written by [`CurveRecorder::finish`], which the caller invokes
//! *after* the trainer returns — not from inside the event callback — so its
//! `tracing` output cannot interleave with a live progress bar. A failed plot
//! must never fail a run that already produced a good adapter, so `finish`
//! logs a warning and swallows the error; [`CurveRecorder::write_to`] is the
//! fallible, testable half.
//!
//! The font is embedded (`assets/fonts/DejaVuSansMono.ttf`, license alongside)
//! and drawn through plotters' pure-Rust `ab_glyph` path, so the feature needs
//! no system freetype/fontconfig.

use anyhow::{Result, anyhow};
use loractl_core::TrainEvent;
use plotters::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Family name the embedded font is registered under.
const FONT: &str = "loractl-mono";

/// Image size in pixels: two stacked panels (loss over LR), one shared x axis
/// meaning (step) so the schedule can be read against the loss it produced.
const SIZE: (u32, u32) = (1000, 700);

static FONT_REGISTERED: OnceLock<std::result::Result<(), String>> = OnceLock::new();

/// Register the embedded font with plotters, once per process.
fn register_font() -> Result<()> {
    FONT_REGISTERED
        .get_or_init(|| {
            plotters::style::register_font(
                FONT,
                plotters::style::FontStyle::Normal,
                include_bytes!("../assets/fonts/DejaVuSansMono.ttf"),
            )
            .map_err(|_| "embedded curve font failed to parse".to_string())
        })
        .clone()
        .map_err(|e| anyhow!(e))
}

/// Where the curve lands: beside the adapter, sharing its stem
/// (`out/lora.safetensors` -> `out/lora.curve.png`). Derived from the adapter
/// path the trainer reported rather than re-derived from `output.name`, so a
/// name with dots or separators lands exactly where the adapter did.
pub fn curve_path(adapter: &Path) -> PathBuf {
    adapter.with_extension("curve.png")
}

/// One recorded optimization step.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Point {
    step: u64,
    loss: f32,
    lr: f64,
}

/// Accumulates the step series and writes the curve PNG at the end of a run.
#[derive(Debug)]
pub struct CurveRecorder {
    /// The adapter path from `Finished`; `None` until the run completes, so an
    /// aborted run (no `Finished`) writes no curve.
    adapter: Option<PathBuf>,
    points: Vec<Point>,
}

impl Default for CurveRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl CurveRecorder {
    /// An empty recorder.
    pub fn new() -> Self {
        Self {
            adapter: None,
            points: Vec::new(),
        }
    }

    /// Number of steps recorded so far.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Feed one event. `Step` is recorded and `Finished` remembers where the
    /// adapter went; nothing is written until [`finish`](Self::finish).
    pub fn observe(&mut self, event: &TrainEvent) {
        match event {
            TrainEvent::Step { step, loss, lr } => self.points.push(Point {
                step: *step,
                loss: *loss,
                lr: *lr,
            }),
            TrainEvent::Finished { adapter_path } => self.adapter = Some(adapter_path.clone()),
            _ => {}
        }
    }

    /// Steps whose loss or lr is NaN/inf. They cannot be placed on an axis, so
    /// the plot omits them; a diverged run is exactly when the operator needs
    /// to be told rather than shown a curve that quietly stops.
    fn non_finite(&self) -> usize {
        self.points
            .iter()
            .filter(|p| !(p.loss.is_finite() && p.lr.is_finite()))
            .count()
    }

    /// If the run reached `Finished`, write the curve beside the adapter. A
    /// failure is logged and swallowed — the adapter already exists. Call this
    /// after the trainer returns (the progress bar is done by then).
    pub fn finish(&self) {
        let Some(adapter) = &self.adapter else {
            return;
        };
        let path = curve_path(adapter);
        match self.write_to(&path) {
            Ok(()) => tracing::info!(path = %path.display(), "wrote loss/LR curve"),
            Err(err) => tracing::warn!(
                path = %path.display(),
                "could not write loss/LR curve (the adapter is unaffected): {err:#}"
            ),
        }
        let skipped = self.non_finite();
        if skipped > 0 {
            tracing::warn!(
                "{skipped} of {} steps had a non-finite loss or learning rate and are omitted from the curve",
                self.points.len()
            );
        }
    }

    /// Render the recorded series to the PNG at `path`.
    ///
    /// An empty series still produces a valid (axes-only) image rather than
    /// panicking on an empty range; non-finite values are skipped.
    pub fn write_to(&self, path: &Path) -> Result<()> {
        register_font()?;
        let finite: Vec<&Point> = self
            .points
            .iter()
            .filter(|p| p.loss.is_finite() && p.lr.is_finite())
            .collect();
        let xs: Vec<f64> = finite.iter().map(|p| p.step as f64).collect();
        let loss: Vec<f64> = finite.iter().map(|p| f64::from(p.loss)).collect();
        let lr: Vec<f64> = finite.iter().map(|p| p.lr).collect();

        let root = BitMapBackend::new(path, SIZE).into_drawing_area();
        root.fill(&WHITE).map_err(plot_err)?;
        let (top, bottom) = root.split_vertically(SIZE.1 / 2);
        panel(&top, "loss", &xs, &loss, &BLUE, fmt_loss)?;
        panel(&bottom, "learning rate", &xs, &lr, &RED, |v| {
            format!("{v:.2e}")
        })?;
        // `present` is where the PNG is actually encoded and written; the
        // drop-time flush swallows its error, so call it explicitly.
        root.present().map_err(plot_err)?;
        Ok(())
    }
}

/// plotters' backend errors are not `Send + Sync`; keep the message only.
fn plot_err<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("plot: {e}")
}

/// Loss tick label. Fixed-point is readable for ordinary losses but overflows
/// the label gutter for a diverged run (1e30) and collapses to `0.0000` for a
/// tiny one (1e-6), so those fall back to scientific notation.
fn fmt_loss(v: &f64) -> String {
    let a = v.abs();
    if a != 0.0 && !(1e-3..1e4).contains(&a) {
        format!("{v:.2e}")
    } else {
        format!("{v:.4}")
    }
}

/// `(lo, hi)` padded so a flat or single-value series still has a non-empty
/// y range (a constant LR is the common case until a schedule varies it).
fn padded(values: &[f64]) -> (f64, f64) {
    let (mut lo, mut hi) = values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    if !lo.is_finite() || !hi.is_finite() {
        return (0.0, 1.0);
    }
    let span = hi - lo;
    let pad = if span > 0.0 {
        span * 0.05
    } else if hi != 0.0 {
        hi.abs() * 0.1
    } else {
        1.0
    };
    lo -= pad;
    hi += pad;
    (lo, hi)
}

/// One chart panel: a line (plus point markers, so a single step is visible)
/// of `ys` against `xs` in the given drawing area.
fn panel(
    area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    title: &str,
    xs: &[f64],
    ys: &[f64],
    color: &RGBColor,
    fmt_y: impl Fn(&f64) -> String,
) -> Result<()> {
    // min/max, not first/last: nothing in the event contract forbids a step
    // sequence that is not strictly increasing, and a resumed run starts at
    // `resume_step + 1`, not 1.
    let (x_lo, x_hi) = match xs.iter().copied().reduce(f64::min) {
        Some(a) => {
            let b = xs.iter().copied().fold(a, f64::max);
            if b > a { (a, b) } else { (a, a + 1.0) }
        }
        None => (0.0, 1.0),
    };
    let (y_lo, y_hi) = padded(ys);

    let mut chart = ChartBuilder::on(area)
        .caption(title, (FONT, 20))
        .margin(10)
        .x_label_area_size(36)
        .y_label_area_size(90)
        .build_cartesian_2d(x_lo..x_hi, y_lo..y_hi)
        .map_err(plot_err)?;
    chart
        .configure_mesh()
        .x_desc("step")
        .x_label_formatter(&|v| format!("{v:.0}"))
        .y_label_formatter(&fmt_y)
        // Only the major grid: the default minor lines make a dense hatch.
        .light_line_style(TRANSPARENT)
        .label_style((FONT, 12))
        .axis_desc_style((FONT, 14))
        .draw()
        .map_err(plot_err)?;
    chart
        .draw_series(LineSeries::new(
            xs.iter().copied().zip(ys.iter().copied()),
            color.stroke_width(2),
        ))
        .map_err(plot_err)?;
    // A one-point series has no line to draw; markers keep it visible.
    if xs.len() <= 1 {
        chart
            .draw_series(
                xs.iter()
                    .zip(ys)
                    .map(|(&x, &y)| Circle::new((x, y), 4, color.filled())),
            )
            .map_err(plot_err)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const PNG_MAGIC: &[u8] = &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

    fn step(step: u64, loss: f32, lr: f64) -> TrainEvent {
        TrainEvent::Step { step, loss, lr }
    }

    fn finished_at(adapter: &Path) -> TrainEvent {
        TrainEvent::Finished {
            adapter_path: adapter.to_path_buf(),
        }
    }

    fn tmp() -> tempdir::Dir {
        tempdir::Dir::new()
    }

    /// Tiny scoped temp dir (no extra dev-dependency for one test helper).
    mod tempdir {
        use std::path::{Path, PathBuf};
        pub struct Dir(PathBuf);
        impl Dir {
            pub fn new() -> Self {
                let p = std::env::temp_dir().join(format!(
                    "loractl-plot-{}-{:?}",
                    std::process::id(),
                    std::thread::current().id()
                ));
                let _ = std::fs::remove_dir_all(&p);
                std::fs::create_dir_all(&p).expect("create temp dir");
                Dir(p)
            }
            pub fn path(&self) -> &Path {
                &self.0
            }
        }
        impl Drop for Dir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    fn png(path: &Path) -> Vec<u8> {
        std::fs::read(path).expect("curve written")
    }

    #[test]
    fn curve_path_shares_the_adapter_stem() {
        assert_eq!(
            curve_path(Path::new("out/lora.safetensors")),
            PathBuf::from("out/lora.curve.png")
        );
        // A dotted name must not lose its tail to `with_extension`.
        assert_eq!(
            curve_path(Path::new("out/lora.v2.safetensors")),
            PathBuf::from("out/lora.v2.curve.png")
        );
    }

    #[test]
    fn finish_after_finished_writes_a_valid_nonempty_png() {
        let dir = tmp();
        let adapter = dir.path().join("run.safetensors");
        let mut rec = CurveRecorder::new();
        for i in 1..=50u64 {
            rec.observe(&step(i, 2.0 / i as f32, 1e-4 * (1.0 - i as f64 / 60.0)));
        }
        assert_eq!(rec.len(), 50);
        rec.observe(&finished_at(&adapter));
        rec.finish();

        let bytes = png(&dir.path().join("run.curve.png"));
        assert!(bytes.len() > 1000, "suspiciously small: {}", bytes.len());
        assert_eq!(&bytes[..8], PNG_MAGIC);
    }

    #[test]
    fn observe_alone_writes_nothing_and_unfinished_runs_write_no_curve() {
        let dir = tmp();
        let mut rec = CurveRecorder::new();
        rec.observe(&TrainEvent::Started { total_steps: 3 });
        rec.observe(&TrainEvent::Warning {
            message: "w".into(),
        });
        assert_eq!(rec.len(), 0);
        rec.observe(&step(1, 0.5, 1e-4));
        // No `Finished` (an aborted run): finish() is a no-op.
        rec.finish();
        // And even with `Finished`, observe() itself never touches the disk.
        rec.observe(&finished_at(&dir.path().join("a.safetensors")));
        assert!(!dir.path().join("a.curve.png").exists());
    }

    #[test]
    fn empty_series_does_not_panic_and_still_yields_a_png() {
        let dir = tmp();
        let path = dir.path().join("empty.curve.png");
        CurveRecorder::new()
            .write_to(&path)
            .expect("empty series renders");
        assert_eq!(&png(&path)[..8], PNG_MAGIC);
    }

    #[test]
    fn single_point_and_constant_series_do_not_panic() {
        let dir = tmp();
        let one = dir.path().join("one.curve.png");
        let mut rec = CurveRecorder::new();
        rec.observe(&step(1, 0.5, 1e-4));
        rec.write_to(&one).expect("single point renders");
        assert_eq!(&png(&one)[..8], PNG_MAGIC);

        // Flat loss and flat LR: a degenerate y range must be padded, not
        // handed to plotters as an empty interval.
        let flat = dir.path().join("flat.curve.png");
        let mut rec = CurveRecorder::new();
        for i in 1..=5 {
            rec.observe(&step(i, 1.0, 1e-4));
        }
        rec.write_to(&flat).expect("flat series renders");
        assert_eq!(&png(&flat)[..8], PNG_MAGIC);
    }

    #[test]
    fn resumed_and_out_of_order_steps_render() {
        let dir = tmp();
        let path = dir.path().join("resume.curve.png");
        let mut rec = CurveRecorder::new();
        // A resumed run starts mid-count; a (contract-legal) non-monotonic
        // sequence must not collapse the x range to first..last.
        for s in [501, 502, 503, 700, 650, 600] {
            rec.observe(&step(s, 1.0 / s as f32, 1e-4));
        }
        rec.write_to(&path).expect("resumed series renders");
        assert_eq!(&png(&path)[..8], PNG_MAGIC);
    }

    #[test]
    fn extreme_losses_render() {
        let dir = tmp();
        let path = dir.path().join("extreme.curve.png");
        let mut rec = CurveRecorder::new();
        rec.observe(&step(1, 1e-9, 1e-4));
        rec.observe(&step(2, 3.0e38, 1e-4));
        rec.write_to(&path).expect("extreme magnitudes render");
        assert_eq!(&png(&path)[..8], PNG_MAGIC);
    }

    #[test]
    fn non_finite_points_are_skipped_and_counted() {
        let dir = tmp();
        let path = dir.path().join("nan.curve.png");
        let mut rec = CurveRecorder::new();
        rec.observe(&step(1, f32::NAN, 1e-4));
        rec.observe(&step(2, 0.4, 1e-4));
        rec.observe(&step(3, f32::INFINITY, 1e-4));
        rec.observe(&step(4, 0.3, f64::NAN));
        assert_eq!(rec.non_finite(), 3);
        rec.write_to(&path).expect("non-finite values are filtered");
        assert_eq!(&png(&path)[..8], PNG_MAGIC);

        // Entirely non-finite (a run that diverged on step 1) still renders.
        let mut all = CurveRecorder::new();
        all.observe(&step(1, f32::NAN, 1e-4));
        all.write_to(&dir.path().join("all.curve.png"))
            .expect("all-NaN series renders axes only");
    }

    #[test]
    fn unwritable_destination_is_an_error_and_finish_swallows_it() {
        let dir = tmp();
        // The parent directory does not exist, so the PNG cannot be created.
        let adapter = dir
            .path()
            .join("missing")
            .join("nested")
            .join("run.safetensors");
        let mut rec = CurveRecorder::new();
        rec.observe(&step(1, 0.5, 1e-4));
        assert!(
            rec.write_to(&curve_path(&adapter)).is_err(),
            "write must report the failure"
        );
        // The run-facing path must not panic or propagate.
        rec.observe(&finished_at(&adapter));
        rec.finish();
        assert!(!curve_path(&adapter).exists());
    }

    #[test]
    fn loss_labels_stay_short_at_both_extremes() {
        assert_eq!(fmt_loss(&0.0), "0.0000");
        assert_eq!(fmt_loss(&0.1234), "0.1234");
        assert_eq!(fmt_loss(&1e-6), "1.00e-6");
        assert_eq!(fmt_loss(&1e30), "1.00e30");
        assert_eq!(fmt_loss(&-2.5), "-2.5000");
    }

    #[test]
    fn padded_range_is_never_empty() {
        for vals in [
            &[][..],
            &[0.0][..],
            &[3.0][..],
            &[1.0, 1.0][..],
            &[1.0, 2.0][..],
        ] {
            let (lo, hi) = padded(vals);
            assert!(hi > lo, "{vals:?} -> ({lo}, {hi})");
        }
    }
}
