//! Loss / learning-rate curve PNG (`plot` feature, #215).
//!
//! A CLI-side renderer over the [`TrainEvent`] stream, exactly like the
//! `indicatif` bar: [`CurveRecorder`] accumulates `(step, loss, lr)` from
//! [`TrainEvent::Step`] and, on [`TrainEvent::Finished`], writes
//! `<output.dir>/<output.name>.curve.png` next to the adapter. `loractl-core`
//! knows nothing about it — core emits events, front-ends render.
//!
//! A failed plot must never fail a run that already produced a good adapter,
//! so [`CurveRecorder::observe`] logs a `tracing` warning and swallows the
//! error; [`CurveRecorder::write`] is the fallible, testable half.
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

/// Where the curve lands: next to the adapter, `<dir>/<name>.curve.png`.
pub fn curve_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.curve.png"))
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
    path: PathBuf,
    points: Vec<Point>,
}

impl CurveRecorder {
    /// A recorder that will write to `path` (see [`curve_path`]).
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            points: Vec::new(),
        }
    }

    /// Number of steps recorded so far.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Feed one event. `Step` is recorded; `Finished` writes the PNG, and a
    /// failure there is logged and swallowed — the adapter already exists.
    pub fn observe(&mut self, event: &TrainEvent) {
        match event {
            TrainEvent::Step { step, loss, lr } => self.points.push(Point {
                step: *step,
                loss: *loss,
                lr: *lr,
            }),
            TrainEvent::Finished { .. } => match self.write() {
                Ok(()) => tracing::info!(path = %self.path.display(), "wrote loss/LR curve"),
                Err(err) => tracing::warn!(
                    path = %self.path.display(),
                    "could not write loss/LR curve (the adapter is unaffected): {err:#}"
                ),
            },
            _ => {}
        }
    }

    /// Render the recorded series to the PNG at the recorder's path.
    ///
    /// An empty series still produces a valid (axes-only) image rather than
    /// panicking on an empty range; non-finite values are skipped.
    pub fn write(&self) -> Result<()> {
        register_font()?;
        let finite: Vec<&Point> = self
            .points
            .iter()
            .filter(|p| p.loss.is_finite() && p.lr.is_finite())
            .collect();
        let xs: Vec<f64> = finite.iter().map(|p| p.step as f64).collect();
        let loss: Vec<f64> = finite.iter().map(|p| f64::from(p.loss)).collect();
        let lr: Vec<f64> = finite.iter().map(|p| p.lr).collect();

        let root = BitMapBackend::new(&self.path, SIZE).into_drawing_area();
        root.fill(&WHITE).map_err(plot_err)?;
        let (top, bottom) = root.split_vertically(SIZE.1 / 2);
        panel(&top, "loss", &xs, &loss, &BLUE, |v| format!("{v:.4}"))?;
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
    let (x_lo, x_hi) = match (xs.first(), xs.last()) {
        (Some(&a), Some(&b)) if b > a => (a, b),
        (Some(&a), _) => (a, a + 1.0),
        _ => (0.0, 1.0),
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

    fn finished() -> TrainEvent {
        TrainEvent::Finished {
            adapter_path: PathBuf::from("unused.safetensors"),
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

    #[test]
    fn curve_path_sits_next_to_the_adapter() {
        assert_eq!(
            curve_path(Path::new("out"), "lora"),
            PathBuf::from("out/lora.curve.png")
        );
    }

    #[test]
    fn finished_writes_a_valid_nonempty_png() {
        let dir = tmp();
        let path = curve_path(dir.path(), "run");
        let mut rec = CurveRecorder::new(path.clone());
        for i in 1..=50u64 {
            rec.observe(&step(i, 2.0 / i as f32, 1e-4 * (1.0 - i as f64 / 60.0)));
        }
        assert_eq!(rec.len(), 50);
        rec.observe(&finished());

        let bytes = std::fs::read(&path).expect("curve written on Finished");
        assert!(bytes.len() > 1000, "suspiciously small: {}", bytes.len());
        assert_eq!(&bytes[..8], PNG_MAGIC);
    }

    #[test]
    fn non_step_events_are_not_recorded_and_write_nothing() {
        let dir = tmp();
        let path = curve_path(dir.path(), "run");
        let mut rec = CurveRecorder::new(path.clone());
        rec.observe(&TrainEvent::Started { total_steps: 3 });
        rec.observe(&TrainEvent::Warning {
            message: "w".into(),
        });
        assert_eq!(rec.len(), 0);
        assert!(!path.exists(), "nothing is written before Finished");
    }

    #[test]
    fn empty_series_does_not_panic_and_still_yields_a_png() {
        let dir = tmp();
        let path = curve_path(dir.path(), "empty");
        CurveRecorder::new(path.clone())
            .write()
            .expect("empty series renders");
        assert_eq!(&std::fs::read(&path).unwrap()[..8], PNG_MAGIC);
    }

    #[test]
    fn single_point_and_constant_series_do_not_panic() {
        let dir = tmp();
        let one = curve_path(dir.path(), "one");
        let mut rec = CurveRecorder::new(one.clone());
        rec.observe(&step(1, 0.5, 1e-4));
        rec.write().expect("single point renders");
        assert_eq!(&std::fs::read(&one).unwrap()[..8], PNG_MAGIC);

        // Flat loss and flat LR: a degenerate y range must be padded, not
        // handed to plotters as an empty interval.
        let flat = curve_path(dir.path(), "flat");
        let mut rec = CurveRecorder::new(flat.clone());
        for i in 1..=5 {
            rec.observe(&step(i, 1.0, 1e-4));
        }
        rec.write().expect("flat series renders");
        assert_eq!(&std::fs::read(&flat).unwrap()[..8], PNG_MAGIC);
    }

    #[test]
    fn non_finite_points_are_skipped_not_fatal() {
        let dir = tmp();
        let path = curve_path(dir.path(), "nan");
        let mut rec = CurveRecorder::new(path.clone());
        rec.observe(&step(1, f32::NAN, 1e-4));
        rec.observe(&step(2, 0.4, 1e-4));
        rec.observe(&step(3, f32::INFINITY, 1e-4));
        rec.write().expect("non-finite values are filtered");
        assert_eq!(&std::fs::read(&path).unwrap()[..8], PNG_MAGIC);
    }

    #[test]
    fn unwritable_destination_is_an_error_and_observe_swallows_it() {
        let dir = tmp();
        // The parent directory does not exist, so the PNG cannot be created.
        let path = curve_path(&dir.path().join("missing").join("nested"), "run");
        let mut rec = CurveRecorder::new(path.clone());
        rec.observe(&step(1, 0.5, 1e-4));
        assert!(rec.write().is_err(), "write must report the failure");
        // The run-facing path must not panic or propagate.
        rec.observe(&finished());
        assert!(!path.exists());
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
