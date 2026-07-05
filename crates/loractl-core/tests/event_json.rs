//! Golden test pinning the `TrainEvent` wire schema byte-for-byte.
//!
//! These strings are the canonical JSON contract consumed by any renderer
//! that serializes events (the M5 HTTP/SSE API) and are reproduced verbatim
//! in `docs/api/events.md` — if this test and that doc ever disagree, the
//! doc has drifted.

use std::path::PathBuf;

use loractl_core::TrainEvent;

#[test]
fn train_event_wire_shapes() {
    let cases: Vec<(TrainEvent, &str)> = vec![
        (
            TrainEvent::Started { total_steps: 1000 },
            r#"{"type":"started","total_steps":1000}"#,
        ),
        (
            TrainEvent::Step {
                step: 42,
                loss: 1.2345,
                lr: 0.0001,
            },
            r#"{"type":"step","step":42,"loss":1.2345,"lr":0.0001}"#,
        ),
        (
            TrainEvent::Checkpoint {
                step: 250,
                path: PathBuf::from("output/checkpoint-250.safetensors"),
            },
            r#"{"type":"checkpoint","step":250,"path":"output/checkpoint-250.safetensors"}"#,
        ),
        (
            TrainEvent::Sample {
                step: 500,
                path: PathBuf::from("output/sample-500.png"),
            },
            r#"{"type":"sample","step":500,"path":"output/sample-500.png"}"#,
        ),
        (
            TrainEvent::Warning {
                message: "lr clipped".to_string(),
            },
            r#"{"type":"warning","message":"lr clipped"}"#,
        ),
        (
            TrainEvent::Finished {
                adapter_path: PathBuf::from("output/lora.safetensors"),
            },
            r#"{"type":"finished","adapter_path":"output/lora.safetensors"}"#,
        ),
    ];

    for (event, expected) in cases {
        let actual = serde_json::to_string(&event).expect("TrainEvent must serialize");
        assert_eq!(actual, expected, "wire shape drifted for {event:?}");
    }
}
