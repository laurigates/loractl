//! Repro: `Tensor::backward()` called inside a custom op's
//! `Backward::backward` deadlocks permanently (burn 0.21.0; the locking
//! structure is unchanged at main@e5467f0). Filed upstream as
//! tracel-ai/burn#5193; fix PR tracel-ai/burn#5194 (try_lock in
//! `cleanup_orphaned_entries`).
//!
//! Re-run this probe on every burn version bump (esp. the 0.22 migration,
//! #79): a clean exit 0 means the `block_ckpt.rs` two-phase structure could
//! be replaced by a custom op with a nested backward.
//!
//! Mechanism (`burn-autodiff/src/runtime/graph.rs`):
//! 1. The outer backward executes every step while holding its graph's
//!    `state` mutex (`GraphMutexClient::backward` wraps `server.backward`
//!    in `graph.state.lock()`).
//! 2. Any inner `backward()` ends with
//!    `GraphCleaner::cleanup_orphaned_entries()`, which locks EVERY
//!    registered graph's mutex — including the outer one this same thread
//!    already holds. `parking_lot::Mutex` is non-reentrant → permanent hang,
//!    even though the inner graph shares no tensors with the outer one.
//!
//! Run: cargo run -p loractl-core --example nested_backward_probe
//! Buggy behavior: prints "entering nested inner backward…" then hangs;
//! the watchdog exits(1) after 15 s. Fixed behavior: prints the outer
//! gradient and exits 0.

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind, unary};
use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Tensor, TensorPrimitive};

type Inner = NdArray;
type Ad = Autodiff<Inner>;

/// Forward = identity; backward runs a fully disjoint inner autodiff graph
/// (the natural shape of op-level gradient checkpointing: recompute the
/// region and backprop it inside the outer backward).
fn nested_identity(x: Tensor<Ad, 1>) -> Tensor<Ad, 1> {
    #[derive(Debug)]
    struct NestedIdentity;

    impl Backward<Inner, 1> for NestedIdentity {
        type State = ();

        fn backward(
            self,
            ops: Ops<Self::State, 1>,
            grads: &mut Gradients,
            _checkpointer: &mut Checkpointer,
        ) {
            unary::<Inner, _>(ops.parents, ops.node, grads, |grad| {
                eprintln!("entering nested inner backward…");
                // A brand-new graph: fresh tensors, no connection to the
                // outer graph whatsoever.
                let a = Tensor::<Ad, 1>::ones([4], &Default::default()).require_grad();
                let inner_loss = a.clone().sum();
                let inner_grads = inner_loss.backward(); // ← never returns
                let _ = a.grad(&inner_grads);
                eprintln!("inner backward returned (bug is fixed)");
                grad
            });
        }
    }

    let TensorPrimitive::Float(xp) = x.into_primitive() else {
        unreachable!("float input")
    };
    let out = xp.primitive.clone();
    let out = match NestedIdentity
        .prepare::<burn::backend::autodiff::checkpoint::strategy::NoCheckpointing>([xp
            .node
            .clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish((), out),
        OpsKind::UnTracked(prep) => prep.finish(out),
    };
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

fn main() {
    // The deadlock is permanent; a watchdog converts the hang into a
    // diagnosable non-zero exit.
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(15));
        eprintln!("DEADLOCK: outer backward did not complete within 15 s");
        std::process::exit(1);
    });

    let x = Tensor::<Ad, 1>::ones([4], &Default::default()).require_grad();
    let y = nested_identity(x.clone());
    let loss = y.sum();
    eprintln!("starting outer backward…");
    let grads = loss.backward();
    let gx = x.grad(&grads).expect("grad for x");
    println!("outer backward completed; grad = {gx}");
}
