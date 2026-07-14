//! Binary shell for `loractl-api`: env config, tracing init, the real
//! trainer factory, and the server loop. Everything else lives in the
//! library so integration tests exercise the exact same `app()`.

use loractl_api::{ApiConfig, TrainerFactory};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    // The analogue of cli.rs's trainer seam: routing on `model.base` lives
    // in core (`select_trainer`) so the CLI and the API cannot drift apart.
    let factory: TrainerFactory = Arc::new(loractl_core::select_trainer);

    let config = ApiConfig::from_env()?;
    let addr = std::env::var("LORACTL_API_ADDR").unwrap_or_else(|_| String::from("127.0.0.1:3000"));
    // Built before the listener: a bad output base must fail on boot, not on
    // the first request against an already-listening socket.
    let app = loractl_api::app(factory, config.clone())?;
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(
        run_retention = config.run_retention,
        max_concurrent_runs = config.max_concurrent_runs,
        output_base = %config.output_base.display(),
        "loractl-api listening on http://{addr}"
    );
    axum::serve(listener, app).await?;
    Ok(())
}
