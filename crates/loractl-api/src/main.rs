//! Binary shell for `loractl-api`: env config, tracing init, the one real
//! trainer line, and the server loop. Everything else lives in the library
//! so integration tests exercise the exact same `app()`.

use loractl_api::TrainerFactory;
use loractl_core::BurnTrainer;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    // The analogue of cli.rs's single `BurnTrainer` construction line: the
    // only place the API names a concrete trainer.
    let factory: TrainerFactory = Arc::new(|| Box::new(BurnTrainer));

    let addr = std::env::var("LORACTL_API_ADDR").unwrap_or_else(|_| String::from("127.0.0.1:3000"));
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("loractl-api listening on http://{addr}");
    axum::serve(listener, loractl_api::app(factory)).await?;
    Ok(())
}
