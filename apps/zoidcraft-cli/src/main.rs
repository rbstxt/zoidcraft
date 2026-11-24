//! ZoidCraft headless CLI and verification runner (L5).

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "zoidcraft-cli",
    version,
    about = "ZoidCraft headless renderer and CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Print workspace readiness info.
    Info,
}

fn main() {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    match cli.command.unwrap_or(Commands::Info) {
        Commands::Info => {
            let session = zc_session::Session::new();
            println!(
                "zoidcraft-cli ready (commands: {}, plugins: {}, render: {}, export: {})",
                session.registry.len(),
                session.plugins.len(),
                zc_render::crate_name(),
                zc_export::crate_name(),
            );
        }
    }
}
