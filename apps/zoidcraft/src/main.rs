//! ZoidCraft desktop application entry point (L5).

use eframe::egui;

struct ZoidCraftApp;

impl eframe::App for ZoidCraftApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ui, |ui| {
            ui.heading("ZoidCraft");
            ui.label(format!(
                "Commands registered: {}",
                zc_session::CommandRegistry::new_default().len()
            ));
        });
    }
}

fn main() -> eframe::Result {
    tracing_subscriber::fmt::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("ZoidCraft"),
        ..Default::default()
    };
    eframe::run_native(
        "ZoidCraft",
        options,
        Box::new(|_cc| Ok(Box::new(ZoidCraftApp))),
    )
}
