//! Tauri application entry point; wires plugins, registers commands, and sets up the macOS menu.

mod commands;
#[cfg(target_os = "macos")]
mod menu;
mod run_registry;
mod window_emitter;

use run_registry::RunRegistry;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default()
        .manage(RunRegistry::default())
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            #[cfg(target_os = "macos")]
            {
                let m = menu::build_menu(app.handle())?;
                app.set_menu(m)?;
            }
            Ok(())
        });

    #[cfg(target_os = "macos")]
    let builder = builder.on_menu_event(menu::handle_menu_event);

    builder
        .invoke_handler(tauri::generate_handler![
            crate::commands::find_power_cmd,
            crate::commands::find_sample_size_cmd,
            crate::commands::cancel_run_cmd,
            crate::commands::parse_formula_cmd,
            crate::commands::get_effects_from_data_cmd,
            crate::commands::effect_skeleton_cmd,
            crate::commands::set_n_threads_cmd,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
