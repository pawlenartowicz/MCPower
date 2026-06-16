//! macOS native menu definition and event forwarding; emits menu item IDs as `"menu"` events on the app handle.

use tauri::menu::{Menu, MenuBuilder, MenuEvent, MenuItemBuilder, SubmenuBuilder};
use tauri::{AppHandle, Emitter, Runtime};

/// Assemble the app menu; item IDs (`file.new`, …) are matched by the
/// frontend's `menu`-event listener.
pub fn build_menu<R: Runtime>(handle: &AppHandle<R>) -> tauri::Result<Menu<R>> {
    let file = SubmenuBuilder::new(handle, "File")
        .item(&MenuItemBuilder::with_id("file.new", "New analysis").build(handle)?)
        .item(
            &MenuItemBuilder::with_id("file.export_results", "Export results…")
                .accelerator("CmdOrCtrl+E")
                .build(handle)?,
        )
        .separator()
        .quit()
        .build()?;

    let edit = SubmenuBuilder::new(handle, "Edit")
        .item(&MenuItemBuilder::with_id("edit.reset_family", "Reset family config").build(handle)?)
        .build()?;

    let view = SubmenuBuilder::new(handle, "View")
        .item(&MenuItemBuilder::with_id("view.toggle_config", "Toggle Config pane").build(handle)?)
        .item(
            &MenuItemBuilder::with_id("view.toggle_results", "Toggle Results pane")
                .build(handle)?,
        )
        .separator()
        .item(&MenuItemBuilder::with_id("view.settings", "Settings…").build(handle)?)
        .item(&MenuItemBuilder::with_id("view.history", "History").build(handle)?)
        .build()?;

    let run = SubmenuBuilder::new(handle, "Run")
        .item(
            &MenuItemBuilder::with_id("run.find_power", "Find power")
                .accelerator("CmdOrCtrl+R")
                .build(handle)?,
        )
        .item(
            &MenuItemBuilder::with_id("run.find_n", "Find sample")
                .accelerator("CmdOrCtrl+Shift+R")
                .build(handle)?,
        )
        .item(&MenuItemBuilder::with_id("run.cancel", "Cancel current run").build(handle)?)
        .separator()
        .item(
            &MenuItemBuilder::with_id("run.rerun", "Re-run last")
                .accelerator("CmdOrCtrl+Return")
                .build(handle)?,
        )
        .build()?;

    let help = SubmenuBuilder::new(handle, "Help")
        .item(&MenuItemBuilder::with_id("help.documentation", "Documentation").build(handle)?)
        .separator()
        .item(&MenuItemBuilder::with_id("help.acknowledgments", "Acknowledgments").build(handle)?)
        .build()?;

    MenuBuilder::new(handle)
        .items(&[&file, &edit, &view, &run, &help])
        .build()
}

/// Forward a menu click to JS as a `menu` event carrying the item ID.
pub fn handle_menu_event<R: Runtime>(app: &AppHandle<R>, event: MenuEvent) {
    let id = event.id().as_ref().to_string();
    let _ = app.emit("menu", id);
}
