# Installing the desktop app

The desktop app is the same MCPower interface as the [[tutorial-app/index|web app]], packaged as a native download for Windows, macOS, and Linux. It runs entirely on your machine — no Python, no internet connection needed after install. If you don't want to install anything, the web app runs the identical interface in your browser.

Installers are published on **GitHub Releases**. Download the file for your platform from the latest release `[needs release]`, then follow the steps below.

## Windows

1. Download the `.msi` installer from the latest release `[needs release]`.
2. Double-click the file and follow the installer prompts.
3. Launch **MCPower** from the Start menu.

If Windows SmartScreen warns about an unrecognised app, choose **More info → Run anyway**.

## macOS

1. Download the `.dmg` for your chip — Apple silicon (M-series) or Intel `[needs release]`.
2. Open the `.dmg` and drag **MCPower** into your **Applications** folder.
3. Launch it from Applications.

The first launch may need **right-click → Open** (or **System Settings → Privacy & Security → Open Anyway**) to bypass Gatekeeper for an app installed outside the App Store.

## Linux

Two formats are provided; pick the one that suits your distribution.

- **AppImage** — download the `.AppImage` `[needs release]`, mark it executable (`chmod +x` or via your file manager's Properties), and run it. No system install; it is self-contained.
- **.deb** — on Debian/Ubuntu, download the `.deb` `[needs release]` and install it with your package manager (e.g. `sudo apt install ./<file>.deb`). MCPower then appears in your applications menu.

## Updating

To update, download the newer installer from GitHub Releases and install over the existing version — your history and settings are preserved.
