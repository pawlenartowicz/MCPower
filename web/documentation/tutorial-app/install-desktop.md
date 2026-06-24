---
title: "Installing the MCPower desktop app"
description: "Download & install the MCPower desktop app on Windows, macOS, or Linux - offline power analysis with no Python required."
---
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

Two native packages are provided; pick the one that suits your distribution.

- **.deb** — on Debian, Ubuntu, or Mint, download the `.deb` `[needs release]` and install it with your package manager (e.g. `sudo apt install ./<file>.deb`), or double-click it to open your graphical software installer. MCPower then appears in your applications menu.
- **.rpm** — on Fedora, RHEL, or openSUSE, download the `.rpm` `[needs release]` and install it with `sudo dnf install ./<file>.rpm` (or `sudo zypper install`), or double-click it to open your graphical software installer. MCPower then appears in your applications menu.

## Updating

To update, download the newer installer from GitHub Releases and install over the existing version — your history and settings are preserved.
