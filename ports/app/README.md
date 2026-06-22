# MCPower — Desktop App

```
███╗   ███╗  ██████╗ ██████╗ 
████╗ ████║ ██╔════╝ ██╔══██╗ ██████╗ ██╗    ██╗███████╗██████╗ 
██╔████╔██║ ██║      ██║  ██║██╔═══██╗██║    ██║██╔════╝██╔══██╗
██║╚██╔╝██║ ██║      ██████╔╝██║   ██║██║ █╗ ██║█████╗  ██████╔╝
██║ ╚═╝ ██║ ██║      ██╔═══╝ ██║   ██║██║███╗██║██╔══╝  ██╔══██╗
██║     ██║ ╚██████╗ ██║     ╚██████╔╝╚███╔███╔╝███████╗██║  ██║
╚═╝     ╚═╝  ╚═════╝ ╚═╝      ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝
```

**Power analysis by simulation — any design from t-test to mixed models, in your browser, on your desktop, or in Python and R.**

[![Download](https://img.shields.io/badge/Download-GitHub_Releases-2ea44f?logo=github)](https://github.com/pawlenartowicz/mcpower/releases)
[![Platforms](https://img.shields.io/badge/Platforms-Windows_%7C_macOS_%7C_Linux-informational)](https://github.com/pawlenartowicz/mcpower/releases)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-green.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.16502734-blue)](https://doi.org/10.5281/zenodo.16502734)

The MCPower desktop application — a no-code GUI over the same native Rust engine that powers the Python and R packages. Run Monte Carlo power analysis without writing code.

**Download:** ready-to-run installers for Windows, macOS, and Linux from the [Releases page](https://github.com/pawlenartowicz/mcpower/releases).

## First launch

The current builds are not yet code-signed, so the OS shows a one-time warning the first time you open the app:

- **macOS:** right-click (or Control-click) the app → **Open** → **Open** in the dialog. If macOS refuses outright, run `xattr -dr com.apple.quarantine /Applications/MCPower.app` in Terminal, then open it normally.
- **Windows:** at the SmartScreen "Windows protected your PC" prompt, click **More info** → **Run anyway**.

This is a one-time step per install. Installing via `winget` (once available) or running the [browser app at mcpower.app](https://mcpower.app/online) avoids the prompt entirely.

**Docs:** [App tutorial](https://docs.mcpower.app/tutorial-app/index) · **Project:** [MCPower on GitHub](https://github.com/pawlenartowicz/mcpower)
