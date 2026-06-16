# MCPower — Browser (WASM)

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

[![WebAssembly](https://img.shields.io/badge/WebAssembly-654FF0?logo=webassembly&logoColor=white)](https://webassembly.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-green.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.16502734-blue)](https://doi.org/10.5281/zenodo.16502734)

The MCPower engine compiled to WebAssembly — run Monte Carlo power analysis entirely in the browser, no server, over the same native Rust engine that powers the Python and R packages.

**Use it:** [mcpower.app](https://mcpower.app) — the web app built on this package, no install needed.

This package (`@mcpower/engine-wasm`) is the browser engine layer — wasm-pack output + JS worker-pool runtime — consumed from source by the web app. It is a private workspace package, not published to npm.

**Docs:** [App tutorial](https://docs.mcpower.app/tutorial-app/index) · **Project:** [MCPower on GitHub](https://github.com/pawlenartowicz/MCPower)
