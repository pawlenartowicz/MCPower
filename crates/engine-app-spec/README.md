# engine-app-spec

Host-agnostic adapter layer shared by every MCPower port that takes the
GUI-style `AppSpec` envelope (Tauri desktop, browser WASM). Holds the spec
types, assembler, find-power / find-sample-size drivers, and the
`ProgressEmitter` trait. No host-specific imports.

The orchestrator contract (`engine-orchestrator`) is the load-bearing surface this crate binds to.
