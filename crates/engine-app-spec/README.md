# engine-app-spec

Host-agnostic adapter layer shared by every MCPower port that takes the
GUI-style `AppSpec` envelope (Tauri desktop, browser WASM). Holds the spec
types, assembler, find-power / find-sample-size drivers, and the
`ProgressEmitter` trait. No host-specific imports.

See the workspace-root `CLAUDE.md` §2 for the contract and §3 for the per-port wiring.
