#!/usr/bin/env bash
# Smoke-test the engine-wasm wasm-bindgen glue under Node.
#
# Runs the marshalling round-trips in crates/engine-wasm/tests/wasm_roundtrip.rs
# (JSON in -> JSON out for find_power / merge_power_results / parse_formula) --
# the only coverage of the wasm-bindgen export layer, which the native unit
# tests cannot reach (the glue is #[cfg(target_arch = "wasm32")]). This is a
# glue smoke gate, not a numeric oracle.
#
# Prereqs (one-time, user-level):
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-pack            # or your distro's package
set -euo pipefail

# Resolve the workspace root (mcpower/) relative to this script.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(dirname "$script_dir")"

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "error: wasm-pack not found -- install it: cargo install wasm-pack" >&2
  exit 1
fi
if ! rustup target list --installed 2>/dev/null | grep -qx 'wasm32-unknown-unknown'; then
  echo "error: wasm32 target missing -- add it: rustup target add wasm32-unknown-unknown" >&2
  exit 1
fi

cd "$workspace_dir"
exec wasm-pack test --node crates/engine-wasm
