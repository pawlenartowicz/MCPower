#!/usr/bin/env bash
# preview-docs.sh — serve mcpower/documentation as a leyline vault at
# http://127.0.0.1:8101/ for local theme + content preview. Run from anywhere.
#
# leyline-web (the vault server) is not part of this repo. The script prefers a
# `leyline-web` already on PATH; otherwise it builds one from a local leyline
# checkout (override the location with LEYLINE_REPO; default ~/Projekty/leyline).
# It writes a throwaway config mounting the docs vault at / and serves it with
# leyline's bundled themes.
#
# The docs theme overlay lives in documentation/.leyline/vaultconfig/. If it is
# absent the vault still renders, falling back to the bundled `notes` theme via
# default_theme below.
#
# Usage (from anywhere):  scripts/preview-docs.sh
#   LEYLINE_REPO=/path/to/leyline  scripts/preview-docs.sh   # non-default checkout
#   LEYLINE_PORT=9000              scripts/preview-docs.sh   # port already in use

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VAULT="$REPO_ROOT/documentation"
LEYLINE_REPO="${LEYLINE_REPO:-$HOME/Projekty/leyline}"
PORT="${LEYLINE_PORT:-8101}"

ENGINE="$LEYLINE_REPO/repos/web-source"        # Go module with cmd/leyline-web
THEMES="$LEYLINE_REPO/repos/web/config/themes" # notes + leyline_base live here

[ -d "$VAULT" ] || { echo "preview-docs.sh: docs vault not found at $VAULT" >&2; exit 1; }
[ -d "$THEMES" ] || {
  echo "preview-docs.sh: leyline themes not found at $THEMES — set LEYLINE_REPO to your leyline checkout" >&2
  exit 1
}

# Prefer an installed binary; fall back to building from the checkout.
if command -v leyline-web >/dev/null 2>&1; then
  BIN="$(command -v leyline-web)"
else
  [ -d "$ENGINE" ] || {
    echo "preview-docs.sh: no leyline-web on PATH and engine source not found at $ENGINE — set LEYLINE_REPO" >&2
    exit 1
  }
  BIN=/tmp/leyline-web-mcpower
  echo "==> Building leyline-web (from $ENGINE)"
  go -C "$ENGINE" build -o "$BIN" ./cmd/leyline-web
fi

# leyline-web has no --vault flag; it reads a config.yaml whose `vaults` map
# points URL prefixes at vault directories. Mount the docs vault at the root.
CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "$CONFIG"' EXIT
cat >"$CONFIG" <<YAML
domain: localhost
listen: "127.0.0.1:$PORT"
dev_mode: true
default_theme: notes
vaults:
  "/": "$VAULT"
YAML

echo "==> Serving $VAULT"
echo "→ Open http://127.0.0.1:$PORT/ in your browser. Ctrl-C to stop."
# Not `exec` — keeping the shell as parent lets the EXIT trap delete $CONFIG when
# the server is stopped (Ctrl-C propagates to the child through the process group).
"$BIN" -config "$CONFIG" --themes "$THEMES"
