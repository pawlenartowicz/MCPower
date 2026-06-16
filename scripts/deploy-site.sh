#!/usr/bin/env bash
# deploy-site.sh — build + deploy mcpower.app to Cloudflare Pages (direct upload).
#
# Assembles the static site (site/, copied verbatim) with the VITE_TARGET=wasm
# app build injected at /online/, then `wrangler pages deploy`s the staging dir
# to the `mcpower-site` Pages project. No wrangler.toml — direct upload only.
# One-time setup: `pnpm dlx wrangler login`; the first deploy prompts to create
# the project (production branch: main).
#
# Usage (from anywhere): scripts/deploy-site.sh
#
# wrangler prints the deployment URL (*.pages.dev preview or production) —
# verify there before/after the custom domain exists.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# site/online would silently shadow or collide with the injected app build —
# that path belongs to this script alone.
if [[ -e "$REPO_ROOT/site/online" ]]; then
  echo "deploy-site.sh: site/online exists — that path is owned by the deploy script; remove it" >&2
  exit 1
fi

echo "==> Building wasm app (wasm-pack + vite)"
(cd "$REPO_ROOT/ports/app" && pnpm build:wasm-app)

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "==> Assembling staging dir: $STAGE"
cp -r "$REPO_ROOT/site/." "$STAGE/"
cp -r "$REPO_ROOT/ports/app/dist" "$STAGE/online"

echo "==> Deploying to Cloudflare Pages (project: mcpower-site)"
(cd "$REPO_ROOT/ports/app" && pnpm dlx wrangler pages deploy "$STAGE" --project-name=mcpower-site)
