#!/usr/bin/env bash
# deploy-site.sh — build + deploy mcpower.app to Cloudflare Pages (direct upload).
#
# Assembles the static site (web/site/, copied verbatim) under assets/ with the
# VITE_TARGET=wasm app build injected at /online/, plus the top-level
# site-functions/ staged as a SIBLING functions/ dir, then `wrangler pages
# deploy assets`s (run from the staging root) to the `mcpower-site` Pages
# project. No wrangler.toml — direct upload only.
#
# functions/ MUST be a sibling of the deployed asset dir (in wrangler's cwd),
# NOT inside it: Cloudflare Pages only compiles functions/ when it sits at the
# project root next to the static root. Nest it under the asset dir and wrangler
# uploads report.ts as a public static file instead of compiling it, so
# POST /api/report 405s and the source leaks. (Pages Functions docs: "Make sure
# the /functions directory is at the root of your Pages project, not in the
# static root such as /dist.")
# One-time setup: `pnpm dlx wrangler login`; the first deploy prompts to create
# the project (production branch: main).
#
# Usage (from anywhere):
#   scripts/deploy-site.sh                # preview deploy (keyed off the current git branch)
#   scripts/deploy-site.sh --production   # production deploy → mcpower.app (branch=main)
#
# wrangler prints the deployment URL (*.pages.dev preview or production) —
# verify there before/after the custom domain exists.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --production forces the Cloudflare "production" deployment (branch=main, the one
# the mcpower.app custom domain serves). Without it wrangler keys the environment
# off the current git branch, so deploying from any non-main branch lands on a
# *.pages.dev preview, NOT mcpower.app. --commit-dirty silences the uncommitted-
# changes warning, which production hotfixes off a feature branch always trip.
DEPLOY_ARGS=(--project-name=mcpower-site)
if [[ "${1:-}" == "--production" ]]; then
  DEPLOY_ARGS+=(--branch=main --commit-dirty=true)
fi

# web/site/online would silently shadow or collide with the injected app build —
# that path belongs to this script alone.
if [[ -e "$REPO_ROOT/web/site/online" ]]; then
  echo "deploy-site.sh: web/site/online exists — that path is owned by the deploy script; remove it" >&2
  exit 1
fi

echo "==> Building wasm app (wasm-pack + vite)"
(cd "$REPO_ROOT/ports/app" && pnpm build:wasm-app)

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "==> Assembling staging dir: $STAGE"
mkdir -p "$STAGE/assets"
cp -r "$REPO_ROOT/web/site/." "$STAGE/assets/"
cp -r "$REPO_ROOT/ports/app/dist" "$STAGE/assets/online"
# Pages Functions live outside web/site/ (every web/site/ file is a public URL); stage
# only the api/ tree as functions/api/ (→ /api/report), a sibling of assets/ so
# wrangler compiles it — package.json/test/ stay out of the deploy.
mkdir -p "$STAGE/functions"
cp -r "$REPO_ROOT/site-functions/api" "$STAGE/functions/"

echo "==> Deploying to Cloudflare Pages (project: mcpower-site)"
# Run wrangler with $STAGE as cwd so it discovers the sibling functions/ next to
# the deployed assets/ dir; --dir only points pnpm at the app toolchain that
# resolves wrangler (it does not change wrangler's cwd).
(cd "$STAGE" && pnpm --dir "$REPO_ROOT/ports/app" dlx wrangler pages deploy assets "${DEPLOY_ARGS[@]}")
