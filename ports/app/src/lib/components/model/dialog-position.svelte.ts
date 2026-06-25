// Positions the model-editor popups (predictor/cluster Advanced, Model more
// options) over the CONFIG (left) pane when it is wide enough to host the
// dialog, so the popup sits next to what it edits instead of covering the
// results pane. Falls back to the default viewport-centered position when the
// pane is too narrow or the app is in single-pane mode.
import { innerWidth } from 'svelte/reactivity/window';
import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';

// Default DialogContent is sm:max-w-md (28rem = 448px); require a little margin
// around it. Because the dialog centers on the pane midpoint via
// -translate-x-1/2, its left half (≈ width/2) must fit left of that midpoint or
// it clips off the screen edge — so the pane must be at least the dialog's full
// width. Wider dialogs (e.g. the effect-sizes sm:max-w-2xl) pass their own
// requiredPanePx so they fall back to viewport-center sooner instead of clipping.
const DIALOG_SPACE_PX = 480;
// Mirrors App.svelte's `isNarrow` breakpoint (single-pane below 900px) — change together.
const SINGLE_PANE_BELOW_PX = 900;

/** Inline style overriding the dialog's `left-1/2` when the config pane can
 *  host it; `undefined` keeps the default centered position. Call inside a
 *  reactive context — reads the window width and the splitter fraction.
 *  `requiredPanePx` is the minimum left-pane width (≈ dialog width + margin)
 *  below which it falls back to viewport-center. */
export function configPaneDialogStyle(requiredPanePx = DIALOG_SPACE_PX): string | undefined {
  const w = innerWidth.current ?? 0;
  if (w < SINGLE_PANE_BELOW_PX) return undefined;
  const fraction = sharedPrefs.splitterFraction;
  if (fraction * w < requiredPanePx) return undefined;
  // Center of the left pane; the dialog's own -translate-x-1/2 still applies.
  return `left: ${((fraction * 100) / 2).toFixed(2)}vw`;
}
