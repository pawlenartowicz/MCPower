<script lang="ts">
// Draggable vertical splitter dividing left (config) and right (results) panes; fraction persisted in sharedPrefs.
import type { Snippet } from 'svelte';
import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';

interface Props {
  left: Snippet;
  right: Snippet;
}
const { left, right }: Props = $props();

let container = $state<HTMLDivElement>();
let dragging = $state(false);
let containerWidth = $state(0);

const MIN_PANE_PX = 240;
const STEP = 0.02;
const STEP_LARGE = 0.1;
const DEFAULT_FRACTION = 0.33;

function bounds(widthPx: number): { min: number; max: number } {
  const tooNarrow = widthPx < MIN_PANE_PX * 2;
  const min = tooNarrow ? 0.2 : MIN_PANE_PX / widthPx;
  return { min, max: 1 - min };
}

function clampFraction(f: number, widthPx: number): number {
  const { min, max } = bounds(widthPx);
  return Math.max(min, Math.min(max, f));
}

function syncWidth() {
  if (container) containerWidth = container.getBoundingClientRect().width;
}

$effect(() => {
  syncWidth();
  if (!container) return;
  const ro = new ResizeObserver(syncWidth);
  ro.observe(container);
  return () => ro.disconnect();
});

const ariaBounds = $derived(bounds(containerWidth || 1));
const ariaValueNow = $derived(Math.round(sharedPrefs.splitterFraction * 100));
const ariaValueMin = $derived(Math.round(ariaBounds.min * 100));
const ariaValueMax = $derived(Math.round(ariaBounds.max * 100));

function onPointerDown(e: PointerEvent) {
  dragging = true;
  (e.target as Element).setPointerCapture(e.pointerId);
}
function onPointerMove(e: PointerEvent) {
  if (!dragging || !container) return;
  const rect = container.getBoundingClientRect();
  const f = (e.clientX - rect.left) / rect.width;
  sharedPrefs.splitterFraction = clampFraction(f, rect.width);
}
function onPointerUp(e: PointerEvent) {
  dragging = false;
  (e.target as Element).releasePointerCapture(e.pointerId);
}
function reset() {
  sharedPrefs.splitterFraction = DEFAULT_FRACTION;
}
function onKeyDown(e: KeyboardEvent) {
  if (!container) return;
  const width = container.getBoundingClientRect().width;
  const step = e.shiftKey ? STEP_LARGE : STEP;
  let next = sharedPrefs.splitterFraction;
  switch (e.key) {
    case 'ArrowLeft':
      next -= step;
      break;
    case 'ArrowRight':
      next += step;
      break;
    case 'Home':
      next = bounds(width).min;
      break;
    case 'End':
      next = bounds(width).max;
      break;
    case 'Enter':
    case ' ':
      next = DEFAULT_FRACTION;
      break;
    default:
      return;
  }
  e.preventDefault();
  sharedPrefs.splitterFraction = clampFraction(next, width);
}
</script>

<div bind:this={container} class="relative flex flex-1 overflow-hidden">
  <div class="overflow-hidden" style="flex-basis: {sharedPrefs.splitterFraction * 100}%">
    {@render left()}
  </div>
  <div
    role="separator"
    aria-orientation="vertical"
    aria-valuenow={ariaValueNow}
    aria-valuemin={ariaValueMin}
    aria-valuemax={ariaValueMax}
    aria-label="Resize panes"
    tabindex="0"
    class="w-1 cursor-col-resize bg-border hover:bg-primary/40 focus-visible:bg-primary/60 focus-visible:outline-none"
    onpointerdown={onPointerDown}
    onpointermove={onPointerMove}
    onpointerup={onPointerUp}
    ondblclick={reset}
    onkeydown={onKeyDown}
  ></div>
  <div class="overflow-hidden" style="flex-basis: {(1 - sharedPrefs.splitterFraction) * 100}%">
    {@render right()}
  </div>
</div>
