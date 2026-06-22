// Shared embed helper: parses a theme-naked Vega-Lite spec, relabels generic
// target tokens to real effect names, sizes it to the host while preserving the
// engine's aspect ratio, applies a theme overlay (CSS live or print artifact),
// and embeds it via vega-embed.
// Returns the vega-embed `Result` so callers can access `.view` and call
// `.finalize()` when done. VegaChart and ExportTab both use this path.
import { readChartColors, buildVegaConfig, applyPrintTheme, type ChartColors } from './theme';
import { relabelTargets } from './labels';

/** vega-embed Result shape — only the fields we use. */
export interface EmbedResult {
  view: {
    toImageURL(type: 'png' | 'svg', scale?: number): Promise<string>;
    toSVG(scaleFactor?: number): Promise<string>;
    finalize(): void;
  };
  finalize(): void;
}

// Readable bounds for a single chart panel's rendered width. The engine emits a
// fixed ~360 px panel; we scale toward the host width but never below MIN (too
// cramped to read CIs) nor above MAX (a lone chart shouldn't swallow a wide
// window and flatten into an unreadable strip).
const MIN_PANEL_WIDTH = 300;
const MAX_PANEL_WIDTH = 600;
// Per-column allowance in faceted specs (axis labels, tick marks, gutters).
const FACET_AXIS_ALLOWANCE = 40;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(v, hi));
}

/**
 * In-place: size a single panel toward `containerWidth` while **preserving the
 * engine's aspect ratio** — both width and height scale by the same factor.
 * This is the fix for the old `width:"container"` behaviour, which stretched
 * width to the container but left height fixed, flattening curves and bars into
 * unreadable strips. Panels without numeric width/height (shouldn't happen for
 * engine specs) are left untouched.
 */
function fitPanel(panel: Record<string, unknown>, containerWidth: number): void {
  const w = panel.width;
  const h = panel.height;
  if (typeof w !== 'number' || typeof h !== 'number' || w <= 0) return;
  const target = clamp(containerWidth > 0 ? containerWidth : w, MIN_PANEL_WIDTH, MAX_PANEL_WIDTH);
  const scale = target / w;
  panel.width = Math.round(w * scale);
  panel.height = Math.round(h * scale);
}

/**
 * In-place aspect-ratio-preserving sizing. Single panels scale toward the host
 * width; a top-level `vconcat` scales each child panel; top-level `facet` specs
 * scale the inner per-panel size so the full grid (engine default: 3 columns)
 * fits `containerWidth` — panel width ≈ containerWidth / columns minus axis
 * allowance, clamped to the same MIN/MAX as a single panel.
 */
export function fitSpec(spec: Record<string, unknown>, containerWidth: number): void {
  if ('facet' in spec) {
    // Faceted spec: scale the inner panel so the full grid fits containerWidth.
    // The engine emits `columns: N` nested inside the `facet` object and
    // `width`/`height` on the inner `spec` object.
    const facet = spec.facet as Record<string, unknown> | undefined;
    const columnsRaw = facet && typeof facet.columns === 'number' ? facet.columns
      : typeof spec.columns === 'number' ? spec.columns : undefined;
    const columns = typeof columnsRaw === 'number' && columnsRaw > 0 ? columnsRaw : 3;
    const rawPanelWidth = containerWidth > 0
      ? Math.max(0, Math.floor(containerWidth / columns) - FACET_AXIS_ALLOWANCE)
      : 0;
    const inner = spec.spec;
    if (inner && typeof inner === 'object' && !Array.isArray(inner)) {
      fitPanel(inner as Record<string, unknown>, rawPanelWidth);
    }
    return;
  }
  if (Array.isArray(spec.vconcat)) {
    for (const child of spec.vconcat as Record<string, unknown>[]) {
      if (child && typeof child === 'object' && !('facet' in child)) {
        fitPanel(child, containerWidth);
      }
    }
    return;
  }
  fitPanel(spec, containerWidth);
}

/**
 * In-place: make error-bar CIs legible against same-coloured bars/lines. The
 * engine emits theme-naked `errorbar` marks with no colour, so they inherit the
 * bar colour and vanish. Vega-Lite forbids `color` in `config.errorbar`, so the
 * contrasting colour must be set on the mark itself — this stays in host
 * theme-overlay territory (colour + size only). End ticks are enabled so the
 * interval bounds are visible. Single-series error bars (no `color` encoding)
 * get a foreground whisker; grouped/multi-scenario error bars keep their
 * per-series colour and just gain ticks.
 */
function styleErrorBars(spec: Record<string, unknown>, colors: ChartColors): void {
  walkLayers(spec, colors);
}

function walkLayers(node: unknown, colors: ChartColors): void {
  if (Array.isArray(node)) {
    for (const child of node) walkLayers(child, colors);
    return;
  }
  if (!node || typeof node !== 'object') return;
  const obj = node as Record<string, unknown>;
  const markType = typeof obj.mark === 'string' ? obj.mark : (obj.mark as { type?: string })?.type;
  if (markType === 'errorbar') {
    const md: Record<string, unknown> =
      typeof obj.mark === 'string' ? { type: 'errorbar' } : { ...(obj.mark as object) };
    const hasColorEncoding = !!(obj.encoding as { color?: unknown })?.color;
    if (hasColorEncoding) {
      md.ticks = true; // grouped bars keep per-series colour; ticks add end caps
    } else {
      md.ticks = { color: colors.fg };
      md.rule = { color: colors.fg, strokeWidth: 1.5 };
    }
    obj.mark = md;
  }
  for (const value of Object.values(obj)) walkLayers(value, colors);
}

/**
 * Parse `specString`, relabel target tokens (if `labelMap` given), size it to
 * the host width (aspect-ratio preserving), apply the live CSS theme + CI
 * styling, and embed via vega-embed into `host`. Returns the embed result (call
 * `.finalize()` when done). Throws if the spec JSON is malformed.
 */
export async function embedThemed(
  host: HTMLElement,
  specString: string,
  labelMap?: Record<string, string>,
): Promise<EmbedResult> {
  const parsed: Record<string, unknown> = JSON.parse(specString);
  if (labelMap) relabelTargets(parsed, labelMap);
  fitSpec(parsed, host.clientWidth);
  const colors = readChartColors();
  styleErrorBars(parsed, colors);
  parsed.config = buildVegaConfig(colors);
  const { default: embed } = await import('vega-embed');
  return embed(host, parsed as never, { actions: false, renderer: 'svg' }) as Promise<EmbedResult>;
}

/**
 * Parse `specString`, apply the `print` theme (white background, black axes,
 * colour-blind palette — no live CSS), and embed via vega-embed into `host`.
 * Used by ExportTab for saved artifacts. `printTheme` should be
 * `plotThemes.print` from `$configs/plot-themes.json`.
 * Returns the embed result (call `.finalize()` when done).
 * No container fitting: exported files keep the engine spec's exact dimensions
 * (same file style as the py/R ports); the caller controls scale when
 * exporting (toImageURL / toSVG).
 */
export async function embedPrint(
  host: HTMLElement,
  specString: string,
  printTheme: Record<string, unknown>,
): Promise<EmbedResult> {
  const parsed: Record<string, unknown> = JSON.parse(specString);
  // CI styling: use black for error bars on the print theme (fg = #000000).
  styleErrorBars(parsed, { chart1: '#0072B2', chart2: '#E69F00', chart3: '#009E73',
    chart4: '#D55E00', chart5: '#56B4E9', fg: '#000000', mutedFg: '#333333', border: '#dddddd' });
  applyPrintTheme(parsed, printTheme);
  const { default: embed } = await import('vega-embed');
  return embed(host, parsed as never, { actions: false, renderer: 'svg' }) as Promise<EmbedResult>;
}
