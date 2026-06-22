// Reads chart colors from CSS custom properties and provides alpha/theme-change helpers.
export interface ChartColors {
  chart1: string;
  chart2: string;
  chart3: string;
  chart4: string;
  chart5: string;
  fg: string;
  mutedFg: string;
  border: string;
}

function readVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function readChartColors(): ChartColors {
  return {
    chart1: readVar('--chart-1'),
    chart2: readVar('--chart-2'),
    chart3: readVar('--chart-3'),
    chart4: readVar('--chart-4'),
    chart5: readVar('--chart-5'),
    fg: readVar('--foreground'),
    mutedFg: readVar('--muted-foreground'),
    border: readVar('--border'),
  };
}

export function withAlpha(color: string, alpha: number): string {
  const c = color.trim();
  if (c.startsWith('#')) {
    let h = c.slice(1);
    if (h.length === 3) {
      h = h
        .split('')
        .map((ch) => ch + ch)
        .join('');
    }
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return c;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  const m = c.match(/^rgba?\(([^)]+)\)$/i);
  if (m?.[1]) {
    const parts = m[1].split(',').map((s) => s.trim());
    if (parts.length >= 3) return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
  }
  return c;
}

export function onThemeChange(cb: () => void): () => void {
  const obs = new MutationObserver(() => cb());
  obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
  return () => obs.disconnect();
}

/**
 * Build a Vega-Lite `config` object (a host-applied theme overlay) from the
 * app's live CSS chart variables. Merged onto the engine's theme-naked spec at
 * render time; re-derived on theme change. The engine spec carries data +
 * encoding only — colors/fonts live here, per the theme-overlay contract.
 */
export function buildVegaConfig(c: ChartColors): Record<string, unknown> {
  return {
    background: 'transparent',
    view: { stroke: 'transparent' },
    range: { category: [c.chart1, c.chart2, c.chart3, c.chart4, c.chart5] },
    mark: { color: c.chart1 },
    axis: {
      labelColor: c.mutedFg,
      titleColor: c.fg,
      gridColor: c.border,
      domainColor: c.border,
      tickColor: c.border,
    },
    legend: { labelColor: c.fg, titleColor: c.fg },
    header: { labelColor: c.fg, titleColor: c.fg },
    title: { color: c.fg },
  };
}

/**
 * Deep-merge `overlay` into `base` (mutates base). Mirrors the `deepMerge`
 * helper in ports/wasm/src/index.ts — ports never import from each other so
 * the tiny helper is copied here.
 */
function deepMerge(base: Record<string, unknown>, overlay: Record<string, unknown>): Record<string, unknown> {
  for (const [key, value] of Object.entries(overlay)) {
    const existing = base[key];
    if (existing !== null && typeof existing === 'object' && !Array.isArray(existing) &&
        value !== null && typeof value === 'object' && !Array.isArray(value)) {
      deepMerge(existing as Record<string, unknown>, value as Record<string, unknown>);
    } else {
      base[key] = value;
    }
  }
  return base;
}

/**
 * Apply the `print` theme from `configs/plot-themes.json` onto a parsed
 * theme-naked Vega-Lite spec object (mutates `spec`). Used by ExportTab to
 * embed the print theme into saved artifacts (white background, black axes,
 * colour-blind–friendly palette — no live CSS involved).
 * Import `plotThemes` via `import plotThemes from '$configs/plot-themes.json'`
 * and pass `plotThemes.print` as `printTheme`.
 */
export function applyPrintTheme(
  spec: Record<string, unknown>,
  printTheme: Record<string, unknown>,
): void {
  const existing = spec['config'];
  const base: Record<string, unknown> =
    existing !== null && typeof existing === 'object' && !Array.isArray(existing)
      ? (existing as Record<string, unknown>)
      : {};
  spec['config'] = deepMerge(base, printTheme);
}
