import { fireEvent, render } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { PlotSpecs } from '$lib/domain/result';
import type { RunTab } from '$lib/stores/run.svelte';

// ---------------------------------------------------------------------------
// Mock plugin-dialog and plugin-fs so the save path can be tested without
// a running Tauri process.
// ---------------------------------------------------------------------------

const saveMock = vi.fn(async () => '/tmp/mcpower-plot.png');
vi.mock('@tauri-apps/plugin-dialog', () => ({ save: saveMock }));

const writeFileMock = vi.fn(async () => {});
const writeTextFileMock = vi.fn(async () => {});
vi.mock('@tauri-apps/plugin-fs', () => ({
  writeFile: writeFileMock,
  writeTextFile: writeTextFileMock,
}));

// ---------------------------------------------------------------------------
// Mock embedPrint: returns a fake view with toImageURL / toSVG / finalize.
// vi.mock is hoisted, so the factory must not reference outer let/const.
// Use vi.hoisted() to declare the spies before hoisting.
// ---------------------------------------------------------------------------

const { finalizeMock, embedPrintMock } = vi.hoisted(() => {
  const finalizeMock = vi.fn();
  const viewFinalizeMock = vi.fn();
  const toImageURLMock = vi.fn(async () => 'data:image/png;base64,abc=');
  const toSVGMock = vi.fn(async () => '<svg></svg>');
  const embedPrintMock = vi.fn(async () => ({
    view: {
      toImageURL: toImageURLMock,
      toSVG: toSVGMock,
      finalize: viewFinalizeMock,
    },
    finalize: finalizeMock,
  }));
  return { finalizeMock, embedPrintMock };
});

// Browser/WASM download seam — used when VITE_TARGET=wasm.
const downloadBlobMock = vi.fn();
vi.mock('$lib/api/menu-wasm', () => ({ downloadBlob: downloadBlobMock }));

vi.mock('$lib/charts/embed', () => ({
  embedPrint: embedPrintMock,
}));

import ExportTab from './ExportTab.svelte';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const PLOTS: PlotSpecs = {
  blocks: [
    { key: 'power', spec: '{"mark":"bar"}' },
  ],
};

const MULTI_PLOTS: PlotSpecs = {
  blocks: [
    { key: 'scenario:default', spec: '{"mark":"line"}' },
    { key: 'scenario:optimistic', spec: '{"mark":"line"}' },
    { key: 'overlay', spec: '{"mark":"line"}' },
    { key: 'at_least_k', spec: '{"mark":"line"}' },
    { key: 'exactly_k', spec: '{"mark":"line"}' },
  ],
};

function makeTab(plots?: PlotSpecs): RunTab {
  const result = {
    n: 80,
    n_sims: 1600,
    target_indices: [0],
    power_uncorrected: [0.8],
    power_corrected: [0.8],
    ci_uncorrected: [{ lo: 0.7, hi: 0.9 }],
    ci_corrected: [{ lo: 0.7, hi: 0.9 }],
    convergence_rate: 1,
    boundary_hit: [],
    estimator_extras: { estimator: 'ols' as const },
  };
  return {
    id: 'tab-1',
    label: 'Run 1',
    kind: 'find-power',
    subView: 'export',
    spec: {
      family: 'linear' as const,
      parsed_formula: {
        outcome: 'y',
        predictors: ['x'],
        interaction_terms: [],
      },
      var_types: [{ kind: 'numeric' as const, name: 'x' }],
      effects: [{ name: 'x', value: 0.3 }],
      correlations: null,
      alpha: 0.05,
      target_power: 0.8,
      n_sims: 1600,
      seed: 2137,
      tests: { kind: 'all' as const },
      correction: 'none' as const,
      scenarios: [],
      csv: null,
      report_overall: true,
      contrasts: [],
    },
    effect_names: ['x'],
    result,
    scenarios: [['default', result]],
    plots,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('ExportTab — no-plots state', () => {
  it('shows the no-plots message and no save button when plots is absent', () => {
    const { getByTestId, queryByTestId } = render(ExportTab, {
      props: { tab: makeTab(undefined) },
    });
    expect(getByTestId('export-tab')).toBeTruthy();
    expect(getByTestId('export-no-plots')).toBeTruthy();
    expect(queryByTestId('export-save-btn')).toBeNull();
  });
});

describe('ExportTab — format and scale UI', () => {
  it('shows format selector, scale input (default PNG), and Save button', () => {
    const { getByTestId } = render(ExportTab, { props: { tab: makeTab(PLOTS) } });
    const formatSelect = getByTestId('export-format-select') as HTMLSelectElement;
    expect(formatSelect.value).toBe('png');
    expect(getByTestId('export-scale-input')).toBeTruthy();
    expect(getByTestId('export-save-btn')).toBeTruthy();
  });

  it('hides the scale input when SVG is selected', async () => {
    const { getByTestId, queryByTestId } = render(ExportTab, {
      props: { tab: makeTab(PLOTS) },
    });
    const formatSelect = getByTestId('export-format-select') as HTMLSelectElement;
    await fireEvent.change(formatSelect, { target: { value: 'svg' } });
    expect(queryByTestId('export-scale-input')).toBeNull();
  });

  it('shows the scale input again when PNG is re-selected', async () => {
    const { getByTestId } = render(ExportTab, { props: { tab: makeTab(PLOTS) } });
    const formatSelect = getByTestId('export-format-select') as HTMLSelectElement;
    await fireEvent.change(formatSelect, { target: { value: 'svg' } });
    await fireEvent.change(formatSelect, { target: { value: 'png' } });
    expect(getByTestId('export-scale-input')).toBeTruthy();
  });
});

describe('ExportTab — block selector', () => {
  it('shows a block selector with one option for single-block runs', () => {
    const { getByTestId } = render(ExportTab, { props: { tab: makeTab(PLOTS) } });
    const sel = getByTestId('export-block-select') as HTMLSelectElement;
    expect(sel).toBeTruthy();
    expect(sel.options.length).toBe(1);
    expect(sel.value).toBe('power');
  });

  it('shows all blocks in the selector for multi-block runs', () => {
    const { getByTestId } = render(ExportTab, { props: { tab: makeTab(MULTI_PLOTS) } });
    const sel = getByTestId('export-block-select') as HTMLSelectElement;
    expect(sel.options.length).toBe(5); // scenario:default, scenario:optimistic, overlay, at_least_k, exactly_k
    // First option is the first block.
    expect(sel.value).toBe('scenario:default');
  });
});

describe('ExportTab — print theme', () => {
  beforeEach(() => {
    embedPrintMock.mockClear();
    saveMock.mockClear();
    writeFileMock.mockClear();
    finalizeMock.mockClear();
  });

  it('calls embedPrint (not embedThemed) when saving', async () => {
    const { getByTestId } = render(ExportTab, { props: { tab: makeTab(PLOTS) } });
    await fireEvent.click(getByTestId('export-save-btn'));
    await new Promise((r) => setTimeout(r, 20));
    expect(embedPrintMock).toHaveBeenCalledOnce();
    // The third argument to embedPrint is the print theme — must include background:#ffffff.
    const call = embedPrintMock.mock.calls[0] as unknown as [unknown, unknown, Record<string, unknown>];
    const printTheme = call[2];
    expect(printTheme['background']).toBe('#ffffff');
  });
});

describe('ExportTab — save path (PNG)', () => {
  beforeEach(() => {
    embedPrintMock.mockClear();
    saveMock.mockClear();
    writeFileMock.mockClear();
    writeTextFileMock.mockClear();
    finalizeMock.mockClear();
  });

  it('calls writeFile with bytes decoded from the data URL when PNG is saved', async () => {
    const { getByTestId } = render(ExportTab, { props: { tab: makeTab(PLOTS) } });
    await fireEvent.click(getByTestId('export-save-btn'));
    // Allow async work to complete
    await new Promise((r) => setTimeout(r, 20));
    expect(saveMock).toHaveBeenCalledOnce();
    expect(writeFileMock).toHaveBeenCalledOnce();
    const [path, bytes] = writeFileMock.mock.calls[0] as unknown as [string, Uint8Array];
    expect(path).toBe('/tmp/mcpower-plot.png');
    expect(bytes).toBeInstanceOf(Uint8Array);
    // 'abc=' base64 decodes to 1 byte (0x69)
    expect(bytes.length).toBeGreaterThan(0);
    expect(finalizeMock).toHaveBeenCalledOnce();
  });

  it('does not call writeFile when the dialog is cancelled (returns null)', async () => {
    saveMock.mockResolvedValueOnce(null as unknown as string);
    const { getByTestId } = render(ExportTab, { props: { tab: makeTab(PLOTS) } });
    await fireEvent.click(getByTestId('export-save-btn'));
    await new Promise((r) => setTimeout(r, 20));
    expect(writeFileMock).not.toHaveBeenCalled();
  });
});

describe('ExportTab — save path (SVG)', () => {
  beforeEach(() => {
    embedPrintMock.mockClear();
    saveMock.mockClear();
    writeTextFileMock.mockClear();
    finalizeMock.mockClear();
  });

  it('calls writeTextFile with the SVG string when SVG is saved', async () => {
    saveMock.mockResolvedValueOnce('/tmp/mcpower-plot.svg');
    const { getByTestId } = render(ExportTab, { props: { tab: makeTab(PLOTS) } });
    const formatSelect = getByTestId('export-format-select') as HTMLSelectElement;
    await fireEvent.change(formatSelect, { target: { value: 'svg' } });
    await fireEvent.click(getByTestId('export-save-btn'));
    await new Promise((r) => setTimeout(r, 20));
    expect(writeTextFileMock).toHaveBeenCalledOnce();
    const [path, content] = writeTextFileMock.mock.calls[0] as unknown as [string, string];
    expect(path).toBe('/tmp/mcpower-plot.svg');
    expect(content).toBe('<svg></svg>');
    expect(finalizeMock).toHaveBeenCalledOnce();
  });
});

describe('ExportTab — browser/WASM save path', () => {
  beforeEach(() => {
    embedPrintMock.mockClear();
    downloadBlobMock.mockClear();
    saveMock.mockClear();
    writeFileMock.mockClear();
    writeTextFileMock.mockClear();
    finalizeMock.mockClear();
    vi.stubEnv('VITE_TARGET', 'wasm');
  });
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('downloads a PNG blob via the browser path, never touching Tauri plugins', async () => {
    const { getByTestId } = render(ExportTab, { props: { tab: makeTab(PLOTS) } });
    await fireEvent.click(getByTestId('export-save-btn'));
    await new Promise((r) => setTimeout(r, 20));
    expect(saveMock).not.toHaveBeenCalled();
    expect(writeFileMock).not.toHaveBeenCalled();
    expect(downloadBlobMock).toHaveBeenCalledOnce();
    const [filename, blob] = downloadBlobMock.mock.calls[0] as unknown as [string, Blob];
    expect(filename).toBe('mcpower-plot.png');
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toBe('image/png');
    expect(finalizeMock).toHaveBeenCalledOnce();
  });

  it('downloads an SVG blob via the browser path when SVG is selected', async () => {
    const { getByTestId } = render(ExportTab, { props: { tab: makeTab(PLOTS) } });
    const formatSelect = getByTestId('export-format-select') as HTMLSelectElement;
    await fireEvent.change(formatSelect, { target: { value: 'svg' } });
    await fireEvent.click(getByTestId('export-save-btn'));
    await new Promise((r) => setTimeout(r, 20));
    expect(writeTextFileMock).not.toHaveBeenCalled();
    expect(downloadBlobMock).toHaveBeenCalledOnce();
    const [filename, blob] = downloadBlobMock.mock.calls[0] as unknown as [string, Blob];
    expect(filename).toBe('mcpower-plot.svg');
    expect(blob.type).toBe('image/svg+xml');
  });
});
