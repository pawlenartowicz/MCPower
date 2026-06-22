import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/svelte';
import { tick } from 'svelte';

async function flushAll() {
  // Flush microtasks multiple rounds to drain dynamic imports + chained awaits.
  for (let i = 0; i < 20; i++) {
    await tick();
    await new Promise((r) => setTimeout(r, 0));
  }
}

// Mock vega-embed's default export: capture the call, return a fake view.
const finalize = vi.fn();
const embed = vi.fn(async () => ({ view: { finalize } }));
vi.mock('vega-embed', () => ({ default: embed }));

// Provide deterministic CSS vars (jsdom getComputedStyle returns '' otherwise).
vi.mock('$lib/charts/theme', async (orig) => {
  const real = (await orig()) as Record<string, unknown>;
  return {
    ...real,
    readChartColors: () => ({
      chart1: '#111111', chart2: '#222222', chart3: '#333333',
      chart4: '#444444', chart5: '#555555',
      fg: '#000000', mutedFg: '#666666', border: '#cccccc',
    }),
    onThemeChange: () => () => {},
  };
});

import VegaChart from './VegaChart.svelte';

describe('VegaChart', () => {
  beforeEach(() => {
    embed.mockClear();
    finalize.mockClear();
  });

  it('embeds the parsed spec with a derived theme config', async () => {
    const spec = JSON.stringify({
      $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
      data: { values: [{ a: 1, b: 2 }] },
      mark: 'bar',
      encoding: { x: { field: 'a' }, y: { field: 'b' } },
    });
    render(VegaChart, { props: { spec, testid: 'bars-view' } });
    await flushAll();
    expect(embed).toHaveBeenCalledTimes(1);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const passedSpec = (embed.mock.calls as any)[0][1] as Record<string, any>;
    expect(passedSpec.config.range.category[0]).toBe('#111111');
    expect(passedSpec.mark).toBe('bar'); // engine encoding preserved
  });

  it('exposes the testid container', () => {
    const { getByTestId } = render(VegaChart, {
      props: { spec: '{"mark":"bar"}', testid: 'curve-view' },
    });
    expect(getByTestId('curve-view')).toBeTruthy();
  });

  it('warns (with the testid) on a malformed spec instead of failing silently', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    render(VegaChart, { props: { spec: '{not valid json', testid: 'bad-view' } });
    await flushAll();
    expect(warn).toHaveBeenCalled();
    const message = warn.mock.calls[0]?.[0] as string;
    expect(message).toContain('bad-view'); // testid surfaced for debugging
    warn.mockRestore();
  });
});
