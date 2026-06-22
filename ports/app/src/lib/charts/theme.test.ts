import { describe, it, expect } from 'vitest';
import { buildVegaConfig, type ChartColors } from './theme';

const COLORS: ChartColors = {
  chart1: '#111111',
  chart2: '#222222',
  chart3: '#333333',
  chart4: '#444444',
  chart5: '#555555',
  fg: '#000000',
  mutedFg: '#666666',
  border: '#cccccc',
};

describe('buildVegaConfig', () => {
  it('maps CSS chart colors into a Vega-Lite config', () => {
    const cfg = buildVegaConfig(COLORS) as Record<string, any>;
    expect(cfg.range.category).toEqual(['#111111', '#222222', '#333333', '#444444', '#555555']);
    expect(cfg.axis.titleColor).toBe('#000000');
    expect(cfg.axis.labelColor).toBe('#666666');
    expect(cfg.axis.gridColor).toBe('#cccccc');
    expect(cfg.legend.labelColor).toBe('#000000');
    expect(cfg.mark.color).toBe('#111111');
    expect(cfg.background).toBe('transparent');
  });
});
