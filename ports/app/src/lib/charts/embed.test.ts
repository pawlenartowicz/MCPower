import { describe, it, expect } from 'vitest';
import { fitSpec } from './embed';

describe('fitSpec', () => {
  it('scales a single panel toward containerWidth while preserving aspect ratio', () => {
    const spec: Record<string, unknown> = { width: 360, height: 240 };
    fitSpec(spec, 480);
    expect(spec.width).toBe(480);
    expect(spec.height).toBe(320); // 240 * (480/360)
  });

  it('falls back to natural size (clamped to MIN) when containerWidth is 0', () => {
    const spec: Record<string, unknown> = { width: 360, height: 240 };
    fitSpec(spec, 0);
    // containerWidth=0 → target = clamp(360, 300, 600) = 360; scale=1
    expect(spec.width).toBe(360);
    expect(spec.height).toBe(240);
  });

  it('caps at MAX_PANEL_WIDTH=600 when containerWidth exceeds the maximum', () => {
    const spec: Record<string, unknown> = { width: 360, height: 240 };
    fitSpec(spec, 1000);
    // target = clamp(1000, 300, 600) = 600; scale = 600/360
    expect(spec.width).toBe(600);
    expect(spec.height).toBe(400); // round(240 * 600/360)
  });

  it('scales the inner panel of a faceted spec to fit containerWidth / columns', () => {
    // Mirrors the real engine shape: `columns` nested INSIDE the facet object.
    // columns:2 (≠ the default 3, so the read path is genuinely exercised).
    // containerWidth=720 → rawPanelWidth = floor(720/2) - 40 = 320.
    // target = clamp(320, 300, 600) = 320; scale = 320/200 = 1.6
    // inner width = round(200*1.6) = 320, height = round(240*1.6) = 384.
    const spec: Record<string, unknown> = {
      facet: { field: 'scenario', columns: 2 },
      spec: { width: 200, height: 240 },
    };
    fitSpec(spec, 720);
    expect(spec.width).toBeUndefined();
    expect((spec.spec as Record<string, unknown>).width).toBe(320);
    expect((spec.spec as Record<string, unknown>).height).toBe(384);
  });

  it('faceted spec: falls back to columns=3 when columns is absent', () => {
    // No columns anywhere → defaults to 3.
    // containerWidth=720 → rawPanelWidth = floor(720/3) - 40 = 200.
    // same math as above → 300×360.
    const spec: Record<string, unknown> = {
      facet: { field: 'scenario' },
      spec: { width: 200, height: 240 },
    };
    fitSpec(spec, 720);
    expect((spec.spec as Record<string, unknown>).width).toBe(300);
    expect((spec.spec as Record<string, unknown>).height).toBe(360);
  });

  it('scales each non-faceted child of a vconcat spec (wrapper gets no width)', () => {
    const spec: Record<string, unknown> = {
      vconcat: [
        { width: 360, height: 240 },
        { width: 360, height: 240 },
      ],
    };
    fitSpec(spec, 480);
    // The wrapper itself must not acquire a width property.
    expect(spec.width).toBeUndefined();
    for (const child of spec.vconcat as Record<string, unknown>[]) {
      expect(child.width).toBe(480);
      expect(child.height).toBe(320);
    }
  });
});
