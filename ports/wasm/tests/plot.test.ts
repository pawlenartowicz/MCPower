import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { plotHtml, downloadPlot } from '../src/index';

// A minimal theme-naked Vega-Lite spec (no config block).
const NAKED_SPEC = '{"$schema":"https://vega.github.io/schema/vega-lite/v5.json","mark":"point","data":{"values":[]}}';
// A spec that already has a consumer config key not in the light-print theme.
const SPEC_WITH_CONFIG = '{"$schema":"https://vega.github.io/schema/vega-lite/v5.json","mark":"point","data":{"values":[]},"config":{"customKey":"survives"}}';

describe('plotHtml', () => {
  it('returns a string containing the DOCTYPE declaration', () => {
    const html = plotHtml(NAKED_SPEC);
    expect(html).toContain('<!doctype html>');
  });

  it('loads vega@5, vega-lite@5, vega-embed@6 from jsdelivr CDN', () => {
    const html = plotHtml(NAKED_SPEC);
    expect(html).toContain('https://cdn.jsdelivr.net/npm/vega@5');
    expect(html).toContain('https://cdn.jsdelivr.net/npm/vega-lite@5');
    expect(html).toContain('https://cdn.jsdelivr.net/npm/vega-embed@6');
  });

  it('substitutes {{SPECS}} fully — the literal placeholder is absent', () => {
    const html = plotHtml(NAKED_SPEC);
    expect(html).not.toContain('{{SPECS}}');
  });

  it('uses the multi-spec array template (scaleFactor, forEach loop)', () => {
    const html = plotHtml(NAKED_SPEC);
    // Template-driven: scaleFactor:2 and the forEach loop are present.
    expect(html).toContain('scaleFactor');
    expect(html).toContain('forEach');
  });

  it('applies the light-print theme: background #ffffff is present in the output', () => {
    const html = plotHtml(NAKED_SPEC);
    expect(html).toContain('"background":"#ffffff"');
  });

  it('applies the light-print theme: legend block is present in the output', () => {
    const html = plotHtml(NAKED_SPEC);
    expect(html).toContain('"legend"');
    expect(html).toContain('"labelColor":"#000000"');
  });

  it('deep-merge: consumer config keys not in the light-print theme survive', () => {
    const html = plotHtml(SPEC_WITH_CONFIG);
    expect(html).toContain('"customKey":"survives"');
  });

  it('deep-merge: light-print theme keys land even when spec already has a config', () => {
    const html = plotHtml(SPEC_WITH_CONFIG);
    expect(html).toContain('"background":"#ffffff"');
  });

  it('escapes "</script>" inside the spec payload so it cannot break out of the script block', () => {
    // A spec whose data would carry a closing script tag.
    const spec = '{"mark":"point","data":{"values":[{"x":"a</script>b"}]}}';
    const html = plotHtml(spec);
    // The raw unescaped closing tag must NOT appear in the rendered SPECS payload.
    // (The template's own </script> tags are fine.)
    // Count </script> occurrences: template has exactly 4 (3 CDN + 1 closing embed block).
    const count = (html.match(/<\/script>/g) ?? []).length;
    expect(count).toBe(4);
    // The escaped form is present.
    expect(html).toContain('<\\/script>');
  });
});

describe('downloadPlot', () => {
  let clickSpy: ReturnType<typeof vi.fn>;
  let appendChildSpy: ReturnType<typeof vi.fn>;
  let removeChildSpy: ReturnType<typeof vi.fn>;
  let createElementSpy: ReturnType<typeof vi.fn>;
  let createObjectURLSpy: ReturnType<typeof vi.fn>;
  let revokeObjectURLSpy: ReturnType<typeof vi.fn>;
  let capturedBlob: { content: string; options: { type: string } } | null = null;

  beforeEach(() => {
    capturedBlob = null;
    clickSpy = vi.fn();
    appendChildSpy = vi.fn();
    removeChildSpy = vi.fn();

    const mockAnchor = { href: '', download: '', click: clickSpy };
    createElementSpy = vi.fn().mockReturnValue(mockAnchor);

    createObjectURLSpy = vi.fn().mockReturnValue('blob:mock-url');
    revokeObjectURLSpy = vi.fn();

    // Mock Blob globally
    (globalThis as Record<string, unknown>).Blob = class MockBlob {
      constructor(parts: string[], options: { type: string }) {
        capturedBlob = { content: parts.join(''), options };
      }
    };

    // Mock URL globally
    (globalThis as Record<string, unknown>).URL = {
      createObjectURL: createObjectURLSpy,
      revokeObjectURL: revokeObjectURLSpy,
    };

    // Mock document globally
    (globalThis as Record<string, unknown>).document = {
      createElement: createElementSpy,
      body: { appendChild: appendChildSpy, removeChild: removeChildSpy },
    };
  });

  afterEach(() => {
    delete (globalThis as Record<string, unknown>).Blob;
    delete (globalThis as Record<string, unknown>).URL;
    delete (globalThis as Record<string, unknown>).document;
  });

  it('creates a Blob with the spec content and application/json type', () => {
    downloadPlot(NAKED_SPEC, 'power.json');
    expect(capturedBlob).not.toBeNull();
    expect(capturedBlob!.content).toBe(NAKED_SPEC);
    expect(capturedBlob!.options.type).toBe('application/json');
  });

  it('sets href to the object URL and download to the filename', () => {
    downloadPlot(NAKED_SPEC, 'power.json');
    const anchor = createElementSpy.mock.results[0]?.value as { href: string; download: string };
    expect(anchor.href).toBe('blob:mock-url');
    expect(anchor.download).toBe('power.json');
  });

  it('appends the anchor, clicks it, then removes it and revokes the URL', () => {
    downloadPlot(NAKED_SPEC, 'power.json');
    expect(appendChildSpy).toHaveBeenCalledOnce();
    expect(clickSpy).toHaveBeenCalledOnce();
    expect(removeChildSpy).toHaveBeenCalledOnce();
    expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:mock-url');
  });

  it('passes the filename through unchanged', () => {
    downloadPlot(NAKED_SPEC, 'my-scenario-plot.json');
    const anchor = createElementSpy.mock.results[0]?.value as { download: string };
    expect(anchor.download).toBe('my-scenario-plot.json');
  });
});
