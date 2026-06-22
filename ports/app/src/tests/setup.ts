import '@testing-library/jest-dom/vitest';
import { afterEach, vi } from 'vitest';

// bits-ui Dialog/Popover lock body scroll while open and, on unmount, schedule a
// ~24ms setTimeout to restore the body style (bits-ui body-scroll-lock.svelte).
// testing-library auto-unmounts after each test; if the file then finishes
// before that timer fires, vitest tears down jsdom and the timer throws
// "document is not defined" — an uncaught error that fails the run even though
// every test passed. When a lock is still pending (it sets --scrollbar-width on
// <body>), let the timer fire here, while document still exists. Gated so only
// the handful of overlay tests pay the delay.
afterEach(async () => {
  if (document.body.style.getPropertyValue('--scrollbar-width')) {
    await new Promise((resolve) => setTimeout(resolve, 30));
  }
});

vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(async () => {
    throw new Error('invoke() must not be called in tests');
  }),
  // Tests run in jsdom, not a Tauri webview: report not-Tauri so openExternal()
  // takes its window.open() branch instead of the desktop opener plugin.
  isTauri: () => false,
}));
vi.mock('@tauri-apps/api/event', () => ({
  listen: vi.fn(async () => () => {}),
  emit: vi.fn(async () => {}),
}));

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  }),
});

globalThis.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as unknown as typeof ResizeObserver;

// Stub HTMLCanvasElement.getContext and Path2D so any canvas consumer
// doesn't throw in jsdom, which has no real 2D canvas implementation.
const noopCtx: Partial<CanvasRenderingContext2D> = {
  arc: () => {},
  beginPath: () => {},
  clearRect: () => {},
  clip: () => {},
  closePath: () => {},
  createLinearGradient: () => ({} as CanvasGradient),
  drawImage: () => {},
  fill: () => {},
  fillRect: () => {},
  fillText: () => {},
  lineTo: () => {},
  measureText: () => ({ width: 0 } as TextMetrics),
  moveTo: () => {},
  rect: () => {},
  restore: () => {},
  save: () => {},
  setLineDash: () => {},
  setTransform: () => {},
  stroke: () => {},
  strokeRect: () => {},
  strokeText: () => {},
  translate: () => {},
};
// eslint-disable-next-line @typescript-eslint/no-explicit-any
(HTMLCanvasElement.prototype as any).getContext = () => noopCtx;

// jsdom also lacks Path2D (used by some canvas consumers).
if (typeof (globalThis as Record<string, unknown>)['Path2D'] === 'undefined') {
  (globalThis as Record<string, unknown>)['Path2D'] = class {
    constructor(_d?: string) {}
    addPath() {}
    closePath() {}
    moveTo() {}
    lineTo() {}
    arc() {}
    arcTo() {}
    bezierCurveTo() {}
    quadraticCurveTo() {}
    rect() {}
  };
}
