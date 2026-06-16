import { describe, it, expect, vi, beforeEach } from 'vitest';

// Use paths that no other test file touches so the shared module-level cache
// never has stale entries from another test file's mock.
const PATH_A = '__test-tauri-store-a__.json';
const PATH_B = '__test-tauri-store-b__.json';

const mockStore = { get: vi.fn(), set: vi.fn(), save: vi.fn() };

vi.mock('@tauri-apps/plugin-store', () => ({
  Store: { load: vi.fn(async () => mockStore) },
}));

import { Store } from '@tauri-apps/plugin-store';
import { openTauriStore, invalidateTauriStore } from './tauri-store';

describe('tauri-store shared opener', () => {
  beforeEach(() => {
    vi.mocked(Store.load).mockClear();
    invalidateTauriStore(PATH_A);
    invalidateTauriStore(PATH_B);
  });

  it('memoizes one store promise per path', async () => {
    const a = openTauriStore(PATH_A);
    const b = openTauriStore(PATH_A);
    expect(a).toBe(b);
    await a;
    expect(vi.mocked(Store.load)).toHaveBeenCalledTimes(1);
  });

  it('different paths get different promises', async () => {
    // Open stores for two distinct paths sequentially so each dynamic import
    // lands in the already-cached mock module rather than racing with an
    // in-flight module load.
    const a = openTauriStore(PATH_A);
    await a;
    const b = openTauriStore(PATH_B);
    await b;

    // Still different Promise objects because the cache keys differ.
    expect(a).not.toBe(b);
    // Store.load was called once per distinct path — two in total.
    expect(vi.mocked(Store.load)).toHaveBeenCalledTimes(2);
  });

  it('invalidateTauriStore clears the cache for that path', async () => {
    const a = openTauriStore(PATH_A);
    await a;
    invalidateTauriStore(PATH_A);
    const b = openTauriStore(PATH_A);
    expect(a).not.toBe(b);
    await b;
  });
});
