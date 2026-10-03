import { describe, it, expect, vi, beforeEach } from 'vitest';

// Per-family load/save uses a per-family Store instance — mock at module level.
const stores = new Map<
  string,
  { get: ReturnType<typeof vi.fn>; set: ReturnType<typeof vi.fn>; save: ReturnType<typeof vi.fn> }
>();

vi.mock('@tauri-apps/plugin-store', () => ({
  Store: {
    load: vi.fn(async (path: string) => {
      if (!stores.has(path)) {
        // Backed by a Map so save→load is a real round trip by key.
        const data = new Map<string, unknown>();
        stores.set(path, {
          get: vi.fn(async (k: string) => data.get(k)),
          set: vi.fn(async (k: string, v: unknown) => void data.set(k, v)),
          save: vi.fn(async () => undefined),
        });
      }
      return stores.get(path)!;
    }),
  },
}));

import { loadFamily, saveFamily, resetPerFamilyCache } from './per-family';
import { defaultFamilyConfig } from '$lib/domain/family';

describe('per-family persistence', () => {
  beforeEach(() => {
    stores.clear();
    resetPerFamilyCache();
  });

  it('saveFamily writes to per-family/<family>.json then save()', async () => {
    const cfg = defaultFamilyConfig('regression');
    await saveFamily('regression', cfg);
    const store = stores.get('per-family/regression.json');
    expect(store).toBeTruthy();
    expect(store!.set).toHaveBeenCalledWith('config', cfg);
    expect(store!.save).toHaveBeenCalled();
  });

  it('loadFamily returns null when nothing stored', async () => {
    const r = await loadFamily('mixed');
    expect(r).toBeNull();
  });

  it('loadFamily returns persisted config', async () => {
    const cfg = { ...defaultFamilyConfig('regression'), formula: 'y ~ x1 + x2' };
    await saveFamily('regression', cfg);
    expect(await loadFamily('regression')).toEqual(cfg);
    // Families keep separate files: another family's load does not see it.
    expect(await loadFamily('anova')).toBeNull();
  });
});
