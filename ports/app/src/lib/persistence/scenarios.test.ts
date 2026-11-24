/**
 * Unit tests for loadScenarios: a device copy is kept only when it matches the
 * bundled shape (same preset names, lme block present); otherwise it reseeds.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

const mockGet = vi.fn();
const mockLoad = vi.fn(async () => ({ get: mockGet, set: vi.fn(), save: vi.fn() }));

vi.mock('@tauri-apps/plugin-store', () => ({
  Store: { load: mockLoad },
}));

import { loadScenarios, resetScenariosCache } from './scenarios';
import { BUNDLED_SCENARIOS } from '$lib/configs/scenarios-bundled';

describe('loadScenarios', () => {
  beforeEach(() => {
    mockGet.mockReset();
    resetScenariosCache();
  });

  it('keeps a device copy with the bundled preset names', async () => {
    mockGet.mockResolvedValueOnce(BUNDLED_SCENARIOS);
    expect(await loadScenarios()).toEqual(BUNDLED_SCENARIOS);
  });

  it('reseeds a device copy saved under an old preset name', async () => {
    const old = BUNDLED_SCENARIOS.map((s) =>
      s.name === 'pessimistic' ? { ...s, name: 'doomer' } : s,
    );
    mockGet.mockResolvedValueOnce(old);
    expect(await loadScenarios()).toBeNull();
  });
});
