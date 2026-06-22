import { describe, it, expect, vi, beforeEach } from 'vitest';

const mockStore = {
  get: vi.fn(),
  set: vi.fn(),
  save: vi.fn(),
};

vi.mock('@tauri-apps/plugin-store', () => ({
  Store: { load: vi.fn(async () => mockStore) },
}));

import {
  loadHistory,
  pushHistoryEntry,
  removeHistoryEntry,
  clearHistory,
  resetHistoryCache,
  type HistoryEntry,
} from './history';

function entry(id: string): HistoryEntry {
  return {
    id,
    ts: Date.now(),
    family: 'linear',
    kind: 'find-power',
    spec: {} as any,
    sample_size: 80,
    effect_names: [],
    result: {} as any,
  };
}

describe('history persistence', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockStore.get.mockReset();
    mockStore.set.mockReset();
    mockStore.save.mockReset();
    resetHistoryCache();
  });

  it('loadHistory returns empty when nothing stored', async () => {
    mockStore.get.mockResolvedValueOnce(undefined);
    expect(await loadHistory()).toEqual([]);
  });

  it('pushHistoryEntry caps at 25 entries (FIFO eviction of oldest)', async () => {
    const existing: HistoryEntry[] = Array.from({ length: 25 }, (_, i) => entry(`old-${i}`));
    mockStore.get.mockResolvedValueOnce(existing);
    const out = await pushHistoryEntry(entry('new'));
    expect(out.length).toBe(25);
    expect(out[0]?.id).toBe('new'); // newest first
    expect(out.find((e) => e.id === 'old-0')).toBeFalsy(); // oldest evicted
  });

  it('removeHistoryEntry drops by id', async () => {
    mockStore.get.mockResolvedValueOnce([entry('a'), entry('b'), entry('c')]);
    const out = await removeHistoryEntry('b');
    expect(out.map((e) => e.id)).toEqual(['a', 'c']);
  });

  it('clearHistory empties the list', async () => {
    await clearHistory();
    expect(mockStore.set).toHaveBeenCalledWith('entries', []);
    expect(mockStore.save).toHaveBeenCalled();
  });
});
