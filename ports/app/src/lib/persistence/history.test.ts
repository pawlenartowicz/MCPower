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
    // Real specs carry `csv: null` when nothing is uploaded.
    spec: { csv: null } as any,
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

  const upload = (id: string) =>
    ({ ...entry(id), spec: { family: 'linear', csv: { mode: 'none', n_rows: 50, columns: [] } } as any }) as HistoryEntry;
  const stored = (entries: HistoryEntry[]) =>
    mockStore.get.mockImplementation(async (k: string) => (k === 'entries' ? entries : undefined));

  it('a push without uploaded data appends', async () => {
    stored([entry('a')]);
    const out = await pushHistoryEntry(entry('b'));
    expect(out.map((e) => e.id)).toEqual(['b', 'a']);
    expect(mockStore.set).toHaveBeenCalledWith(
      'entries',
      expect.arrayContaining([expect.objectContaining({ id: 'b' })]),
    );
  });

  // An upload run's spec carries the whole dataset; History does not keep it.
  it('a push of a run on uploaded data leaves History unchanged', async () => {
    stored([entry('a'), entry('b')]);
    const out = await pushHistoryEntry(upload('u'));
    expect(out.map((e) => e.id)).toEqual(['b', 'a']);
    expect(mockStore.set).not.toHaveBeenCalled();
    expect(mockStore.save).not.toHaveBeenCalled();
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
