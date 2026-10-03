/**
 * Unit tests for settings persistence: loadSettings/saveSettings, plus the
 * shared-prefs rune store's hydrate and debounced save.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

const mockGet = vi.fn();
const mockSet = vi.fn();
const mockSave = vi.fn();
const mockLoad = vi.fn(async () => ({ get: mockGet, set: mockSet, save: mockSave }));

vi.mock('@tauri-apps/plugin-store', () => ({
  Store: { load: mockLoad },
}));

vi.mock('$lib/errors/report', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/errors/report')>();
  return { ...actual, reportError: vi.fn() };
});

import { loadSettings, saveSettings, resetSettingsCache } from './settings';
import { reportError } from '$lib/errors/report';

describe('settings persistence', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGet.mockReset();
    mockSet.mockReset();
    mockSave.mockReset();
    mockLoad.mockImplementation(async () => ({ get: mockGet, set: mockSet, save: mockSave }));
    resetSettingsCache();
  });

  it('loadSettings returns empty when store has no snapshot', async () => {
    mockGet.mockResolvedValueOnce(undefined);
    const out = await loadSettings();
    expect(out).toEqual({});
  });

  it('loadSettings round-trips a snapshot', async () => {
    const snap = { theme: 'dark' };
    mockGet.mockResolvedValueOnce(snap);
    const out = await loadSettings();
    expect(out).toEqual(snap);
    expect(mockGet).toHaveBeenCalledWith('snapshot');
  });

  it('saveSettings calls set then save', async () => {
    mockSet.mockResolvedValueOnce(undefined);
    mockSave.mockResolvedValueOnce(undefined);
    await saveSettings({
      theme: 'light',
      fontSize: 14,
      splitterFraction: 0.33,
      modelExpanded: true,
      runExpanded: false,
      correlationsExpanded: false,
      uploadExpanded: false,
      activePane: 'config',
      scenariosEnabled: false,
      scriptLanguage: 'python',
      nThreads: null,
    });
    expect(mockSet).toHaveBeenCalledWith('snapshot', expect.objectContaining({ theme: 'light' }));
    expect(mockSave).toHaveBeenCalled();
  });

  it('sharedPrefs hydrates nThreads from the snapshot and saves a change back to null (all cores)', async () => {
    mockGet.mockResolvedValue({ nThreads: 4 });
    mockSet.mockResolvedValue(undefined);
    mockSave.mockResolvedValue(undefined);
    // A fresh store instance, so it hydrates from the snapshot mocked above.
    vi.resetModules();
    const { sharedPrefs } = await import('$lib/stores/shared-prefs.svelte');
    await vi.waitFor(() => expect(sharedPrefs.ready).toBe(true));
    expect(sharedPrefs.nThreads).toBe(4);

    sharedPrefs.nThreads = null;
    await vi.waitFor(() =>
      expect(mockSet).toHaveBeenCalledWith('snapshot', expect.objectContaining({ nThreads: null })),
    );
  });

  it('reports a background error when the store save fails (instead of silently losing it)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    mockSet.mockResolvedValueOnce(undefined);
    mockSave.mockRejectedValueOnce(new Error('disk full'));
    await saveSettings({
      theme: 'light',
      fontSize: 14,
      splitterFraction: 0.33,
      modelExpanded: true,
      runExpanded: false,
      correlationsExpanded: false,
      uploadExpanded: false,
      activePane: 'config',
      scenariosEnabled: false,
      scriptLanguage: 'python',
      nThreads: null,
    });
    expect(reportError).toHaveBeenCalledWith(expect.objectContaining({ severity: 'background' }));
    warn.mockRestore();
  });
});
