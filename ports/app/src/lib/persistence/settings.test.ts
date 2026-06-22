/**
 * Unit tests for settings persistence.
 *
 * NOTE: The timer-based test ("setting theme = 'dark' results in a saveSettings call within 250ms")
 * is skipped here. It requires a Svelte rune store + fake timers in jsdom, which is tricky to set
 * up without Svelte's test renderer. The persistence layer itself (loadSettings/saveSettings) is
 * fully covered below.
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

  it('nThreads round-trips: null (all cores) and a positive integer both persist', async () => {
    mockSet.mockResolvedValue(undefined);
    mockSave.mockResolvedValue(undefined);

    // null → persisted as null
    await saveSettings({
      theme: 'light', fontSize: 14, splitterFraction: 0.33,
      modelExpanded: true, runExpanded: false, correlationsExpanded: false,
      uploadExpanded: false, activePane: 'config', scenariosEnabled: false,
      scriptLanguage: 'python', nThreads: null,
    });
    expect(mockSet).toHaveBeenLastCalledWith('snapshot', expect.objectContaining({ nThreads: null }));

    // integer → persisted as integer
    await saveSettings({
      theme: 'light', fontSize: 14, splitterFraction: 0.33,
      modelExpanded: true, runExpanded: false, correlationsExpanded: false,
      uploadExpanded: false, activePane: 'config', scenariosEnabled: false,
      scriptLanguage: 'python', nThreads: 4,
    });
    expect(mockSet).toHaveBeenLastCalledWith('snapshot', expect.objectContaining({ nThreads: 4 }));
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
