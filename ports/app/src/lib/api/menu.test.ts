import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock the engine so route handlers that trigger runs don't hit @tauri-apps.
vi.mock('$lib/api/engine', () => ({
  findPower: vi.fn(async () => ({ run_id: 'r1', result: { scenarios: [] } })),
  findSampleSize: vi.fn(async () => ({ run_id: 'r1', result: { scenarios: [] } })),
  cancelRun: vi.fn(async () => true),
  onProgress: vi.fn(async () => () => {}),
}));

// Mock spec adaptation so the run-shortcut tests exercise routing only, without
// the adapter's async formula-parsing side effects.
vi.mock('$lib/domain/app-spec-adapter', () => ({
  familyConfigToAppSpec: vi.fn(),
}));

import { routeMenuEvent } from './menu';
import { uiStore } from '$lib/stores/ui.svelte';
import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';
import { runStore } from '$lib/stores/run.svelte';
import { familyStore } from '$lib/stores/family.svelte';
import { familyConfigToAppSpec } from '$lib/domain/app-spec-adapter';
import { DOCS_BASE_URL } from '$lib/content/render-doc';

describe('routeMenuEvent — fallthrough regression coverage', () => {
  beforeEach(() => {
    uiStore.settingsOpen = false;
    uiStore.historyOpen = false;
    uiStore.acknowledgmentsOpen = false;
    uiStore.resetConfirmOpen = false;
    sharedPrefs.activePane = 'config';
  });

  it('view.settings opens settings only', () => {
    routeMenuEvent('view.settings');
    expect(uiStore.settingsOpen).toBe(true);
    expect(uiStore.historyOpen).toBe(false);
  });

  it('view.history opens history only', () => {
    routeMenuEvent('view.history');
    expect(uiStore.historyOpen).toBe(true);
    expect(uiStore.settingsOpen).toBe(false);
  });

  it('view.toggle_config sets activePane=config', () => {
    sharedPrefs.activePane = 'results';
    routeMenuEvent('view.toggle_config');
    expect(sharedPrefs.activePane).toBe('config');
  });

  it('view.toggle_results sets activePane=results', () => {
    routeMenuEvent('view.toggle_results');
    expect(sharedPrefs.activePane).toBe('results');
  });

  it('edit.reset_family opens reset confirm only', () => {
    routeMenuEvent('edit.reset_family');
    expect(uiStore.resetConfirmOpen).toBe(true);
    expect(uiStore.settingsOpen).toBe(false);
  });

  it('help.documentation opens the tutorial in the system browser', () => {
    const openSpy = vi.spyOn(window, 'open').mockReturnValue(null);
    routeMenuEvent('help.documentation');
    expect(openSpy).toHaveBeenCalledWith(
      `${DOCS_BASE_URL}/tutorial-app/index`,
      '_blank',
      expect.any(String),
    );
    openSpy.mockRestore();
  });

  it('help.acknowledgments opens ack only', () => {
    routeMenuEvent('help.acknowledgments');
    expect(uiStore.acknowledgmentsOpen).toBe(true);
  });

  it('file.new resets the active family (no overlay flips)', () => {
    familyStore.active = 'regression';
    familyStore.byFamily.regression.formula = 'y ~ x';
    routeMenuEvent('file.new');
    expect(familyStore.byFamily.regression.formula).toBe('');
    expect(uiStore.settingsOpen).toBe(false);
  });

  it('unknown id logs no-op without throwing', () => {
    const spy = vi.spyOn(console, 'log').mockImplementation(() => {});
    expect(() => routeMenuEvent('unknown.event')).not.toThrow();
    expect(spy).toHaveBeenCalledWith(expect.stringContaining('no-op'));
    spy.mockRestore();
  });
});

describe('routeMenuEvent — run shortcuts', () => {
  it('run.cancel calls runStore.cancel', () => {
    const spy = vi.spyOn(runStore, 'cancel');
    routeMenuEvent('run.cancel');
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it('run.rerun calls runStore.replayLast', () => {
    const spy = vi.spyOn(runStore, 'replayLast');
    routeMenuEvent('run.rerun');
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it('run.find_power drives the active family, not just regression', () => {
    // Mirrors StatusBar's Run button: a wired non-regression family runs too.
    familyStore.active = 'mixed';
    vi.mocked(familyConfigToAppSpec).mockReturnValueOnce({
      spec: { family: 'mixed' },
      errors: [],
      warnings: [],
    } as any);
    const spy = vi.spyOn(runStore, 'startFindPower').mockResolvedValue(undefined);
    routeMenuEvent('run.find_power');
    expect(familyConfigToAppSpec).toHaveBeenCalledWith('mixed', expect.anything(), expect.anything());
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
    familyStore.active = 'regression';
  });

  it('run.find_n drives the active family, not just regression', () => {
    familyStore.active = 'anova';
    vi.mocked(familyConfigToAppSpec).mockReturnValueOnce({
      spec: { family: 'anova' },
      errors: [],
      warnings: [],
    } as any);
    const spy = vi.spyOn(runStore, 'startFindSampleSize').mockResolvedValue(undefined);
    routeMenuEvent('run.find_n');
    expect(familyConfigToAppSpec).toHaveBeenCalledWith('anova', expect.anything(), expect.anything());
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
    familyStore.active = 'regression';
  });
});

describe('routeMenuEvent — export results', () => {
  it('file.export_results route exists and does not throw', () => {
    // jsdom has no @tauri-apps; the dynamic import will fail but the error is
    // caught and logged. The router itself should not throw.
    expect(() => routeMenuEvent('file.export_results')).not.toThrow();
  });
});
