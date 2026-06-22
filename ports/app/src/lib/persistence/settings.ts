/**
 * Persist sharedPrefs to `settings.json` under app_data_dir.
 * Mirrors scenarios.ts: lazy single-shot store load, debounce in caller.
 *
 * - Load once on first access via `loadSettings()`.
 * - Write changes via `saveSettings(snapshot)`; callers should debounce.
 * - Tests mock `@tauri-apps/plugin-store` because there's no IPC in jsdom.
 */
import type { Theme } from '$lib/stores/shared-prefs.svelte';
import type { ScriptLanguage } from '$lib/domain/script-generator';
import { openTauriStore, invalidateTauriStore } from './tauri-store';
import { reportError, errorDetail } from '$lib/errors/report';

export interface SharedPrefsSnapshot {
  theme: Theme;
  fontSize: number;
  splitterFraction: number;
  modelExpanded: boolean;
  runExpanded: boolean;
  correlationsExpanded: boolean;
  uploadExpanded: boolean;
  activePane: 'config' | 'results';
  scenariosEnabled: boolean;
  scriptLanguage: ScriptLanguage;
  // Tauri-only: rayon thread pool size (null = all cores).
  nThreads: number | null;
}

const FILE = 'settings.json';

/** Exposed for tests to reset the module-level cache between test runs. */
export function resetSettingsCache(): void {
  invalidateTauriStore(FILE);
}

export async function loadSettings(): Promise<Partial<SharedPrefsSnapshot>> {
  try {
    const store = await openTauriStore(FILE);
    const snap = await store.get<Partial<SharedPrefsSnapshot>>('snapshot');
    return snap ?? {};
  } catch (err) {
    console.warn('loadSettings failed:', err);
    return {};
  }
}

export async function saveSettings(snapshot: SharedPrefsSnapshot): Promise<void> {
  try {
    const store = await openTauriStore(FILE);
    await store.set('snapshot', snapshot);
    await store.save();
  } catch (err) {
    console.warn('saveSettings failed:', err);
    reportError({
      severity: 'background',
      title: "Couldn't save settings",
      message: 'Your latest preference change may not persist.',
      detail: errorDetail(err),
    });
  }
}
