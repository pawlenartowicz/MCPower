/**
 * Persist the user-editable scenarios app copy to `scenarios.json` under app_data_dir.
 *
 * Mirrors `settings.ts`: lazy single-shot store load, debounce in caller.
 */
import type { ScenarioConfig } from '$lib/configs/scenarios';
import { openTauriStore, invalidateTauriStore } from './tauri-store';
import { reportError, errorDetail } from '$lib/errors/report';

const FILE = 'scenarios.json';

/** Test-only cache reset. */
export function resetScenariosCache(): void {
  invalidateTauriStore(FILE);
}

export async function loadScenarios(): Promise<ScenarioConfig[] | null> {
  try {
    const store = await openTauriStore(FILE);
    const snap = await store.get<ScenarioConfig[]>('scenarios');
    if (!snap || !Array.isArray(snap)) return null;
    // Reseed if a pre-lme device copy is detected (lacking the lme sub-object).
    return snap.every((s) => s.lme) ? snap : null;
  } catch (err) {
    console.warn('loadScenarios failed:', err);
    return null;
  }
}

export async function saveScenarios(scenarios: ScenarioConfig[]): Promise<void> {
  try {
    const store = await openTauriStore(FILE);
    await store.set('scenarios', scenarios);
    await store.save();
  } catch (err) {
    console.warn('saveScenarios failed:', err);
    reportError({
      severity: 'background',
      title: "Couldn't save scenarios",
      message: 'Your scenario edits may not persist between sessions.',
      detail: errorDetail(err),
    });
  }
}
