/**
 * Persist run history (max 25 entries) to `history.json` under app_data_dir.
 *
 * - Load once on first access via `loadHistory()`.
 * - New entries prepended (newest-first); FIFO eviction beyond MAX_ENTRIES.
 * - Tests mock `@tauri-apps/plugin-store` because there's no IPC in jsdom.
 */
import type { AppSpec } from '$lib/domain/app-spec';
import type { PowerResult, SampleSizeResult, SampleSizeMethod } from '$lib/domain/result';
import { openTauriStore, invalidateTauriStore } from './tauri-store';
import { reportError, errorDetail } from '$lib/errors/report';

export interface HistoryEntry {
  id: string;
  ts: number;
  /** Wire family from AppSpec — 'linear' | 'logit'. Not the UI entrypoint. */
  family: AppSpec['family'];
  kind: 'find-power' | 'find-sample-size';
  spec: AppSpec;
  sample_size?: number;
  bounds?: [number, number];
  method?: SampleSizeMethod;
  effect_names: string[];
  result: PowerResult | SampleSizeResult;
}

const FILE = 'history.json';
const MAX_ENTRIES = 25;

/** Exposed for tests to reset the module-level cache between test runs. */
export function resetHistoryCache(): void {
  invalidateTauriStore(FILE);
}

export async function loadHistory(): Promise<HistoryEntry[]> {
  try {
    const store = await openTauriStore(FILE);
    return (await store.get<HistoryEntry[]>('entries')) ?? [];
  } catch (err) {
    console.warn('loadHistory failed:', err);
    return [];
  }
}

export async function saveHistory(entries: HistoryEntry[]): Promise<void> {
  try {
    const store = await openTauriStore(FILE);
    await store.set('entries', entries);
    await store.save();
  } catch (err) {
    console.warn('saveHistory failed:', err);
    reportError({
      severity: 'background',
      title: "Couldn't save run history",
      message: 'This run may not appear in History next time.',
      detail: errorDetail(err),
    });
  }
}

export async function pushHistoryEntry(entry: HistoryEntry): Promise<HistoryEntry[]> {
  // loadHistory returns oldest-first (append order); we append the new entry,
  // then evict from the front (oldest) when over MAX_ENTRIES.
  const current = await loadHistory();
  const appended = [...current, entry];
  const trimmed = appended.length > MAX_ENTRIES ? appended.slice(appended.length - MAX_ENTRIES) : appended;
  await saveHistory(trimmed);
  // Return newest-first for immediate consumption by historyStore / callers.
  return [...trimmed].reverse();
}

export async function removeHistoryEntry(id: string): Promise<HistoryEntry[]> {
  const current = await loadHistory();
  const updated = current.filter((e) => e.id !== id);
  await saveHistory(updated);
  return updated;
}

export async function clearHistory(): Promise<void> {
  try {
    const store = await openTauriStore(FILE);
    await store.set('entries', []);
    await store.save();
  } catch (err) {
    console.warn('clearHistory failed:', err);
    reportError({
      severity: 'background',
      title: "Couldn't clear run history",
      message: 'History may still show previous runs.',
      detail: errorDetail(err),
    });
  }
}
