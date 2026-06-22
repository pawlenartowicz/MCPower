/**
 * Persist each family's FamilyConfig to `per-family/<family>.json` under app_data_dir.
 *
 * - One Store per family file keeps each snapshot bounded.
 * - Load on demand via `loadFamily(family)`.
 * - Write via `saveFamily(family, config)`.
 * - Tests mock `@tauri-apps/plugin-store` because there's no IPC in jsdom.
 */
import type { Entrypoint, FamilyConfig } from '$lib/domain/family';
import { reportError, errorDetail } from '$lib/errors/report';

const FILE_PREFIX = 'per-family/';
const FILE_SUFFIX = '.json';

type StoreInstance = {
  get<T>(k: string): Promise<T | undefined>;
  set(k: string, v: unknown): Promise<void>;
  save(): Promise<void>;
};

const storeCache = new Map<Entrypoint, Promise<StoreInstance>>();

async function openStore(family: Entrypoint): Promise<StoreInstance> {
  if (!storeCache.has(family)) {
    const path = `${FILE_PREFIX}${family}${FILE_SUFFIX}`;
    const p = (async () => {
      if (import.meta.env.VITE_TARGET === 'wasm') {
        const { openBrowserStore } = await import('./browser-store');
        return openBrowserStore(path);
      }
      const { Store } = await import('@tauri-apps/plugin-store');
      return Store.load(path);
    })();
    storeCache.set(family, p);
  }
  return storeCache.get(family)!;
}

/** Exposed for tests to reset the module-level cache between test runs. */
export function resetPerFamilyCache(): void {
  storeCache.clear();
}

export async function loadFamily(family: Entrypoint): Promise<FamilyConfig | null> {
  try {
    const store = await openStore(family);
    return (await store.get<FamilyConfig>('config')) ?? null;
  } catch (err) {
    console.warn(`loadFamily(${family}) failed:`, err);
    return null;
  }
}

export async function saveFamily(family: Entrypoint, config: FamilyConfig): Promise<void> {
  try {
    const store = await openStore(family);
    await store.set('config', config);
    await store.save();
  } catch (err) {
    console.warn(`saveFamily(${family}) failed:`, err);
    reportError({
      severity: 'background',
      title: "Couldn't save model configuration",
      message: 'Your latest changes may not persist between sessions.',
      detail: errorDetail(err),
    });
  }
}
