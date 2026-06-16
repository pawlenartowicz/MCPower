/**
 * Shared lazy Store opener for single-file persistence modules.
 *
 * - `openTauriStore(path)` returns a memoized promise so at most one Store
 *   instance is created per path (matching the previous per-module pattern).
 * - `invalidateTauriStore(path)` drops the cached promise; used by module-level
 *   reset functions (`resetSettingsCache`, etc.) so tests can clear state
 *   between runs.
 * - `per-family.ts` is NOT served by this helper — its opener is keyed per
 *   Entrypoint and clears ALL families at once, which a path-keyed cache
 *   cannot reproduce.
 */
export type StoreInstance = {
  get<T>(k: string): Promise<T | undefined>;
  set(k: string, v: unknown): Promise<void>;
  save(): Promise<void>;
};

const cache = new Map<string, Promise<StoreInstance>>();

export function openTauriStore(path: string): Promise<StoreInstance> {
  let p = cache.get(path);
  if (!p) {
    p =
      import.meta.env.VITE_TARGET === 'wasm'
        ? import('./browser-store').then((m) => m.openBrowserStore(path))
        : (async () => {
            const { Store } = await import('@tauri-apps/plugin-store');
            return Store.load(path);
          })();
    cache.set(path, p);
  }
  return p;
}

export function invalidateTauriStore(path: string): void {
  cache.delete(path);
}
