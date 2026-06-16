// Browser persistence backend: an IndexedDB-backed StoreInstance matching the
// @tauri-apps/plugin-store surface the app uses (get/set/save). One DB, one
// object store; each logical "<path>" is one record holding that store's blob.
import type { StoreInstance } from './tauri-store';

const DB = 'mcpower';
const STORE = 'kv';

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function idb<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function openBrowserStore(path: string): Promise<StoreInstance> {
  const db = await openDb();
  const blob = (await idb(db.transaction(STORE, 'readonly').objectStore(STORE).get(path))) as
    | Record<string, unknown>
    | undefined;
  const data: Record<string, unknown> = blob ?? {};

  return {
    async get<T>(k: string): Promise<T | undefined> {
      return data[k] as T | undefined;
    },
    async set(k: string, v: unknown): Promise<void> {
      data[k] = v;
    },
    async save(): Promise<void> {
      await idb(db.transaction(STORE, 'readwrite').objectStore(STORE).put(data, path));
    },
  };
}
