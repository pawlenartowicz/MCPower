// History store: reactive wrapper around the persistent run-history log, exposing entries in newest-first order.
import {
  loadHistory,
  pushHistoryEntry,
  removeHistoryEntry,
  clearHistory,
  type HistoryEntry,
} from '$lib/persistence/history';

function createHistoryStore() {
  let entries = $state<HistoryEntry[]>([]);
  let ready = $state<boolean>(false);

  void loadHistory()
    .then((loaded) => {
      // loadHistory returns oldest-first (storage order); reverse for newest-first UI.
      entries = [...loaded].reverse();
    })
    .finally(() => {
      ready = true;
    });

  return {
    get entries() {
      return entries;
    },
    get ready() {
      return ready;
    },
    async push(entry: HistoryEntry) {
      // $state.snapshot before persisting — see family.svelte.ts for why (DataCloneError).
      entries = await pushHistoryEntry($state.snapshot(entry) as HistoryEntry);
    },
    async remove(id: string) {
      // removeHistoryEntry returns oldest-first; reverse for newest-first UI.
      entries = [...(await removeHistoryEntry(id))].reverse();
    },
    async clear() {
      await clearHistory();
      entries = [];
    },
  };
}

export const historyStore = createHistoryStore();
