import { beforeEach, describe, expect, it, vi } from 'vitest';

// Capture every config handed to the persistence layer so we can assert it is
// structured-clone-safe. The wasm backend persists via IndexedDB (structured
// clone), which throws DataCloneError on a Svelte $state proxy; $state.snapshot
// strips it. vi.mock is hoisted above the store import so the singleton binds
// the mock.
const saved: unknown[] = [];
vi.mock('$lib/persistence/per-family', () => ({
  loadFamily: vi.fn(async () => null),
  saveFamily: vi.fn(async (_family: unknown, config: unknown) => {
    saved.push(config);
  }),
  resetPerFamilyCache: vi.fn(),
}));

const { familyStore } = await import('./family.svelte');

describe('familyStore — persisted config carries no reactive proxy', () => {
  beforeEach(() => {
    saved.length = 0;
  });

  it('hands the persistence layer a structured-cloneable config (switch + reset)', () => {
    familyStore.active = 'anova'; // persists the outgoing 'regression'
    familyStore.resetActive(); // persists the reset 'anova'
    familyStore.resetAll(); // persists all three families

    expect(saved.length).toBeGreaterThan(0);
    // structuredClone throws DataCloneError on any Proxy (incl. a Svelte $state
    // proxy) — this is the guard against the web DataCloneError crash.
    for (const config of saved) {
      expect(() => structuredClone(config)).not.toThrow();
    }
  });
});
