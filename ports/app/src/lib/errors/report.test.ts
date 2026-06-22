import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('svelte-sonner', () => ({ toast: { error: vi.fn() } }));

import { errorStore } from '$lib/stores/error.svelte';
import { reportError, toRunError, type AppError } from './report';

describe('reportError routing', () => {
  beforeEach(() => {
    errorStore.crash = null;
    vi.restoreAllMocks();
  });

  it("routes 'background' to errorStore.toast", () => {
    const spy = vi.spyOn(errorStore, 'toast').mockImplementation(() => {});
    const e: AppError = { severity: 'background', title: 'A', message: 'b' };
    reportError(e);
    expect(spy).toHaveBeenCalledWith(e);
    expect(errorStore.crash).toBeNull();
  });

  it("routes 'crash' to errorStore.crash", () => {
    const spy = vi.spyOn(errorStore, 'toast').mockImplementation(() => {});
    const e: AppError = { severity: 'crash', title: 'Boom', message: 'x' };
    reportError(e);
    expect(errorStore.crash).toEqual(e);
    expect(spy).not.toHaveBeenCalled();
  });

  it("ignores 'run' and 'field' (handled at their source, not the funnel)", () => {
    const spy = vi.spyOn(errorStore, 'toast').mockImplementation(() => {});
    reportError({ severity: 'run', title: 'r', message: 'm' });
    reportError({ severity: 'field', title: 'f', message: 'm' });
    expect(spy).not.toHaveBeenCalled();
    expect(errorStore.crash).toBeNull();
  });
});

describe('toRunError', () => {
  it('maps a cluster_setup RunError payload to the dedicated card', () => {
    const e = toRunError({
      kind: 'cluster_setup',
      message: 'FixedSize cluster regime needs cluster_size >= 2; got 1',
    });
    expect(e.severity).toBe('run');
    expect(e.kind).toBe('cluster_setup');
    expect(e.title).toMatch(/cluster/i);
    // the engine's own message (with the fix hint) is surfaced, not buried
    expect(e.message).toContain('cluster_size');
    expect(e.detail).toContain('cluster_size');
  });

  it('maps a generic RunError payload to the failure card', () => {
    const e = toRunError({ kind: 'generic', message: 'engine exploded' });
    expect(e.kind).toBe('generic');
    expect(e.title).toBe('Run failed');
    expect(e.message).toBe('engine exploded');
  });

  it('maps a thrown Error (mock / pre-invoke path) to a generic run error', () => {
    const e = toRunError(new Error("factor 'group' has only one observed level"));
    expect(e.kind).toBe('generic');
    expect(e.message).toContain('only one observed level');
    // full text preserved for "Copy details" / "Show details"
    expect(e.detail).toBeTruthy();
  });

  it('falls back for a non-Error, non-payload throw', () => {
    const e = toRunError('weird string');
    expect(e.kind).toBe('generic');
    expect(e.message).toBe('The run failed unexpectedly.');
    expect(e.detail).toBe('weird string');
  });
});
