import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('svelte-sonner', () => ({
  toast: { error: vi.fn() },
}));

import { toast as sonnerToast } from 'svelte-sonner';
import { errorStore } from './error.svelte';
import type { AppError } from '$lib/errors/report';

describe('errorStore', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    errorStore.crash = null;
  });

  it('toast() forwards title as the toast and message as the description', () => {
    const e: AppError = {
      severity: 'background',
      title: "Couldn't save settings",
      message: 'Disk may be read-only.',
    };
    errorStore.toast(e);
    expect(sonnerToast.error).toHaveBeenCalledWith(
      "Couldn't save settings",
      expect.objectContaining({ description: 'Disk may be read-only.' }),
    );
  });

  it('toast() attaches a Details action only when detail is present', () => {
    errorStore.toast({ severity: 'background', title: 'A', message: 'b', detail: 'stack trace' });
    const withDetail = vi.mocked(sonnerToast.error).mock.calls[0]?.[1];
    expect(withDetail?.action).toMatchObject({ label: 'Details' });

    vi.mocked(sonnerToast.error).mockClear();
    errorStore.toast({ severity: 'background', title: 'A', message: 'b' });
    const withoutDetail = vi.mocked(sonnerToast.error).mock.calls[0]?.[1];
    expect(withoutDetail?.action).toBeUndefined();
  });

  it('crash holds and clears the modal payload', () => {
    const e: AppError = { severity: 'crash', title: 'Boom', message: 'unexpected' };
    expect(errorStore.crash).toBeNull();
    errorStore.crash = e;
    expect(errorStore.crash).toEqual(e);
    errorStore.crash = null;
    expect(errorStore.crash).toBeNull();
  });
});
