import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import CrashModal from './CrashModal.svelte';
import { errorStore } from '$lib/stores/error.svelte';

describe('CrashModal', () => {
  beforeEach(() => {
    errorStore.crash = null;
  });
  afterEach(() => {
    errorStore.crash = null;
  });

  it('renders the crash title and message when a crash payload is set', async () => {
    errorStore.crash = {
      severity: 'crash',
      title: 'Something went wrong',
      message: 'An unexpected error occurred.',
    };
    render(CrashModal);
    expect(await screen.findByText('Something went wrong')).toBeTruthy();
    expect(screen.getByText('An unexpected error occurred.')).toBeTruthy();
  });

  it('reveals the technical details only after expanding', async () => {
    errorStore.crash = { severity: 'crash', title: 'Boom', message: 'msg', detail: 'STACK TRACE LINE' };
    render(CrashModal);
    const trigger = await screen.findByText('Technical details');
    expect(screen.queryByText('STACK TRACE LINE')).toBeNull();
    await fireEvent.click(trigger);
    expect(await screen.findByText('STACK TRACE LINE')).toBeTruthy();
  });

  it('Copy writes the detail to the clipboard', async () => {
    const writeText = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    errorStore.crash = { severity: 'crash', title: 'Boom', message: 'msg', detail: 'STACK' };
    render(CrashModal);
    await fireEvent.click(await screen.findByText('Copy'));
    expect(writeText).toHaveBeenCalledWith('STACK');
  });

  it('Close clears the crash payload', async () => {
    errorStore.crash = { severity: 'crash', title: 'Boom', message: 'msg' };
    render(CrashModal);
    await fireEvent.click(await screen.findByText('Close'));
    expect(errorStore.crash).toBeNull();
  });
});
