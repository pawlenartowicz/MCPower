// errorStore — home for the two error severities with no natural surface of their own:
// `background` (a side operation failed) becomes a sonner toast, and `crash` (a truly
// unhandled rejection) is held as the modal payload. `run` lives on runStore and `field`
// stays component-local, so neither passes through here.
import { toast as sonnerToast } from 'svelte-sonner';
import type { AppError } from '$lib/errors/report';

const TOAST_DURATION_MS = 6000;

let crash = $state<AppError | null>(null);

export const errorStore = {
  toast(e: AppError): void {
    sonnerToast.error(e.title, {
      description: e.message,
      duration: TOAST_DURATION_MS,
      action: e.detail
        ? { label: 'Details', onClick: () => void navigator.clipboard.writeText(e.detail ?? '') }
        : undefined,
    });
  },
  get crash(): AppError | null {
    return crash;
  },
  set crash(value: AppError | null) {
    crash = value;
  },
};
