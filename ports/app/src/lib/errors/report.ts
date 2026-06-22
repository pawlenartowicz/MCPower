// Shared error-funnel types for the app. `severity` decides which surface an error reaches.
import { errorStore } from '$lib/stores/error.svelte';

export type Severity = 'field' | 'run' | 'background' | 'crash';

export interface AppError {
  severity: Severity;
  // Sub-category within a severity. 'run' uses it to pick the surface:
  // 'cluster_setup' → the dedicated cluster/sample-size card, else 'generic'.
  kind?: 'cluster_setup' | 'generic';
  title: string; // short, human-readable headline
  message: string; // one or two sentences for the user
  detail?: string; // full engine/stack text, copyable
  context?: string; // for 'field': which control the error belongs to
}

// The copyable full text for an error of unknown type — the stack when available,
// otherwise the message or a string coercion. This is what "Copy details" puts on the
// clipboard, and what was previously lost to console.error.
export function errorDetail(err: unknown): string {
  return err instanceof Error ? (err.stack ?? err.message) : String(err);
}

// The structured payload the two Tauri run commands reject with (see
// commands.rs `RunError`). The E2E mock and any pre-invoke failure throw a
// plain Error instead — toRunError handles both shapes.
interface RunErrorPayload {
  kind: string;
  message: string;
}

function isRunErrorPayload(err: unknown): err is RunErrorPayload {
  return (
    typeof err === 'object' &&
    err !== null &&
    'kind' in err &&
    'message' in err &&
    typeof (err as { message: unknown }).message === 'string'
  );
}

// Map a thrown/rejected run failure to the AppError the run card reads. Cluster-
// vs-sample-size-grid misconfigurations get a dedicated title + kind so
// RunErrorCard frames them as fixable settings, not a crash; the engine's own
// message (which carries the fix hint) is surfaced directly rather than buried.
export function toRunError(err: unknown): AppError {
  if (isRunErrorPayload(err)) {
    const cluster = err.kind === 'cluster_setup';
    return {
      severity: 'run',
      kind: cluster ? 'cluster_setup' : 'generic',
      title: cluster ? "Cluster and sample-size settings don't fit together" : 'Run failed',
      message: err.message,
      detail: err.message,
    };
  }
  return {
    severity: 'run',
    kind: 'generic',
    title: 'Run failed',
    message: err instanceof Error ? err.message : 'The run failed unexpectedly.',
    detail: errorDetail(err),
  };
}

// Routes background → non-blocking toast, crash → blocking modal.
// 'run' and 'field' are not routed here: 'run' would create an import cycle with runStore,
// and 'field' is bound to local control state. Calling reportError with either is a no-op.
export function reportError(e: AppError): void {
  switch (e.severity) {
    case 'background':
      errorStore.toast(e);
      break;
    case 'crash':
      errorStore.crash = e;
      break;
  }
}
