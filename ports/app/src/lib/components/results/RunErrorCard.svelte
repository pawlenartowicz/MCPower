<script lang="ts">
  // The 'run' error surface: shown in the results area when a launched run fails.
  //
  // Cluster-vs-sample-size-grid misconfigurations (kind === 'cluster_setup') get
  // an amber "fixable settings" frame with a lead-in; everything else keeps the
  // red failure frame. "Show details" expands any extra technical text inline
  // (mirrors CrashModal) and "Copy details" copies it.
  import { Button } from '$lib/components/ui/button';
  import AlertTriangle from '@lucide/svelte/icons/alert-triangle';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import { runStore } from '$lib/stores/run.svelte';

  const error = $derived(runStore.lastError);
  const isClusterSetup = $derived(error?.kind === 'cluster_setup');
  // Only offer the inline preview when `detail` holds more than the visible
  // message. Cluster/engine errors already show their full text as the message,
  // so re-revealing it would be redundant.
  const hasMore = $derived(!!error?.detail && error.detail !== error.message);

  let showDetails = $state(false);

  async function copyDetails() {
    if (error?.detail) await navigator.clipboard.writeText(error.detail);
  }

  function dismiss() {
    showDetails = false;
    runStore.lastError = null;
  }
</script>

{#if error}
  <div
    class="m-4 rounded-md border p-4 {isClusterSetup
      ? 'border-amber-400/60 bg-amber-50'
      : 'border-destructive/50 bg-destructive/10'}"
    role="alert"
  >
    <div class="flex items-start gap-3">
      <AlertTriangle
        class="mt-0.5 h-5 w-5 shrink-0 {isClusterSetup ? 'text-amber-600' : 'text-destructive'}"
      />
      <div class="min-w-0 flex-1 space-y-2">
        <p class="font-medium {isClusterSetup ? 'text-amber-900' : 'text-destructive'}">
          {error.title}
        </p>
        {#if isClusterSetup}
          <p class="text-sm text-amber-900/80">
            These settings can't produce a valid run — adjust the cluster size or the sample-size
            range:
          </p>
        {/if}
        <p
          class="text-sm whitespace-pre-wrap break-words {isClusterSetup
            ? 'text-amber-900'
            : 'text-foreground/90'}"
        >
          {error.message}
        </p>
        {#if hasMore && showDetails}
          <pre
            class="max-h-48 overflow-auto rounded-md border border-border bg-muted/30 p-3 font-mono text-xs break-words whitespace-pre-wrap">{error.detail}</pre>
        {/if}
        <div class="flex gap-2 pt-1">
          {#if hasMore}
            <Button
              variant="ghost"
              size="sm"
              class="px-1 text-muted-foreground"
              aria-expanded={showDetails}
              onclick={() => (showDetails = !showDetails)}
            >
              <ChevronRight
                class="mr-1 h-3 w-3 transition-transform {showDetails ? 'rotate-90' : ''}"
              />
              {showDetails ? 'Hide details' : 'Show details'}
            </Button>
          {/if}
          {#if error.detail}
            <Button variant="outline" size="sm" onclick={copyDetails}>Copy details</Button>
          {/if}
          <Button variant="ghost" size="sm" onclick={dismiss}>Dismiss</Button>
        </div>
      </div>
    </div>
  </div>
{/if}
