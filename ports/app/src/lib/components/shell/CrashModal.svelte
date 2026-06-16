<script lang="ts">
  // The 'crash' error surface: a blocking modal for genuinely unhandled rejections that
  // have no run/control context. Reads errorStore.crash; the technical details are
  // collapsed by default and copyable, and Close clears the payload.
  import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
  } from '$lib/components/ui/dialog';
  import { Button } from '$lib/components/ui/button';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import { errorStore } from '$lib/stores/error.svelte';

  const crash = $derived(errorStore.crash);
  let showDetails = $state(false);

  function close() {
    errorStore.crash = null;
    showDetails = false;
  }

  async function copy() {
    if (crash?.detail) await navigator.clipboard.writeText(crash.detail);
  }
</script>

<Dialog open={crash != null} onOpenChange={(v) => { if (!v) close(); }}>
  <DialogContent showCloseButton={false}>
    <DialogHeader>
      <DialogTitle>{crash?.title ?? 'Something went wrong'}</DialogTitle>
      <DialogDescription>{crash?.message ?? ''}</DialogDescription>
    </DialogHeader>

    {#if crash?.detail}
      <div class="space-y-2">
        <Button
          variant="ghost"
          size="sm"
          class="px-1 text-muted-foreground"
          aria-expanded={showDetails}
          onclick={() => (showDetails = !showDetails)}
        >
          <ChevronRight class="mr-1 h-3 w-3 transition-transform {showDetails ? 'rotate-90' : ''}" />
          Technical details
        </Button>
        {#if showDetails}
          <pre class="max-h-48 overflow-auto rounded-md border border-border bg-muted/30 p-3 font-mono text-xs break-words whitespace-pre-wrap">{crash.detail}</pre>
        {/if}
      </div>
    {/if}

    <DialogFooter>
      {#if crash?.detail}
        <Button variant="outline" onclick={copy}>Copy</Button>
      {/if}
      <Button onclick={close}>Close</Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
