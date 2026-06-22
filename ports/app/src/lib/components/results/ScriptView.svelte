<script lang="ts">
  // Renders the reproducible script (Python or R) for a run tab; the language
  // choice persists globally via sharedPrefs.
  import { Button } from '$lib/components/ui/button';
  import Copy from '@lucide/svelte/icons/copy';
  import type { RunTab } from '$lib/stores/run.svelte';
  import { generateScript } from '$lib/domain/script-generator';
  import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';

  interface Props { tab: RunTab; }
  const { tab }: Props = $props();

  const LANGS: Array<'python' | 'r'> = ['python', 'r'];

  const script = $derived(
    generateScript(tab.spec, tab.kind, {
      sample_size: tab.sample_size,
      bounds: tab.bounds,
      method: tab.method,
    }, sharedPrefs.scriptLanguage),
  );

  async function copy() {
    await navigator.clipboard.writeText(script);
  }
</script>

<div class="space-y-2">
  <div class="flex items-center gap-2">
    <div role="group" aria-label="Script language" class="flex gap-1 text-sm">
      {#each LANGS as l (l)}
        <button
          type="button"
          aria-pressed={sharedPrefs.scriptLanguage === l}
          class="rounded px-3 py-1 {sharedPrefs.scriptLanguage === l
            ? 'bg-primary text-primary-foreground'
            : 'bg-muted text-muted-foreground hover:bg-muted/70'}"
          onclick={() => (sharedPrefs.scriptLanguage = l)}
        >{l === 'python' ? 'Python' : 'R'}</button>
      {/each}
    </div>
    <Button variant="outline" size="sm" onclick={copy}>
      <Copy class="mr-1 h-3 w-3" /> Copy
    </Button>
  </div>
  <pre class="overflow-auto rounded-md border border-border bg-muted/30 p-3 font-mono text-xs">{script}</pre>
</div>
