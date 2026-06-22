<script lang="ts">
  // Empty-state tutorial column: renders the active family's panel guide and opens its
  // documentation links in the system browser instead of navigating the app's webview.
  import { familyStore } from '$lib/stores/family.svelte';
  import { TUTORIAL } from '$lib/content/tutorials';
  import { renderDoc, openExternal } from '$lib/content/render-doc';

  const html = $derived(renderDoc(TUTORIAL[familyStore.active]));

  function handleLinkClick(e: MouseEvent) {
    const anchor = (e.target as HTMLElement).closest('a');
    if (anchor?.href) {
      e.preventDefault();
      openExternal(anchor.href);
    }
  }
</script>

<!-- svelte-ignore a11y_no_noninteractive_element_interactions, a11y_click_events_have_key_events -->
<article
  data-testid="family-tutorial"
  class="prose prose-sm mx-auto max-w-[78ch] px-4 text-justify"
  onclick={handleLinkClick}
>{@html html}</article>
