<script lang="ts">
  // Help-circle icon → popover backed by a documentation leaf: the first paragraph
  // of a vault page/section (single-sourced via TIP_SOURCES) plus a "see more →"
  // link to the published page. A delegated click handler routes the paragraph's
  // wiki-links and the see-more link through the system browser, so the app's own
  // webview never navigates.
  import HelpCircle from '@lucide/svelte/icons/help-circle';
  import { Popover, PopoverContent, PopoverTrigger } from '$lib/components/ui/popover';
  import { TIP_SOURCES } from '$lib/content/tips';
  import { renderDoc, firstParagraph, openExternal, DOCS_BASE_URL } from '$lib/content/render-doc';

  interface Props {
    tipKey: string;
  }
  const { tipKey }: Props = $props();

  const src = $derived(TIP_SOURCES[tipKey]);
  const body = $derived(src ? renderDoc(firstParagraph(src.md, src.anchor)) : '');
  const seeMoreHref = $derived(
    src ? `${DOCS_BASE_URL}/${src.doc}${src.anchor ? `#${src.anchor}` : ''}` : '',
  );

  function handleLinkClick(e: MouseEvent) {
    const anchor = (e.target as HTMLElement).closest('a');
    if (anchor?.href) {
      e.preventDefault();
      openExternal(anchor.href);
    }
  }
</script>

<Popover>
  <PopoverTrigger
    class="inline-flex h-4 w-4 items-center justify-center rounded text-muted-foreground hover:text-foreground"
    aria-label={`Help for ${tipKey}`}
  >
    <HelpCircle class="h-3.5 w-3.5" />
  </PopoverTrigger>
  <PopoverContent class="w-80 text-sm">
    <!-- svelte-ignore a11y_no_static_element_interactions, a11y_click_events_have_key_events -->
    <div class="prose prose-sm max-w-none" onclick={handleLinkClick}>
      {@html body}
      {#if seeMoreHref}
        <p class="mt-2">
          <a href={seeMoreHref} class="font-medium text-primary hover:underline">see more →</a>
        </p>
      {/if}
    </div>
  </PopoverContent>
</Popover>
