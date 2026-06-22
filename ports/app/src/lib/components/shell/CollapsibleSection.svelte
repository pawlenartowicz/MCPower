<script lang="ts">
  // Shared collapsible section shell for the left config pane. One component renders
  // every top-level section (Upload / Model / Correlations / Run) so they stay
  // visually consistent: a card with a bold header, muted summary, optional info
  // icon, and a collapsible body. Sections used to each roll their own header,
  // which is how they drifted apart.
  import {
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,
  } from '$lib/components/ui/collapsible';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    /** Muted summary shown after the title (e.g. the formula, "all tests"). */
    summary?: string;
    /** Render an italic "· optional" tag instead of a summary. */
    optional?: boolean;
    /** Tip key for the trailing info icon; omit to hide the icon. */
    tipKey?: string;
    open?: boolean;
    children: Snippet;
  }
  let {
    title,
    summary,
    optional = false,
    tipKey,
    open = $bindable(false),
    children,
  }: Props = $props();
</script>

<section
  class="rounded-lg border border-border bg-card shadow-sm transition-colors {open
    ? 'border-primary/25'
    : ''}"
>
  <Collapsible bind:open>
    <div class="flex items-center gap-1.5 px-3.5">
      <CollapsibleTrigger
        class="flex min-w-0 flex-1 items-center gap-2.5 py-3 text-left"
      >
        {#if open}
          <ChevronDown class="h-4 w-4 shrink-0 text-primary" />
        {:else}
          <ChevronRight class="h-4 w-4 shrink-0 text-muted-foreground" />
        {/if}
        <span class="shrink-0 text-[15px] font-bold tracking-tight">{title}</span>
        {#if optional || summary}
          <span class="truncate text-xs font-normal text-muted-foreground">
            ·
            {#if optional}<span class="italic">optional</span>{/if}{#if summary}{summary}{/if}
          </span>
        {/if}
      </CollapsibleTrigger>
      {#if tipKey}
        <InfoIcon {tipKey} />
      {/if}
    </div>
    <CollapsibleContent class="space-y-3 px-3.5 pb-4 pt-1">
      {@render children()}
    </CollapsibleContent>
  </Collapsible>
</section>
