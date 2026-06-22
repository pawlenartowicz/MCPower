<script lang="ts">
// Top app bar: MCPower wordmark, FamilyRibbon selector, and chrome action buttons
// (Settings, History, Tutorial). On narrow viewports it splits into two rows —
// brand + compact icon chrome, then a full-width family ribbon — so the header
// fits a phone instead of overflowing the viewport and shrinking the whole page.
import Bug from '@lucide/svelte/icons/bug';
import GraduationCap from '@lucide/svelte/icons/graduation-cap';
import History from '@lucide/svelte/icons/history';
import Settings from '@lucide/svelte/icons/settings';
import { uiStore } from '$lib/stores/ui.svelte';
import { openExternal, reportUrl, DOCS_BASE_URL } from '$lib/content/render-doc';
import FamilyRibbon from './FamilyRibbon.svelte';
import Logo from './Logo.svelte';

// collapsed (narrow only): hide the family-ribbon row on scroll-down to reclaim space.
let { narrow = false, collapsed = false }: { narrow?: boolean; collapsed?: boolean } = $props();

const chromeActions = [
  { label: 'Settings', Icon: Settings, onClick: () => (uiStore.settingsOpen = true) },
  { label: 'History', Icon: History, onClick: () => (uiStore.historyOpen = true) },
  { label: 'Tutorial', Icon: GraduationCap, onClick: () => openExternal(`${DOCS_BASE_URL}/tutorial-app/index`) },
  { label: 'Report a bug', Icon: Bug, onClick: () => openExternal(reportUrl()) },
];
</script>

{#snippet chrome(compact: boolean)}
  <div class="flex items-center gap-1">
    {#each chromeActions as { label, Icon, onClick } (label)}
      <button
        type="button"
        aria-label={label}
        title={compact ? label : undefined}
        class={compact
          ? 'inline-flex h-11 w-11 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground'
          : 'inline-flex h-14 w-20 flex-col items-center justify-center gap-1 rounded-md border-b-2 border-transparent px-2 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground'}
        onclick={onClick}
      >
        <Icon class={compact ? 'h-5 w-5' : 'h-6 w-6'} />
        {#if !compact}<span class="font-medium">{label}</span>{/if}
      </button>
    {/each}
  </div>
{/snippet}

{#if narrow}
  <header class="flex flex-col gap-1.5 border-b border-border px-3 py-2">
    <div class="flex items-center justify-between">
      <Logo class="h-5 w-auto text-foreground" />
      {@render chrome(true)}
    </div>
    <div
      class="overflow-hidden transition-[max-height,opacity] duration-200 {collapsed
        ? 'max-h-0 opacity-0'
        : 'max-h-16 opacity-100'}"
    >
      <FamilyRibbon narrow />
    </div>
  </header>
{:else}
  <header class="flex items-center justify-between border-b border-border px-4 py-2">
    <div class="flex items-center gap-4">
      <Logo class="h-5 w-auto text-foreground" />
      <FamilyRibbon />
    </div>
    {@render chrome(false)}
  </header>
{/if}
