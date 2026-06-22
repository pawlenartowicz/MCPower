<script lang="ts">
  // Native window title bar with minimize / maximize / close controls; rendered only when the Tauri window API is available.
  import CircleDot from '@lucide/svelte/icons/circle-dot';
  import Minus from '@lucide/svelte/icons/minus';
  import Square from '@lucide/svelte/icons/square';
  import X from '@lucide/svelte/icons/x';
  import { onMount } from 'svelte';
  import Logo from './Logo.svelte';

  type WindowApi = {
    minimize: () => Promise<void> | void;
    maximize: () => Promise<void> | void;
    unmaximize: () => Promise<void> | void;
    isMaximized: () => Promise<boolean>;
    close: () => Promise<void> | void;
  };

  let win = $state<WindowApi | null>(null);

  onMount(async () => {
    try {
      const mod = await import('@tauri-apps/api/window');
      const current = mod.getCurrentWindow();
      win = current as unknown as WindowApi;
    } catch (e) {
      console.warn('titlebar: window API unavailable', e);
    }
  });

  const minimize = () => win?.minimize();
  const toggleMaximize = async () => {
    if (!win) return;
    if (await win.isMaximized()) await win.unmaximize();
    else await win.maximize();
  };
  const close = () => win?.close();
</script>

{#if win}
  <div
    class="flex h-7 shrink-0 items-center border-b border-border bg-background select-none"
  >
    <div class="flex items-center gap-1.5 px-2">
      <CircleDot class="h-3.5 w-3.5 text-muted-foreground" />
      <Logo class="h-3.5 w-auto text-foreground" />
    </div>
    <div data-tauri-drag-region class="flex-1 h-full"></div>
    <div class="flex h-full items-center">
      <button
        type="button"
        aria-label="Minimize"
        class="inline-flex h-7 w-9 items-center justify-center text-muted-foreground hover:bg-muted"
        onclick={minimize}
      >
        <Minus class="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        aria-label="Maximize"
        class="inline-flex h-7 w-9 items-center justify-center text-muted-foreground hover:bg-muted"
        onclick={toggleMaximize}
      >
        <Square class="h-3 w-3" />
      </button>
      <button
        type="button"
        aria-label="Close"
        class="inline-flex h-7 w-9 items-center justify-center text-muted-foreground hover:bg-destructive hover:text-destructive-foreground"
        onclick={close}
      >
        <X class="h-3.5 w-3.5" />
      </button>
    </div>
  </div>
{/if}
