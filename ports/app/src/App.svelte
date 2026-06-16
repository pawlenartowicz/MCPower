<script lang="ts">
// Root application shell: mounts the full two-pane layout (config left, results right) with responsive single-pane fallback below 900 px.
import Header from '$lib/components/shell/Header.svelte';
import ResponsivePaneToggle from '$lib/components/shell/ResponsivePaneToggle.svelte';
import Splitter from '$lib/components/shell/Splitter.svelte';
import StatusBar from '$lib/components/shell/StatusBar.svelte';
import TitleBar from '$lib/components/shell/TitleBar.svelte';
import ConfigPanel from '$lib/components/shell/ConfigPanel.svelte';
import ResultsPane from '$lib/components/results/ResultsPane.svelte';
import SettingsSlideOver from '$lib/components/overlays/SettingsSlideOver.svelte';
import HistorySlideOver from '$lib/components/overlays/HistorySlideOver.svelte';
import AcknowledgmentsModal from '$lib/components/overlays/AcknowledgmentsModal.svelte';
import CrashModal from '$lib/components/shell/CrashModal.svelte';
import ThemeProvider from '$lib/theme/ThemeProvider.svelte';
import { Toaster } from '$lib/components/ui/sonner';
import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';
import { runStore } from '$lib/stores/run.svelte';
import { reportError, errorDetail } from '$lib/errors/report';
import { onMount, onDestroy } from 'svelte';
import { attachMenuRouter } from '$lib/api/menu';
import { setNThreads, parseFormula } from '$lib/api/engine';

let unlisten: (() => void) | undefined;
onMount(async () => {
  try {
    unlisten = await attachMenuRouter();
  } catch (e) {
    console.warn('menu router unavailable', e);
  }
});

// Browser shell only: the WASM engine fails to instantiate on WebKit (Safari and
// every iPhone/iPad browser) because faer's compiled SIMD carries relaxed-SIMD
// opcodes WebKit doesn't support yet. Probe the engine once at boot — a trivial
// parse triggers the shared wasm init — so an unsupported browser gets a clear,
// blocking message instead of the raw CompileError leaking out as a bogus
// formula-parse error. Any engine-init failure (not just relaxed-SIMD) lands here.
onMount(async () => {
  if (import.meta.env.VITE_TARGET !== 'wasm') return;
  try {
    await parseFormula('y ~ x');
  } catch (e) {
    reportError({
      severity: 'crash',
      title: "This browser can't run MCPower",
      message:
        "Your browser is missing a WebAssembly feature MCPower needs. It works in Chrome, Firefox, or Edge (desktop or Android), or in the desktop app from mcpower.app — but browsers on iPhone and iPad aren't supported yet.",
      detail: errorDetail(e),
    });
  }
});

// Apply the stored thread count once settings are loaded. Tauri-only: the WASM
// shell uses single-core workers and its setNThreads stub is a no-op. The
// _threadsApplied guard (not a store subscription) avoids a double set on rapid
// `ready` flips — the effect would otherwise rerun whenever sharedPrefs changes.
let _threadsApplied = false;
$effect(() => {
  if (import.meta.env.VITE_TARGET === 'wasm') return;
  if (!sharedPrefs.ready || _threadsApplied) return;
  _threadsApplied = true;
  const n = sharedPrefs.nThreads;
  if (n !== null && n > 0) {
    void setNThreads(n).catch((err: unknown) => {
      console.warn('set_n_threads_cmd failed:', err);
    });
  }
});
onDestroy(() => unlisten?.());

let windowWidth = $state(typeof window !== 'undefined' ? window.innerWidth : 1200);

$effect(() => {
  const onResize = () => (windowWidth = window.innerWidth);
  window.addEventListener('resize', onResize);
  return () => window.removeEventListener('resize', onResize);
});

$effect(() => {
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && runStore.runState === 'running') {
      void runStore.cancel();
    }
  };
  window.addEventListener('keydown', onKey);
  return () => window.removeEventListener('keydown', onKey);
});

// Last-resort catch for promise rejections that no run/control handler claimed — these
// have no contextual surface, so they route to the blocking crash modal. (Engine/IPC
// failures during a run are caught by runStore and shown as the run card, not here.)
$effect(() => {
  const onRejection = (e: PromiseRejectionEvent) => {
    const reason = e.reason;
    reportError({
      severity: 'crash',
      title: 'Something went wrong',
      message: 'An unexpected error occurred. Your inputs are safe; please retry.',
      detail: errorDetail(reason),
    });
  };
  window.addEventListener('unhandledrejection', onRejection);
  return () => window.removeEventListener('unhandledrejection', onRejection);
});

const isNarrow = $derived(windowWidth < 900);

// Mobile header auto-collapse: hide the family-ribbon row while scrolling down a
// pane, reveal on scroll-up or near the top. Only the narrow header reads it.
// Drives the mobile collapse of the family ribbon (Header) AND the scenarios toggle
// (StatusBar): hidden on scroll-down, shown on scroll-up / at the top.
let headerCollapsed = $state(false);
let lastY = 0;
let scrollLock = false;
function onPaneScroll(e: Event) {
  if (!isNarrow || scrollLock) return;
  const el = e.currentTarget as HTMLElement;
  const y = el.scrollTop;
  let next = headerCollapsed;
  if (y < 8) next = false; // at the top → always expanded
  else if (y > lastY + 6) next = true; // scrolling down → collapse
  else if (y < lastY - 6) next = false; // scrolling up → expand
  lastY = y;
  if (next !== headerCollapsed) {
    headerCollapsed = next;
    // Collapsing/expanding re-clamps scrollTop, which fires a burst of scroll events
    // that the direction test reads as the opposite move and reflips the state. Lock
    // until the transition + reflow settle, then resync the baseline.
    scrollLock = true;
    setTimeout(() => {
      scrollLock = false;
      lastY = el.scrollTop;
    }, 300);
  }
}
// Reset the collapse whenever the visible pane changes — the new pane starts at the top.
$effect(() => {
  sharedPrefs.activePane;
  headerCollapsed = false;
  lastY = 0;
});
</script>

<ThemeProvider>
  <!-- h-dvh (dynamic viewport), not h-screen/100vh: on mobile the URL bar shrinks the
       visible area, so 100vh overflows it and the shell scrolls — pushing the header
       and Config/Results toggle off-screen. dvh tracks the visible viewport and equals
       100vh on desktop, so the shell always fits without page scroll. -->
  <div class="flex h-dvh w-screen flex-col bg-background text-foreground">
    <TitleBar />
    <Header narrow={isNarrow} collapsed={headerCollapsed} />
    {#if isNarrow}
      <ResponsivePaneToggle />
      <main class="flex-1 overflow-hidden">
        {#if sharedPrefs.activePane === 'config'}
          <section class="flex h-full flex-col">
            <StatusBar collapsed={headerCollapsed} />
            <div class="flex-1 overflow-auto" onscroll={onPaneScroll}>
              <ConfigPanel />
            </div>
          </section>
        {:else}
          <section class="flex h-full flex-col">
            <StatusBar collapsed={headerCollapsed} />
            <div class="flex-1 overflow-auto" onscroll={onPaneScroll}>
              <ResultsPane />
            </div>
          </section>
        {/if}
      </main>
    {:else}
      <Splitter>
        {#snippet left()}
          <section class="h-full overflow-auto">
            <ConfigPanel />
          </section>
        {/snippet}
        {#snippet right()}
          <section class="flex h-full flex-col">
            <StatusBar collapsed={headerCollapsed} />
            <div class="flex-1 overflow-auto" onscroll={onPaneScroll}>
              <ResultsPane />
            </div>
          </section>
        {/snippet}
      </Splitter>
    {/if}
    <SettingsSlideOver />
    <HistorySlideOver />
    <AcknowledgmentsModal />
    <CrashModal />
    <Toaster />
  </div>
</ThemeProvider>
