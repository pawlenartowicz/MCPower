<script lang="ts">
  // Right-side slide-over settings panel with View, Run config, and Scenarios tabs.
  import { Sheet, SheetContent, SheetHeader, SheetTitle } from '$lib/components/ui/sheet';
  import { Tabs, TabsContent, TabsList, TabsTrigger } from '$lib/components/ui/tabs';
  import { uiStore } from '$lib/stores/ui.svelte';
  import SettingsViewTab from './SettingsViewTab.svelte';
  import SettingsRunConfigTab from './SettingsRunConfigTab.svelte';
  import SettingsScenariosTab from './SettingsScenariosTab.svelte';

  let tab = $state<'view' | 'run-config' | 'scenarios'>('view');

  // Floating panel sits 12px below the app header; measured at open time because the
  // sheet is portaled (fixed) and the header bottom shifts when the Tauri TitleBar
  // renders. Mirrors HistorySlideOver — change together.
  let panelTop = $state(116);
  // Very narrow screens (phones): the panel takes the whole viewport edge-to-edge
  // instead of the floating right-hand card — measured at open time alongside
  // panelTop. Mirrors HistorySlideOver — change together.
  let fullScreen = $state(false);
  $effect(() => {
    if (uiStore.settingsOpen) {
      panelTop = (document.querySelector('header')?.getBoundingClientRect().bottom ?? 104) + 12;
      fullScreen = window.innerWidth < 640;
    }
  });
</script>

<Sheet open={uiStore.settingsOpen} onOpenChange={(v: boolean) => (uiStore.settingsOpen = v)}>
  <SheetContent
    side="right"
    class={fullScreen
      ? '!inset-0 !w-full !max-w-full !rounded-none border-0'
      : 'w-full !max-w-[calc(100vw-1.5rem)] sm:!max-w-xl !right-3 !bottom-3 rounded-xl border shadow-xl'}
    style={fullScreen ? undefined : `top: ${panelTop}px`}
  >
    <SheetHeader>
      <SheetTitle>Settings</SheetTitle>
    </SheetHeader>
    <div class="min-h-0 flex-1 overflow-y-auto px-4 pt-4 pb-4">
      <Tabs bind:value={tab}>
        <TabsList>
          <TabsTrigger value="view">View</TabsTrigger>
          <TabsTrigger value="run-config">Run config</TabsTrigger>
          <TabsTrigger value="scenarios">Scenarios</TabsTrigger>
        </TabsList>
        <TabsContent value="view">
          <SettingsViewTab />
        </TabsContent>
        <TabsContent value="run-config">
          <SettingsRunConfigTab />
        </TabsContent>
        <TabsContent value="scenarios">
          <SettingsScenariosTab />
        </TabsContent>
      </Tabs>
    </div>
  </SheetContent>
</Sheet>
