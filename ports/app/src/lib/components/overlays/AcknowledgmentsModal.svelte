<script lang="ts">
  // Modal dialog listing third-party libraries and design-vocabulary credits for MCPower.
  import { Dialog, DialogContent, DialogTitle } from '$lib/components/ui/dialog';
  import { uiStore } from '$lib/stores/ui.svelte';
</script>

<Dialog
  open={uiStore.acknowledgmentsOpen}
  onOpenChange={(v: boolean) => (uiStore.acknowledgmentsOpen = v)}
>
  <DialogContent>
    <DialogTitle>Acknowledgments</DialogTitle>
    <p class="pt-2 text-sm text-muted-foreground">
      MCPower is built on Tauri, Svelte, shadcn-svelte, Vega-Lite, and the Rust ecosystem.
      Tooltip copy and design vocabulary carried over from the v1 PySide6 application.
    </p>
  </DialogContent>
</Dialog>
