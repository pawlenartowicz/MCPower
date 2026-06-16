<script lang="ts">
// Narrow-viewport toggle that switches the visible pane between Config and Results.
import { ToggleGroup, ToggleGroupItem } from '$lib/components/ui/toggle-group';
import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';
</script>

<div class="flex justify-center border-b border-border px-4 py-1.5">
  <ToggleGroup
    type="single"
    value={sharedPrefs.activePane}
    onValueChange={(v: string) => {
      if (v === 'config' || v === 'results') sharedPrefs.activePane = v;
    }}
  >
    <ToggleGroupItem value="config">Config</ToggleGroupItem>
    <ToggleGroupItem value="results">Results</ToggleGroupItem>
  </ToggleGroup>
</div>
