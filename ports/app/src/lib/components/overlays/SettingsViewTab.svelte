<script lang="ts">
  // Settings tab for visual preferences: theme selection and font size.
  import { Label } from '$lib/components/ui/label';
  import { ToggleGroup, ToggleGroupItem } from '$lib/components/ui/toggle-group';
  import { Slider } from '$lib/components/ui/slider';
  import { sharedPrefs, type Theme } from '$lib/stores/shared-prefs.svelte';

  const THEMES: { value: Theme; label: string }[] = [
    { value: 'light', label: 'Light' },
    { value: 'dark', label: 'Dark' },
    { value: 'system', label: 'System' },
  ];
</script>

<div class="space-y-4 px-1 pt-1">
  <div class="space-y-2">
    <Label>Appearance</Label>
    <ToggleGroup
      type="single"
      variant="outline"
      value={sharedPrefs.theme}
      onValueChange={(v: string) => {
        if (v === 'light' || v === 'dark' || v === 'system') sharedPrefs.theme = v;
      }}
      aria-label="Theme"
    >
      {#each THEMES as t (t.value)}
        <ToggleGroupItem value={t.value} aria-label={t.label}>{t.label}</ToggleGroupItem>
      {/each}
    </ToggleGroup>
  </div>
  <div>
    <Label>Font size</Label>
    <Slider
      type="single"
      value={sharedPrefs.fontSize}
      min={11}
      max={20}
      step={1}
      onValueChange={(v: number) => (sharedPrefs.fontSize = v)}
    />
    <p class="mt-1 text-xs text-muted-foreground">{sharedPrefs.fontSize}px</p>
  </div>
</div>
