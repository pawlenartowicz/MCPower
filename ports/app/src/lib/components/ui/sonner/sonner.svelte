<script lang="ts">
  // shadcn-svelte sonner Toaster wrapper. Colours track the app theme via CSS variables
  // (--popover/--border flip with the .dark class on <html>); the `theme` prop only
  // drives sonner's own default styling, resolved from the user's theme preference.
  import { Toaster as Sonner, type ToasterProps } from 'svelte-sonner';
  import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';
  import { getSystemTheme } from '$lib/theme';

  let restProps: ToasterProps = $props();

  const theme = $derived<'light' | 'dark'>(
    sharedPrefs.theme === 'system' ? getSystemTheme() : sharedPrefs.theme,
  );
</script>

<Sonner
  {theme}
  class="toaster group"
  style="--normal-bg: var(--popover); --normal-text: var(--popover-foreground); --normal-border: var(--border);"
  {...restProps}
/>
