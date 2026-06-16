<script lang="ts">
  import type { Snippet } from 'svelte';
  import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';
  import { getSystemTheme, type ResolvedTheme } from '$lib/theme';

  interface Props {
    children?: Snippet;
  }
  const { children }: Props = $props();

  let systemTheme = $state<'light' | 'dark'>(getSystemTheme());

  $effect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = (e: MediaQueryListEvent) => {
      systemTheme = e.matches ? 'dark' : 'light';
    };
    if (typeof mq.addEventListener === 'function') {
      mq.addEventListener('change', onChange);
      return () => mq.removeEventListener('change', onChange);
    }
    // addListener: deprecated pre-spec fallback for Safari < 14.
    mq.addListener(onChange);
    return () => mq.removeListener(onChange);
  });

  const resolved = $derived<ResolvedTheme>(
    sharedPrefs.theme === 'system' ? systemTheme : sharedPrefs.theme,
  );

  $effect(() => {
    if (typeof document === 'undefined') return;
    document.documentElement.classList.toggle('dark', resolved === 'dark');
  });

  $effect(() => {
    if (typeof document === 'undefined') return;
    document.documentElement.style.fontSize = `${sharedPrefs.fontSize}px`;
  });
</script>

{@render children?.()}
