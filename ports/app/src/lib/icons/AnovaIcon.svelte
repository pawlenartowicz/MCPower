<script lang="ts">
  // Icon representing ANOVA designs: three vertical bracket groups with mean dots.
  type Props = { class?: string };
  let { class: cls = '' }: Props = $props();
</script>

<svg
  xmlns="http://www.w3.org/2000/svg"
  width="24"
  height="24"
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width="2"
  stroke-linecap="round"
  stroke-linejoin="round"
  class={cls}
  aria-hidden="true"
>
  <line x1="4" y1="11" x2="4" y2="21" />
  <line x1="2" y1="11" x2="6" y2="11" />
  <line x1="2" y1="21" x2="6" y2="21" />
  <circle cx="4" cy="16" r="1.6" fill="currentColor" stroke="none" />

  <line x1="12" y1="3" x2="12" y2="13" />
  <line x1="10" y1="3" x2="14" y2="3" />
  <line x1="10" y1="13" x2="14" y2="13" />
  <circle cx="12" cy="8" r="1.6" fill="currentColor" stroke="none" />

  <line x1="20" y1="8" x2="20" y2="18" />
  <line x1="18" y1="8" x2="22" y2="8" />
  <line x1="18" y1="18" x2="22" y2="18" />
  <circle cx="20" cy="13" r="1.6" fill="currentColor" stroke="none" />
</svg>
