<script lang="ts">
  // Icon representing linear regression: scatter dots along a diagonal trend line.
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
  <line x1="3" y1="21" x2="21" y2="3" />
  <circle cx="4" cy="16" r="1.3" fill="currentColor" stroke="none" />
  <circle cx="8" cy="11" r="1.3" fill="currentColor" stroke="none" />
  <circle cx="13" cy="16" r="1.3" fill="currentColor" stroke="none" />
  <circle cx="17" cy="3" r="1.3" fill="currentColor" stroke="none" />
  <circle cx="20" cy="9" r="1.3" fill="currentColor" stroke="none" />
</svg>
