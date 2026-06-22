<script lang="ts">
  // Icon representing mixed-effects models: three clustered dot groups showing within-group variance.
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
  <circle cx="3" cy="18" r="1.4" fill="currentColor" stroke="none" />
  <circle cx="6" cy="21" r="1.4" fill="currentColor" stroke="none" />
  <circle cx="8" cy="17" r="1.4" fill="currentColor" stroke="none" />

  <circle cx="10" cy="4" r="1.4" fill="currentColor" stroke="none" />
  <circle cx="13" cy="7" r="1.4" fill="currentColor" stroke="none" />
  <circle cx="14" cy="3" r="1.4" fill="currentColor" stroke="none" />

  <circle cx="17" cy="14" r="1.4" fill="currentColor" stroke="none" />
  <circle cx="20" cy="11" r="1.4" fill="currentColor" stroke="none" />
  <circle cx="21" cy="15" r="1.4" fill="currentColor" stroke="none" />
</svg>
