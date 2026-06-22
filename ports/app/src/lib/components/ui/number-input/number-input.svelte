<script lang="ts">
  import Minus from '@lucide/svelte/icons/minus';
  import Plus from '@lucide/svelte/icons/plus';
  import { cn } from '$lib/utils';

  type Props = {
    value: number;
    oninput?: (v: number) => void;
    step?: number;
    min?: number;
    max?: number;
    class?: string;
    id?: string;
    'aria-label'?: string;
    disabled?: boolean;
    disableDecrement?: boolean;
    decrementTitle?: string;
    suffix?: string;
    dense?: boolean;
    'data-testid'?: string;
  };

  let {
    value = $bindable(0),
    oninput,
    step = 1,
    min,
    max,
    class: className,
    id,
    'aria-label': ariaLabel,
    disabled = false,
    disableDecrement = false,
    decrementTitle,
    suffix,
    dense = false,
    'data-testid': dataTestid,
  }: Props = $props();

  function clamp(n: number): number {
    if (min !== undefined && n < min) return min;
    if (max !== undefined && n > max) return max;
    return n;
  }

  function decimalsOf(s: number): number {
    if (!Number.isFinite(s)) return 0;
    const str = String(s);
    const dot = str.indexOf('.');
    return dot >= 0 ? str.length - dot - 1 : 0;
  }

  function commit(next: number) {
    const v = clamp(next);
    value = v;
    oninput?.(v);
  }

  function bump(dir: 1 | -1) {
    const d = decimalsOf(step);
    const next = value + dir * step;
    commit(d > 0 ? Number(next.toFixed(d)) : Math.round(next));
  }
</script>

<div
  class={cn(
    'inline-flex h-9 items-stretch rounded-md border border-input bg-transparent shadow-xs transition-[color,box-shadow] focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50 dark:bg-input/30',
    disabled && 'pointer-events-none opacity-50',
    className,
  )}
  data-testid={dataTestid}
>
  <button
    type="button"
    aria-label="Decrement"
    title={decrementTitle}
    disabled={disabled || disableDecrement}
    class="inline-flex {dense ? 'w-6' : 'w-7'} shrink-0 items-center justify-center rounded-l-md border-r border-input text-foreground hover:bg-accent hover:text-accent-foreground focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50"
    onclick={() => bump(-1)}
  >
    <Minus class="h-3 w-3" />
  </button>
  <div class="relative flex w-full min-w-0 items-center">
    <input
      {id}
      type="number"
      aria-label={ariaLabel}
      {step}
      {min}
      {max}
      {disabled}
      {value}
      onchange={(e) => commit(Number((e.currentTarget as HTMLInputElement).value))}
      class={cn(
        'w-full min-w-0 border-0 bg-transparent py-1 text-center text-sm text-foreground outline-none placeholder:text-muted-foreground disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 [appearance:textfield] [&::-webkit-inner-spin-button]:m-0 [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none',
        suffix ? (dense ? 'pl-1 pr-4' : 'pl-2 pr-5') : dense ? 'px-1' : 'px-2',
      )}
    />
    {#if suffix}
      <span
        aria-hidden="true"
        class="pointer-events-none absolute right-1.5 text-xs text-muted-foreground"
      >
        {suffix}
      </span>
    {/if}
  </div>
  <button
    type="button"
    aria-label="Increment"
    {disabled}
    class="inline-flex {dense ? 'w-6' : 'w-7'} shrink-0 items-center justify-center rounded-r-md border-l border-input text-foreground hover:bg-accent hover:text-accent-foreground focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50"
    onclick={() => bump(1)}
  >
    <Plus class="h-3 w-3" />
  </button>
</div>
