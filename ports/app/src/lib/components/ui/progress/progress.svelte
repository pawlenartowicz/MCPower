<script lang="ts">
	import { Progress as ProgressPrimitive } from "bits-ui";
	import { cn, type WithoutChildrenOrChild } from "$lib/utils.js";

	let {
		ref = $bindable(null),
		class: className,
		max = 100,
		value,
		...restProps
	}: WithoutChildrenOrChild<ProgressPrimitive.RootProps> = $props();
</script>

<ProgressPrimitive.Root
	bind:ref
	data-slot="progress"
	class={cn("bg-muted h-1.5 rounded-full relative flex w-full items-center overflow-hidden", className)}
	{value}
	{max}
	{...restProps}
>
	<!--
		translateZ(0) + backface-hidden + will-change pin the animated fill to a
		stable compositor layer. Without it, animating translateX leaves the
		layer edge on fractional device pixels each frame, and a 1px sliver of the
		(near-white) track / amber strip behind it flickers through while moving.
		overflow-hidden (not -x) so the rounded-full corner clip is well-defined.
	-->
	<div
		data-slot="progress-indicator"
		class="bg-primary size-full flex-1 transition-transform duration-150 ease-linear backface-hidden will-change-transform"
		style="transform: translateX(-{100 - (100 * (value ?? 0)) / (max ?? 1)}%) translateZ(0)"
	></div>
</ProgressPrimitive.Root>
