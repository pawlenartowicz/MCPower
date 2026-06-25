<script lang="ts">
  // Pure leaf: inputs -> SVG illustration of ONE standardized effect (or one
  // interaction). No engine, no state writes. Cloud noise RESAMPLES from the
  // vetted survivor pool: a per-change counter (sampleIndex) advances one
  // survivor each time a cloud-driving value changes, so the cloud reshuffles on
  // every setter nudge; y is auto-fit per chart (fitRange) so it also MOVES as the
  // effect changes (#10). Continuous-outcome charts: scatter / grouped columns /
  // 2×2 grid / two slopes. Binary outcome keeps the logistic curve + proportion
  // bars (no cloud — nothing to resample there).
  import { untrack } from 'svelte';
  import { readChartColors } from '$lib/charts/theme';
  import {
    logisticCurve,
    bernoulliScatter,
    groupOutcomeBars,
    continuousScatter,
    groupedColumns,
    gridColumns,
    slopeScatter,
    fitRange,
    X_CLAMP,
  } from '$lib/domain/effect-cartoon';

  let {
    outcomeKind,
    predictorKind,
    role,
    beta,
    intercept = 0,
    // grouped factor: the per-level betas (reference excluded; means = [0, ...levelBetas]).
    levelBetas = [] as number[],
    // interaction partners' kinds + main effects (continuous main feeds the slope /
    // grid columns; binShift is the binary partner's main = the group-1 intercept).
    partnerKind = 'continuous' as 'continuous' | 'binary' | 'factor',
    partnerMainA = 0,
    partnerMainB = 0,
    binShift = 0,
  }: {
    outcomeKind: 'continuous' | 'binary';
    predictorKind: 'continuous' | 'binary' | 'factor';
    role: 'main' | 'interaction';
    beta: number;
    intercept?: number;
    levelBetas?: number[];
    partnerKind?: 'continuous' | 'binary' | 'factor';
    partnerMainA?: number;
    partnerMainB?: number;
    binShift?: number;
  } = $props();

  const colors = readChartColors();
  const groupColors = $derived([colors.chart1, colors.chart2, colors.chart3, colors.chart4, colors.chart5]);

  // Resample counter: advance one survivor each time any cloud-driving value
  // changes, so the cloud reshuffles per setter nudge (rotation, not a fixed
  // base). Indexes the pool mod POOL_SIZE inside the helpers. No advance on mount.
  let sampleIndex = $state(0);
  let prevSig: string | undefined;
  const valueSig = $derived(JSON.stringify([beta, levelBetas, partnerMainA, partnerMainB, binShift]));
  $effect(() => {
    const sig = valueSig;
    untrack(() => {
      if (prevSig !== undefined && sig !== prevSig) sampleIndex += 1;
      prevSig = sig;
    });
  });

  const W = 320;
  const H = 200;
  const PAD = 28;

  // Which illustration to draw. Continuous-outcome interactions get grid (bin×bin)
  // or slopes (bin×cont); everything else with no 2-D picture falls back to text.
  const mode = $derived.by(() => {
    if (role === 'main') {
      if (outcomeKind === 'continuous') return predictorKind === 'continuous' ? 'scatter' : 'grouped';
      return predictorKind === 'continuous' ? 'curve' : 'props';
    }
    if (outcomeKind !== 'continuous') return 'text';
    if (predictorKind === 'binary' && partnerKind === 'binary') return 'grid';
    const oneBinOneCont =
      (predictorKind === 'binary' && partnerKind === 'continuous') ||
      (predictorKind === 'continuous' && partnerKind === 'binary');
    return oneBinOneCont ? 'slopes' : 'text';
  });

  // x mappers. sx: standardized x over ±X_CLAMP. cxCol: column-index x (idx + jit
  // spread) over nCols evenly spaced columns.
  const sx = (x: number) => {
    const c = Math.min(X_CLAMP, Math.max(-X_CLAMP, x));
    return PAD + ((c + X_CLAMP) / (2 * X_CLAMP)) * (W - 2 * PAD);
  };
  const cxCol = (x: number, nCols: number) => PAD + ((x + 0.5) / nCols) * (W - 2 * PAD);
  const syUnit = (p: number) => H - PAD - p * (H - 2 * PAD); // p in [0,1]
  // y -> pixel within an auto-fit [lo,hi] range (continuous-outcome charts).
  const syFit = (y: number, r: { lo: number; hi: number }) =>
    H - PAD - ((y - r.lo) / (r.hi - r.lo)) * (H - 2 * PAD);

  // ---- continuous-outcome chart data + their fitted y-ranges ----
  const scatter = $derived(mode === 'scatter' ? continuousScatter(beta, sampleIndex) : null);
  const scatterRange = $derived(
    scatter ? fitRange([...scatter.points.map((p) => p.y), scatter.trend.y1, scatter.trend.y2]) : null,
  );

  const groupMeans = $derived(predictorKind === 'factor' ? [0, ...levelBetas] : [0, beta]);
  const grouped = $derived(mode === 'grouped' ? groupedColumns(groupMeans, sampleIndex) : null);
  const groupedRange = $derived(
    grouped
      ? fitRange([...grouped.columns.flatMap((c) => c.points.map((p) => p.y)), ...grouped.columns.map((c) => c.mean)])
      : null,
  );

  const grid = $derived(mode === 'grid' ? gridColumns(partnerMainA, partnerMainB, beta, sampleIndex) : null);
  const gridRange = $derived(
    grid
      ? fitRange([
          ...grid.columns.flatMap((c) => c.points.map((p) => p.y)),
          ...grid.columns.map((c) => c.mean),
          ...grid.columns.flatMap((c) => (c.ref != null ? [c.ref] : [])),
        ])
      : null,
  );

  const slopes = $derived(mode === 'slopes' ? slopeScatter(partnerMainA, beta, binShift, sampleIndex) : null);
  const slopesRange = $derived(
    slopes
      ? fitRange([
          ...slopes.group0.points.map((p) => p.y),
          ...slopes.group1.points.map((p) => p.y),
          slopes.group0.line.y1,
          slopes.group0.line.y2,
          slopes.group1.line.y1,
          slopes.group1.line.y2,
        ])
      : null,
  );

  // ---- binary-outcome charts ----
  // Sigmoid + its Bernoulli 0/1 cloud (the data the curve is fitted to). The
  // cloud reacts to intercept (baseline) without resampling — dots flip 0/1 as
  // the baseline drags, which reads better than a reshuffle.
  const curve = $derived(mode === 'curve' ? logisticCurve(intercept, beta) : null);
  const bern = $derived(mode === 'curve' ? bernoulliScatter(intercept, beta, sampleIndex) : null);
  // Stacked proportion bar per group/level (pOne colored over pZero muted).
  const outcomeBars = $derived(mode === 'props' ? groupOutcomeBars(intercept, groupMeans) : null);
  const BERN_JIT_PX = 7; // one-sided inward vertical spread for the 0/1 dots

  const unitLabel = $derived(outcomeKind === 'binary' ? 'log-odds' : 'SD');
</script>

<figure class="rounded-md border border-[var(--border)] bg-[var(--card)] p-2">
  {#if mode === 'text'}
    <p class="p-4 text-sm text-[var(--muted-foreground)]">
      No 2-D illustration for this term. β = {beta.toFixed(2)} ({unitLabel}).
    </p>
  {:else}
    <svg viewBox="0 0 {W} {H}" role="img" aria-label="effect illustration" class="w-full">
      <!-- axes -->
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke={colors.border} />
      <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke={colors.border} />

      {#if scatter && scatterRange}
        {#each scatter.points as pt}
          <circle cx={sx(pt.x)} cy={syFit(pt.y, scatterRange)} r="2.5" fill={colors.chart1} opacity="0.55" />
        {/each}
        <line
          x1={sx(scatter.trend.x1)}
          y1={syFit(scatter.trend.y1, scatterRange)}
          x2={sx(scatter.trend.x2)}
          y2={syFit(scatter.trend.y2, scatterRange)}
          stroke={colors.chart2}
          stroke-width="2"
          stroke-dasharray="5 3"
        />
      {:else if grouped && groupedRange}
        {#each grouped.columns as col, idx}
          {@const fill = groupColors[idx % groupColors.length]}
          {#each col.points as pt}
            <circle cx={cxCol(pt.x, grouped.columns.length)} cy={syFit(pt.y, groupedRange)} r="2.2" {fill} opacity="0.5" />
          {/each}
          <line
            x1={cxCol(idx - 0.34, grouped.columns.length)}
            y1={syFit(col.mean, groupedRange)}
            x2={cxCol(idx + 0.34, grouped.columns.length)}
            y2={syFit(col.mean, groupedRange)}
            stroke={fill}
            stroke-width="3"
          />
        {/each}
      {:else if grid && gridRange}
        {#each grid.columns as col, idx}
          {@const fill = groupColors[idx % groupColors.length]}
          {#each col.points as pt}
            <circle cx={cxCol(pt.x, grid.columns.length)} cy={syFit(pt.y, gridRange)} r="2.2" {fill} opacity="0.5" />
          {/each}
          <line
            x1={cxCol(idx - 0.34, grid.columns.length)}
            y1={syFit(col.mean, gridRange)}
            x2={cxCol(idx + 0.34, grid.columns.length)}
            y2={syFit(col.mean, gridRange)}
            stroke={fill}
            stroke-width="3"
          />
          {#if col.ref != null}
            <line
              x1={cxCol(idx - 0.34, grid.columns.length)}
              y1={syFit(col.ref, gridRange)}
              x2={cxCol(idx + 0.34, grid.columns.length)}
              y2={syFit(col.ref, gridRange)}
              stroke={colors.mutedFg}
              stroke-width="2"
              stroke-dasharray="4 3"
            />
          {/if}
        {/each}
      {:else if slopes && slopesRange}
        {#each slopes.group0.points as pt}
          <circle cx={sx(pt.x)} cy={syFit(pt.y, slopesRange)} r="2.2" fill={colors.chart1} opacity="0.45" />
        {/each}
        {#each slopes.group1.points as pt}
          <circle cx={sx(pt.x)} cy={syFit(pt.y, slopesRange)} r="2.2" fill={colors.chart2} opacity="0.45" />
        {/each}
        <line x1={sx(slopes.group0.line.x1)} y1={syFit(slopes.group0.line.y1, slopesRange)} x2={sx(slopes.group0.line.x2)} y2={syFit(slopes.group0.line.y2, slopesRange)} stroke={colors.chart1} stroke-width="2.5" />
        <line x1={sx(slopes.group1.line.x1)} y1={syFit(slopes.group1.line.y1, slopesRange)} x2={sx(slopes.group1.line.x2)} y2={syFit(slopes.group1.line.y2, slopesRange)} stroke={colors.chart2} stroke-width="2.5" />
        <!-- legend: group 0 / group 1 by binary level -->
        <circle cx={W - PAD - 60} cy={PAD + 4} r="3" fill={colors.chart1} />
        <text x={W - PAD - 52} y={PAD + 7} font-size="9" fill={colors.fg}>group 0</text>
        <circle cx={W - PAD - 60} cy={PAD + 16} r="3" fill={colors.chart2} />
        <text x={W - PAD - 52} y={PAD + 19} font-size="9" fill={colors.fg}>group 1</text>
      {:else if curve}
        <!-- 0/1 outcomes drawn from the curve: 1s near the top, 0s near the bottom,
             jittered one-sided inward so they don't overplot the two y-levels. -->
        {#if bern}
          {#each bern.points as pt}
            {@const top = pt.outcome === 1}
            <circle
              cx={sx(pt.x)}
              cy={syUnit(pt.outcome) + (top ? 1 : -1) * Math.abs(pt.jit) * BERN_JIT_PX}
              r="2.2"
              fill={top ? colors.chart2 : colors.chart3}
              opacity="0.5"
            />
          {/each}
        {/if}
        <polyline
          points={curve.map((c) => `${sx(c.x)},${syUnit(c.p)}`).join(' ')}
          fill="none"
          stroke={colors.chart1}
          stroke-width="2"
        />
      {:else if outcomeBars}
        {@const n = outcomeBars.length}
        {@const barW = ((W - 2 * PAD) / n) * 0.55}
        {#each outcomeBars as bar, idx}
          {@const cx = cxCol(idx, n)}
          <!-- pZero (muted) at the bottom, pOne (colored) stacked on top; full bar = 1.0 -->
          <rect x={cx - barW / 2} y={syUnit(bar.pZero)} width={barW} height={syUnit(0) - syUnit(bar.pZero)} fill={colors.mutedFg} opacity="0.22" />
          <rect x={cx - barW / 2} y={syUnit(1)} width={barW} height={syUnit(bar.pZero) - syUnit(1)} fill={groupColors[idx % groupColors.length]} />
          <text x={cx} y={syUnit(1) - 3} font-size="9" text-anchor="middle" fill={colors.fg}>{Math.round(bar.pOne * 100)}%</text>
        {/each}
      {/if}
    </svg>
    <figcaption class="px-1 pt-1 text-xs text-[var(--muted-foreground)]">
      Illustration · β = {beta.toFixed(2)} ({unitLabel})
    </figcaption>
  {/if}
</figure>
