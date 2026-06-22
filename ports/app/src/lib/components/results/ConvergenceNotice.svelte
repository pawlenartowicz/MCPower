<script lang="ts">
  // Diagnostic badge shown after runs: flags low convergence, GLM baseline drift, high-τ̂ boundary
  // hits, GLMM Laplace-approximation bias, sparse-factor exclusions, and preflight/grid warnings.
  import type { RunTab } from '$lib/stores/run.svelte';
  import type { PowerResult, SampleSizeResult, EstimatorExtras } from '$lib/domain/result';
  import type { AppSpec } from '$lib/domain/app-spec';
  import { REPORT_CONFIG } from '$lib/configs/report-config';
  import { LIMITS } from '$lib/configs/app-config';
  import config from '$configs/config.json';

  interface Props { tab: RunTab; }
  const { tab }: Props = $props();

  const thresholds = REPORT_CONFIG.thresholds;
  // factor_min_level_count comes from limits section (not exposed via ReportConfig) — mirrors config.json limits.factor_min_level_count
  const FACTOR_MIN_LEVEL_COUNT: number = (config as { limits?: { factor_min_level_count?: number } }).limits?.factor_min_level_count ?? 5;

  const r = $derived(
    tab.kind === 'find-power' ? (tab.result as PowerResult) : null,
  );

  const ssr = $derived(
    tab.kind === 'find-sample-size' ? (tab.result as SampleSizeResult) : null,
  );

  const convergenceRate = $derived(r?.convergence_rate ?? 1);

  /** Requested GLM event probability from the run's spec (the value
   * set_baseline_probability would carry), or null when the outcome is not
   * binary. GLMM surfaces under family 'mixed' with a binary outcome, so it must
   * read the mixed source — covering only LogitSpec would leave GLMM drift dead. */
  function requestedBaseline(spec: AppSpec): number | null {
    if (spec.family === 'logit') return spec.baseline_probability;
    if (spec.family === 'mixed' && spec.outcome?.kind === 'binary') return spec.outcome.baseline_probability;
    return null;
  }

  /** Smallest cluster size at the evaluated N: explicit cluster_size, or
   * floor(N / n_clusters). null when the run is not clustered (mixed family). */
  function minClusterSize(spec: AppSpec, n: number): number | null {
    if (spec.family !== 'mixed') return null;
    const dim = spec.cluster_dim;
    return dim.kind === 'cluster_size' ? dim.value : Math.floor(n / dim.value);
  }

  // GLM baseline drift = |realized − requested|, requested read from the spec
  // (NOT baseline_prob_sum/baseline_prob_n, which equals realized → always 0).
  // Estimator must be 'glm' (covers plain GLM and GLMM). Mirrors Py/R diagnostic_warnings.
  const glmDrift = $derived((): number | null => {
    if (!r?.estimator_extras) return null;
    const ex = r.estimator_extras as EstimatorExtras;
    if (ex.estimator !== 'glm') return null;
    const realized = (ex as Extract<EstimatorExtras, { estimator: 'glm' }>).baseline_prob_realized;
    const requested = requestedBaseline(tab.spec);
    if (realized == null || requested == null) return null;
    return Math.abs(realized - requested);
  });

  // High-τ̂ boundary rate from the per-sim boundary_hit array (value 2), divided by
  // n_sims — estimator-agnostic, identical source of truth to Py/R's
  // boundary_hit_rate_high_tau. Benign τ̂=0 (value 1) is deliberately ignored.
  const highBoundary = $derived((): number | null => {
    if (!r) return null;
    const nSims = r.n_sims > 0 ? r.n_sims : 0;
    if (nSims === 0) return null;
    const highCount = (r.boundary_hit ?? []).filter((v) => v === 2).length;
    return highCount / nSims;
  });

  // Laplace-approximation bias for GLMM: large τ̂² with small clusters. Same
  // firing rule as Py/R _glmm_laplace_bias_warning (τ̂² > glmm_tau_sq_warn AND
  // min cluster size < recommended_rows_per_cluster).
  const laplaceBias = $derived((): { tau: number; minSize: number } | null => {
    if (!r?.estimator_extras) return null;
    const ex = r.estimator_extras as EstimatorExtras;
    if (ex.estimator !== 'glm') return null;
    const tau = (ex as Extract<EstimatorExtras, { estimator: 'glm' }>).tau_squared_hat_mean;
    if (tau == null) return null;
    const minSize = minClusterSize(tab.spec, r.n);
    if (minSize == null) return null;
    if (tau > thresholds.glmm_tau_sq_warn && minSize < LIMITS.recommended_rows_per_cluster) {
      return { tau, minSize };
    }
    return null;
  });

  const convergenceProblem = $derived(convergenceRate < thresholds.convergence_min);
  const glmDriftProblem = $derived(glmDrift() != null && glmDrift()! > thresholds.glm_baseline_drift_max);
  const boundaryProblem = $derived(highBoundary() != null && highBoundary()! > thresholds.lme_boundary_hit_max);
  const laplaceProblem = $derived(laplaceBias() != null);

  /** Factor names in spec order from var_types.
   * ANOVA runs are projected to family 'linear' by the adapter (app-spec-adapter.ts) so a
   * RunTab spec never carries family 'anova' at runtime; the guard below is for the type checker. */
  function factorNamesFromSpec(spec: AppSpec): string[] {
    if (spec.family === 'anova') return [];
    return spec.var_types
      .filter((v) => v.kind === 'factor')
      .map((v) => v.name);
  }

  interface ExclusionLine {
    name: string;
    rate: number;
    separation: boolean;
  }

  /** Exclusion advisory lines for the current result — rate above
   * factor_exclusion_max only (single-sourced from configs, mirrors Py/R's
   * factor_exclusion_max gate), computed once. */
  const exclusionLines = $derived((() => {
    const factorNames = factorNamesFromSpec(tab.spec);
    const lines: ExclusionLine[] = [];
    const thExcl = thresholds.factor_exclusion_max;

    if (r) {
      // find-power: flat counts per factor
      const excl = r.factor_exclusion_counts ?? [];
      const sep = r.factor_separation_counts ?? [];
      const nSims = r.n_sims > 0 ? r.n_sims : 1;
      for (let f = 0; f < excl.length; f++) {
        const rate = (excl[f] ?? 0) / nSims;
        if (rate > thExcl) {
          lines.push({ name: factorNames[f] ?? `Factor ${f + 1}`, rate, separation: false });
        }
      }
      for (let f = 0; f < sep.length; f++) {
        const rate = (sep[f] ?? 0) / nSims;
        if (rate > thExcl) {
          lines.push({ name: factorNames[f] ?? `Factor ${f + 1}`, rate, separation: true });
        }
      }
    } else if (ssr) {
      // find-sample-size: per-grid-point counts (number[][]); take max over grid points
      const grid = ssr.grid_or_trace;
      // n_sims is identical across grid points — read once from the first
      const nSims = grid[0]?.n_sims ?? 1;
      const exclGrid = ssr.factor_exclusion_counts ?? [];
      const sepGrid = ssr.factor_separation_counts ?? [];

      const nFactors = exclGrid[0]?.length ?? 0;
      for (let f = 0; f < nFactors; f++) {
        const maxRate = Math.max(...exclGrid.map((pt) => (pt[f] ?? 0) / (nSims > 0 ? nSims : 1)));
        if (maxRate > thExcl) {
          lines.push({ name: factorNames[f] ?? `Factor ${f + 1}`, rate: maxRate, separation: false });
        }
      }
      const nFactorsSep = sepGrid[0]?.length ?? 0;
      for (let f = 0; f < nFactorsSep; f++) {
        const maxRate = Math.max(...sepGrid.map((pt) => (pt[f] ?? 0) / (nSims > 0 ? nSims : 1)));
        if (maxRate > thExcl) {
          lines.push({ name: factorNames[f] ?? `Factor ${f + 1}`, rate: maxRate, separation: true });
        }
      }
    }

    return lines;
  })() as ExclusionLine[]);

  /** grid_warnings from whichever result kind is active — computed once. */
  const gridWarnings = $derived(
    r ? (r.grid_warnings ?? []) : ssr ? (ssr.grid_warnings ?? []) : [] as string[]
  );

  const hasProblem = $derived(convergenceProblem || glmDriftProblem || boundaryProblem || laplaceProblem);
  const hasExclusion = $derived(exclusionLines.length > 0);
  const hasGridWarnings = $derived(gridWarnings.length > 0);

  let expanded = $state(false);
</script>

{#if tab.kind === 'find-power' || hasExclusion || hasGridWarnings}
  <div data-testid="diagnostics-badge" class="mt-3">
    {#if tab.kind === 'find-power' && hasProblem}
      <button
        type="button"
        class="flex w-full items-center gap-2 rounded-md border border-amber-400 bg-amber-50 px-3 py-2 text-sm text-amber-900 hover:bg-amber-100"
        onclick={() => (expanded = !expanded)}
        aria-expanded={expanded}
      >
        <span class="inline-block h-2 w-2 flex-shrink-0 rounded-full bg-amber-500"></span>
        <span class="flex-1 text-left">Diagnostics — issues detected</span>
        <span class="text-xs">{expanded ? '▲' : '▼'}</span>
      </button>
      {#if expanded}
        <div class="mt-1 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900 space-y-1">
          {#if convergenceProblem}
            <p>
              Convergence {(convergenceRate * 100).toFixed(1)}% — below {(thresholds.convergence_min * 100).toFixed(0)}%.
              Mixed-model fits may be unstable; consider more observations per cluster or more simulations.
            </p>
          {/if}
          {#if glmDriftProblem}
            <p>
              GLM baseline probability drifted by {(glmDrift()! * 100).toFixed(1)}% from target
              (threshold {(thresholds.glm_baseline_drift_max * 100).toFixed(0)}%).
            </p>
          {/if}
          {#if boundaryProblem}
            <p>
              High-τ̂ boundary hits {(highBoundary()! * 100).toFixed(1)}% — above {(thresholds.lme_boundary_hit_max * 100).toFixed(0)}%.
              Random-effects variance estimate pinned at a bad boundary; consider more simulations or a simpler model.
            </p>
          {/if}
          {#if laplaceProblem}
            <p>
              Laplace-approximation bias likely: estimated random-intercept variance τ̂² = {laplaceBias()!.tau.toFixed(2)}
              exceeds {thresholds.glmm_tau_sq_warn.toFixed(2)} with small clusters
              (min cluster size {laplaceBias()!.minSize} &lt; {LIMITS.recommended_rows_per_cluster}).
              GLMM power may be optimistic — interpret with caution or increase cluster size.
            </p>
          {/if}
        </div>
      {/if}
    {:else if tab.kind === 'find-power'}
      <div class="flex items-center gap-2 text-xs text-muted-foreground">
        <span class="inline-block h-2 w-2 rounded-full bg-green-500"></span>
        <span>Diagnostics — all clear</span>
      </div>
    {/if}

    {#if hasExclusion}
      <div class="mt-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900 space-y-1">
        {#each exclusionLines as line, i (i)}
          <p>
            {#if line.separation}
              {line.name} dropped by the separation fallback in {(line.rate * 100).toFixed(1)}% of simulations
            {:else}
              {line.name} excluded in {(line.rate * 100).toFixed(1)}% of simulations (a factor level below the {FACTOR_MIN_LEVEL_COUNT}-observation minimum)
            {/if}
          </p>
        {/each}
      </div>
    {/if}

    {#if hasGridWarnings}
      <div class="mt-2 rounded-md border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-900 space-y-1">
        {#each gridWarnings as warning (warning)}
          <p>{warning}</p>
        {/each}
      </div>
    {/if}
  </div>
{/if}
