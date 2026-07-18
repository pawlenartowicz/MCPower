<script lang="ts">
  // Cluster cards for mixed-effects designs: one card per random-effect term in
  // the formula (the formula is the single source of cluster structure —
  // `(1|a)+(1|b)` → crossed, `(1|a/b)` → nested, `(1+x|g)` pre-checks a random
  // slope). The first term is the primary cluster (name, ICC, sizing, ⚙
  // Advanced); later terms get a slim card with a crossed/nested badge, ICC,
  // and an n input. Engine constraints are surfaced here: extra groupings force
  // "by n clusters", and at most one nested grouping is supported.
  import { untrack } from 'svelte';
  import { NumberInput } from '$lib/components/ui/number-input';
  import { Button } from '$lib/components/ui/button';
  import Settings2 from '@lucide/svelte/icons/settings-2';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import ClusterAdvancedDialog from './ClusterAdvancedDialog.svelte';
  import ExtraGroupingAdvancedDialog from './ExtraGroupingAdvancedDialog.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import { parsedFormulaStore, toClusterTerms } from '$lib/stores/parsed-formula.svelte';
  import { LIMITS } from '$lib/configs/app-config';
  import { EXTRA_GROUPING_DEFAULTS, SLOPE_DEFAULTS, mixedOutcomeKind } from '$lib/domain/family';

  // input floor = engine hard floor; 'reliable'/'recommended' are advisory.
  const MIN_OBS_PER_CLUSTER = LIMITS.min_rows_per_cluster;           // 2 (was hardcoded 5, mislabeled "hard minimum")
  const RELIABLE_OBS_PER_CLUSTER = LIMITS.reliable_rows_per_cluster; // 5 (advisory)
  const RECOMMENDED_OBS_PER_CLUSTER = LIMITS.recommended_rows_per_cluster; // 10

  const cfg = $derived(familyStore.byFamily[familyStore.active]);

  let advancedOpen = $state(false);
  // One dialog instance keyed by the open extra grouping's index (mirrors the
  // primary's `advancedOpen`); the card buttons set the index then open it.
  let extraAdvancedOpen = $state(false);
  let extraAdvancedIndex = $state(0);

  // Cluster terms mirror the formula's random effects (read-only structure).
  // getStable: cards must not flash away (and the reconcile below must not act
  // on an empty term list) during the parse round-trip after a formula edit.
  const terms = $derived.by(() => {
    const result = parsedFormulaStore.getStable(cfg.formula).result;
    return result ? toClusterTerms(result) : [];
  });
  const primary = $derived(terms[0] ?? null);
  const extraTerms = $derived(terms.slice(1));
  const nestedCount = $derived(extraTerms.filter((t) => t.parent !== null).length);
  // Poisson has no residual variance to scale an ICC against (family.ts docs
  // this on ClusterConfig.tauSquared) — both the primary and every extra
  // grouping take a raw τ² instead, so the ICC input is meaningless here.
  const isPoisson = $derived(mixedOutcomeKind(cfg.cluster) === 'poisson');

  // Reconcile formula-derived structure into cfg.cluster, preserving user
  // values by cluster name (mirrors the predictor reconcile in PredictorCards).
  // The write is untracked so the effect re-runs only on formula changes.
  $effect(() => {
    const t = terms;
    untrack(() => {
      const cl = cfg.cluster;
      if (!cl || t.length === 0) return;
      cl.clusterName = t[0]!.cluster;

      const prevByName = new Map((cl.extraGroupings ?? []).map((g) => [g.clusterName, g]));
      const next = t.slice(1).map((term) => {
        const prev = prevByName.get(term.cluster);
        const relation = term.parent !== null ? ('nested' as const) : ('crossed' as const);
        // Slopes mirror this term's `(1+x|g)` vars exactly — same rule as the
        // primary below (params preserved by name, stale vars dropped).
        const prevSlopes = new Map((prev?.slopes ?? []).map((s) => [s.predictorName, s]));
        const slopes = term.slopeVars.map(
          (v) =>
            prevSlopes.get(v) ?? {
              predictorName: v,
              slopeVariance: SLOPE_DEFAULTS.variance,
              slopeInterceptCorr: SLOPE_DEFAULTS.corr,
            },
        );
        return {
          clusterName: term.cluster,
          relation,
          icc: prev?.icc ?? EXTRA_GROUPING_DEFAULTS.icc,
          tauSquared: prev?.tauSquared ?? EXTRA_GROUPING_DEFAULTS.tauSquared,
          n:
            prev?.n ??
            (relation === 'nested' ? EXTRA_GROUPING_DEFAULTS.nestedN : EXTRA_GROUPING_DEFAULTS.crossedN),
          slopes,
        };
      });
      const current = cl.extraGroupings ?? [];
      const same =
        current.length === next.length &&
        next.every((g, i) => {
          const p = current[i]!;
          return (
            p.clusterName === g.clusterName &&
            p.relation === g.relation &&
            p.icc === g.icc &&
            p.tauSquared === g.tauSquared &&
            p.n === g.n &&
            // Slope SET identity (by predictor name); params edited in-place via
            // the advanced popup, so a param change must NOT force a rewrite.
            (p.slopes?.length ?? 0) === g.slopes.length &&
            g.slopes.every((s, j) => p.slopes?.[j]?.predictorName === s.predictorName)
          );
        });
      if (!same) cl.extraGroupings = next;

      // Random slopes are formula-driven: cl.slopes mirrors the primary term's
      // slope vars exactly (params preserved by name; entries whose var left
      // the formula are dropped — stale ones otherwise error in the engine
      // forever). The advanced popup edits only variance/corr, never the set.
      const formulaSlopes = t[0]?.slopeVars ?? [];
      const existing = cl.slopes ?? [];
      const byName = new Map(existing.map((s) => [s.predictorName, s]));
      const sameSlopes =
        existing.length === formulaSlopes.length &&
        formulaSlopes.every((v, i) => existing[i]?.predictorName === v);
      if (!sameSlopes) {
        cl.slopes = formulaSlopes.map(
          (v) =>
            byName.get(v) ?? {
              predictorName: v,
              slopeVariance: SLOPE_DEFAULTS.variance,
              slopeInterceptCorr: SLOPE_DEFAULTS.corr,
            },
        );
      }

      // Extra groupings only run with a fixed cluster count on the primary.
      if (t.length > 1 && cl.dimKind === 'cluster_size') cl.dimKind = 'n_clusters';
    });
  });

  // Derived obs/cluster against the Find-power sample size, for the warning.
  const derivedClusterSize = $derived.by(() => {
    if (!cfg.cluster) return null;
    if (cfg.cluster.dimKind === 'cluster_size') return cfg.cluster.clusterSize;
    // No meaningful obs/cluster until the user has entered at least 1 cluster.
    if (cfg.cluster.nClusters < 1) return null;
    const n = cfg.findPower?.n ?? 0;
    return Math.floor(n / cfg.cluster.nClusters);
  });

  const sizeWarning = $derived.by(() => {
    const s = derivedClusterSize;
    if (s == null) return null;
    if (s < MIN_OBS_PER_CLUSTER) return { tone: 'error', msg: `~${s} obs/cluster — below the engine hard minimum of ${MIN_OBS_PER_CLUSTER}; the engine will fail.` };
    if (s < RELIABLE_OBS_PER_CLUSTER) return { tone: 'warn', msg: `~${s} obs/cluster — below the reliable threshold of ${RELIABLE_OBS_PER_CLUSTER}; estimates may be unreliable.` };
    if (s < RECOMMENDED_OBS_PER_CLUSTER) return { tone: 'warn', msg: `~${s} obs/cluster — below the recommended ${RECOMMENDED_OBS_PER_CLUSTER}; estimates may be unstable.` };
    return null;
  });
</script>

{#if cfg.cluster}
  <div class="space-y-2">
    <div class="flex items-center gap-2">
      <span class="text-sm font-semibold">Clusters</span>
      <InfoIcon tipKey="clusterConfig" />
    </div>

    {#if !primary}
      <p class="text-xs text-muted-foreground" data-testid="cluster-name">
        — add (1|name) to the formula —
      </p>
    {:else}
      <!-- Primary cluster card -->
      <div class="rounded-md border border-border bg-card px-2.5 py-2">
        <div class="flex flex-wrap items-center gap-2">
          <span class="font-mono text-[13px] font-semibold" data-testid="cluster-name">{primary.cluster}</span>
          <span class="rounded-full border border-primary/50 px-2 py-0.5 text-[11px] text-primary">primary</span>
          {#if isPoisson}
            <span class="text-xs text-muted-foreground">τ²</span>
            <NumberInput
              id="cluster-tau-squared"
              class="h-7 w-28 shrink-0"
              step={0.05}
              min={0}
              value={cfg.cluster.tauSquared ?? 0}
              oninput={(v) => { if (cfg.cluster) cfg.cluster.tauSquared = v; }}
            />
          {:else}
            <span class="text-xs text-muted-foreground">ICC</span>
            <InfoIcon tipKey="icc" />
            <NumberInput
              id="cluster-icc"
              class="h-7 w-28 shrink-0"
              step={0.05}
              min={0}
              max={0.99}
              value={cfg.cluster.icc}
              oninput={(v) => { if (cfg.cluster) cfg.cluster.icc = v; }}
            />
          {/if}
          <Button
            variant="outline"
            size="sm"
            class="ml-auto h-7 shrink-0 px-2 text-xs"
            data-testid="cluster-advanced"
            onclick={() => (advancedOpen = true)}
          >
            <Settings2 class="mr-1 h-3.5 w-3.5" /> Advanced
          </Button>
        </div>

        <div role="group" aria-label="Cluster dimension" class="mt-2 flex flex-wrap items-center gap-2 text-sm">
          <span class="text-xs text-muted-foreground">Cluster dimension</span>
          <InfoIcon tipKey="clusterDimKind" />
          <button
            type="button"
            data-testid="dim-n-clusters"
            class="rounded px-3 py-1 {cfg.cluster.dimKind === 'n_clusters'
              ? 'bg-primary text-primary-foreground'
              : 'bg-muted text-muted-foreground hover:bg-muted/70'}"
            onclick={() => { if (cfg.cluster) cfg.cluster.dimKind = 'n_clusters'; }}
          >by n clusters</button>
          <button
            type="button"
            data-testid="dim-cluster-size"
            disabled={extraTerms.length > 0}
            title={extraTerms.length > 0
              ? 'Extra grouping factors need a fixed number of primary clusters'
              : undefined}
            class="rounded px-3 py-1 {cfg.cluster.dimKind === 'cluster_size'
              ? 'bg-primary text-primary-foreground'
              : 'bg-muted text-muted-foreground hover:bg-muted/70'} disabled:cursor-not-allowed disabled:opacity-50"
            onclick={() => { if (cfg.cluster) cfg.cluster.dimKind = 'cluster_size'; }}
          >by cluster size</button>
          {#if cfg.cluster.dimKind === 'n_clusters'}
            <NumberInput
              id="cluster-n"
              aria-label="N clusters"
              class="h-7 w-24 shrink-0"
              step={1}
              min={2}
              value={cfg.cluster.nClusters}
              oninput={(v) => { if (cfg.cluster) cfg.cluster.nClusters = v; }}
            />
          {:else}
            <NumberInput
              id="cluster-size"
              aria-label="Observations per cluster"
              class="h-7 w-24 shrink-0"
              step={1}
              min={MIN_OBS_PER_CLUSTER}
              value={cfg.cluster.clusterSize}
              oninput={(v) => { if (cfg.cluster) cfg.cluster.clusterSize = v; }}
            />
          {/if}
        </div>
        {#if sizeWarning}
          <p
            data-testid="cluster-size-warning"
            class="mt-1 text-xs {sizeWarning.tone === 'error' ? 'text-red-600' : 'text-amber-700'}"
          >{sizeWarning.msg}</p>
        {/if}
      </div>

      <!-- Extra grouping cards (crossed / nested from formula syntax) -->
      {#each cfg.cluster.extraGroupings ?? [] as extra, i (extra.clusterName)}
        <div class="rounded-md border border-border bg-card px-2.5 py-2" data-testid={`cluster-extra-${extra.clusterName}`}>
          <div class="flex flex-wrap items-center gap-2">
            <span class="font-mono text-[13px] font-semibold">{extra.clusterName}</span>
            <span class="rounded-full border px-2 py-0.5 text-[11px] {extra.relation === 'nested'
              ? 'border-violet-500/50 text-violet-600 dark:text-violet-400'
              : 'border-sky-500/50 text-sky-600 dark:text-sky-400'}">{extra.relation}</span>
            {#if isPoisson}
              <span class="text-xs text-muted-foreground">τ²</span>
              <NumberInput
                class="h-7 w-24 shrink-0"
                step={0.05}
                min={0}
                value={extra.tauSquared ?? 0}
                oninput={(v) => {
                  const g = cfg.cluster?.extraGroupings?.[i];
                  if (g) g.tauSquared = v;
                }}
              />
            {:else}
              <span class="text-xs text-muted-foreground">ICC</span>
              <NumberInput
                class="h-7 w-24 shrink-0"
                step={0.05}
                min={0}
                max={0.99}
                value={extra.icc}
                oninput={(v) => {
                  const g = cfg.cluster?.extraGroupings?.[i];
                  if (g) g.icc = v;
                }}
              />
            {/if}
            <span class="text-xs text-muted-foreground">{extra.relation === 'nested' ? 'n per parent' : 'n clusters'}</span>
            <NumberInput
              class="h-7 w-24 shrink-0"
              step={1}
              min={2}
              value={extra.n}
              oninput={(v) => {
                const g = cfg.cluster?.extraGroupings?.[i];
                if (g) g.n = v;
              }}
            />
            <Button
              variant="outline"
              size="sm"
              class="ml-auto h-7 shrink-0 px-2 text-xs"
              data-testid={`cluster-extra-advanced-${extra.clusterName}`}
              onclick={() => { extraAdvancedIndex = i; extraAdvancedOpen = true; }}
            >
              <Settings2 class="mr-1 h-3.5 w-3.5" /> Advanced
            </Button>
          </div>
        </div>
      {/each}
      {#if nestedCount > 1}
        <p class="text-xs text-red-600" data-testid="nested-limit-error">
          At most one nested grouping is supported — keep a single (1|parent/child) term.
        </p>
      {/if}
    {/if}
  </div>

  <ClusterAdvancedDialog bind:open={advancedOpen} />
  <ExtraGroupingAdvancedDialog bind:open={extraAdvancedOpen} groupingIndex={extraAdvancedIndex} />
{/if}
