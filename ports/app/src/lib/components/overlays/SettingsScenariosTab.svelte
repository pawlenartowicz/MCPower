<script lang="ts">
  // Settings tab for editing per-scenario robustness parameters (heterogeneity, residuals, LME random effects); edits this device's copy of configs/scenarios.json with a Reset-to-bundled option.
  import { Label } from '$lib/components/ui/label';
  import { NumberInput } from '$lib/components/ui/number-input';
  import { Button } from '$lib/components/ui/button';
  import { Switch } from '$lib/components/ui/switch';
  import { Tabs, TabsContent, TabsList, TabsTrigger } from '$lib/components/ui/tabs';
  import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
  } from '$lib/components/ui/select';
  import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';
  import { scenariosStore } from '$lib/stores/scenarios.svelte';
  import type { RandomEffectDist } from '$lib/configs/scenarios';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';

  const HINTS: Record<string, string> = {
    optimistic: 'Best-case — distributions clean, residuals near normal.',
    realistic: 'Default — moderate heterogeneity and non-normal residuals.',
    doomer: 'Conservative — strong heterogeneity, heavy tails.',
  };

  const RANDOM_EFFECT_DISTS: RandomEffectDist[] = ['normal', 'heavy_tailed'];

  const DIST_LABELS: Record<string, string> = {
    normal: 'Normal',
    heavy_tailed: 'Heavy-tailed',
  };

  let scenarioTab = $state(scenariosStore.scenarios[0]?.name ?? 'optimistic');
</script>

<div class="space-y-4 px-1 pt-1">
  <div class="flex items-baseline justify-between gap-3">
    <p class="text-xs text-muted-foreground">
      Scenarios are {sharedPrefs.scenariosEnabled ? 'on' : 'off'} — toggle in the status bar.
      Edits below modify this device's copy of <code>configs/scenarios.json</code>;
      the default values are restored via Reset.
    </p>
    <Button
      variant="ghost"
      size="sm"
      class="shrink-0 text-xs"
      onclick={() => scenariosStore.resetToBundled()}
    >
      Reset to default
    </Button>
  </div>

  <Tabs bind:value={scenarioTab}>
    <TabsList>
      {#each scenariosStore.scenarios as s (s.name)}
        <TabsTrigger value={s.name} class="capitalize">{s.name}</TabsTrigger>
      {/each}
    </TabsList>
    {#each scenariosStore.scenarios as s (s.name)}
      <TabsContent value={s.name}>
        <fieldset class="space-y-3 pt-2" disabled={!sharedPrefs.scenariosEnabled}>
          <p class="text-xs text-muted-foreground">{HINTS[s.name] ?? ''}</p>

          <div class="flex items-center justify-between gap-3">
            <div class="flex items-center gap-1">
              <Label for="{s.name}-heterogeneity">Heterogeneity</Label>
              <InfoIcon tipKey="scenHeterogeneity" />
            </div>
            <NumberInput
              id="{s.name}-heterogeneity"
              class="w-28"
              step={0.05}
              min={0}
              value={s.heterogeneity}
              oninput={(n: number) => scenariosStore.update(s.name, { heterogeneity: n })}
            />
          </div>
          <div class="flex items-center justify-between gap-3">
            <div class="flex items-center gap-1">
              <Label for="{s.name}-heteroskedasticity-ratio">Heteroskedasticity (variance ratio λ)</Label>
              <InfoIcon tipKey="scenHeteroskedasticity" />
            </div>
            <NumberInput
              id="{s.name}-heteroskedasticity-ratio"
              class="w-28"
              step={0.5}
              min={1}
              value={s.heteroskedasticity_ratio}
              oninput={(n: number) => scenariosStore.update(s.name, { heteroskedasticity_ratio: n })}
            />
          </div>
          <div class="flex items-center justify-between gap-3">
            <div class="flex items-center gap-1">
              <Label for="{s.name}-corr-noise">Correlation noise SD</Label>
              <InfoIcon tipKey="scenCorrNoise" />
            </div>
            <NumberInput
              id="{s.name}-corr-noise"
              class="w-28"
              step={0.05}
              min={0}
              value={s.correlation_noise_sd}
              oninput={(n: number) => scenariosStore.update(s.name, { correlation_noise_sd: n })}
            />
          </div>
          <div class="flex items-center justify-between gap-3">
            <div class="flex items-center gap-1">
              <Label for="{s.name}-dist-change-p">Distribution change prob.</Label>
              <InfoIcon tipKey="scenDistChange" />
            </div>
            <NumberInput
              id="{s.name}-dist-change-p"
              class="w-28"
              step={0.05}
              min={0}
              max={1}
              value={s.distribution_change_prob}
              oninput={(n: number) =>
                scenariosStore.update(s.name, { distribution_change_prob: n })}
            />
          </div>
          <div class="flex items-center justify-between gap-3">
            <div class="flex items-center gap-1">
              <Label for="{s.name}-resid-change-p">Residual change prob.</Label>
              <InfoIcon tipKey="scenResidChange" />
            </div>
            <NumberInput
              id="{s.name}-resid-change-p"
              class="w-28"
              step={0.05}
              min={0}
              max={1}
              value={s.residual_change_prob}
              oninput={(n: number) =>
                scenariosStore.update(s.name, { residual_change_prob: n })}
            />
          </div>
          <div class="flex items-center justify-between gap-3">
            <div class="flex items-center gap-1">
              <Label for="{s.name}-resid-df">Residual df</Label>
              <InfoIcon tipKey="scenResidDf" />
            </div>
            <NumberInput
              id="{s.name}-resid-df"
              class="w-28"
              step={1}
              min={1}
              value={s.residual_df}
              oninput={(n: number) => scenariosStore.update(s.name, { residual_df: n })}
            />
          </div>

          <div class="flex items-center justify-between gap-3">
            <div>
              <div class="flex items-center gap-1">
                <Label for="{s.name}-sampled-shares">Sample factor shares</Label>
                <InfoIcon tipKey="sampledShares" />
              </div>
              <p class="text-xs text-muted-foreground">
                Default for every factor — draw level counts multinomially instead of allocating
                exactly. Individual predictors can override.
              </p>
            </div>
            <Switch
              id="{s.name}-sampled-shares"
              checked={s.sampled_factor_proportions}
              onCheckedChange={(v: boolean) =>
                scenariosStore.update(s.name, { sampled_factor_proportions: v })}
            />
          </div>

          <div class="space-y-3 rounded-md border border-border/60 p-2">
            <div class="text-xs font-medium text-muted-foreground">LME random effect</div>
            <div class="flex items-center justify-between gap-3">
              <div class="flex items-center gap-1">
                <Label for="{s.name}-re-dist">Distribution</Label>
                <InfoIcon tipKey="scenReDist" />
              </div>
              <Select
                type="single"
                value={s.lme.random_effect_dist}
                onValueChange={(v: string) => {
                  if (RANDOM_EFFECT_DISTS.includes(v as RandomEffectDist)) {
                    scenariosStore.updateLme(s.name, { random_effect_dist: v as RandomEffectDist });
                  }
                }}
              >
                <SelectTrigger class="w-40">{DIST_LABELS[s.lme.random_effect_dist] ?? s.lme.random_effect_dist}</SelectTrigger>
                <SelectContent>
                  {#each RANDOM_EFFECT_DISTS as d (d)}
                    <SelectItem value={d}>{DIST_LABELS[d] ?? d}</SelectItem>
                  {/each}
                </SelectContent>
              </Select>
            </div>
            <div class="flex items-center justify-between gap-3">
              <div class="flex items-center gap-1">
                <Label for="{s.name}-re-df">df</Label>
                <InfoIcon tipKey="scenReDf" />
              </div>
              <NumberInput
                id="{s.name}-re-df"
                class="w-28"
                step={1}
                min={1}
                value={s.lme.random_effect_df}
                oninput={(n: number) => scenariosStore.updateLme(s.name, { random_effect_df: n })}
              />
            </div>
            <div class="flex items-center justify-between gap-3">
              <div class="flex items-center gap-1">
                <Label for="{s.name}-icc-noise">ICC noise SD</Label>
                <InfoIcon tipKey="scenIccNoise" />
              </div>
              <NumberInput
                id="{s.name}-icc-noise"
                class="w-28"
                step={0.05}
                min={0}
                value={s.lme.icc_noise_sd}
                oninput={(n: number) => scenariosStore.updateLme(s.name, { icc_noise_sd: n })}
              />
            </div>
          </div>
        </fieldset>
      </TabsContent>
    {/each}
  </Tabs>
</div>
