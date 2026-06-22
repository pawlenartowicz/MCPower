// Info-icon tip registry: every GUI help popover is backed by a leaf fragment of
// a documentation vault page, single-sourced via the $docs alias and raw-imported
// at build (dev hot-reloads on edit). The popover shows the page's (or section's)
// first paragraph; "see more →" links out to the published page. Explicit ?raw
// imports — glob + alias is unreliable.
import modelSpec from '$docs/concepts/model-specification.md?raw';
import variableTypes from '$docs/concepts/variable-types.md?raw';
import effectSizes from '$docs/concepts/effect-sizes.md?raw';
import correlations from '$docs/concepts/correlations.md?raw';
import mixedEffects from '$docs/concepts/mixed-effects.md?raw';
import multipleTesting from '$docs/concepts/multiple-testing.md?raw';
import conceptsIndex from '$docs/concepts/index.md?raw';
import simulationSettings from '$docs/concepts/simulation-settings.md?raw';
import scenarioAnalysis from '$docs/concepts/scenario-analysis.md?raw';
import uploadConcepts from '$docs/concepts/upload-data.md?raw';
import requiredSampleSize from '$docs/concepts/required-sample-size.md?raw';
import uploadApp from '$docs/tutorial-app/upload-data.md?raw';
import regressionApp from '$docs/tutorial-app/regression.md?raw';
import indexApp from '$docs/tutorial-app/index.md?raw';

/** A tip's backing source: the vault page (`doc`), an optional section `anchor`,
 *  and the raw Markdown the popover paragraph is extracted from. */
export interface TipSource {
  doc: string;
  anchor?: string;
  md: string;
}

export const TIP_SOURCES: Record<string, TipSource> = {
  // --- Per-knob / formula / core concept tips ---
  formula:             { doc: 'concepts/model-specification', md: modelSpec },
  variableTypes:       { doc: 'concepts/variable-types', md: variableTypes },
  effects:             { doc: 'concepts/effect-sizes', md: effectSizes },
  correlations:        { doc: 'concepts/correlations', md: correlations },
  clusterConfig:       { doc: 'concepts/mixed-effects', anchor: 'the-icc', md: mixedEffects },
  baselineProbability: { doc: 'concepts/effect-sizes', anchor: 'baseline-probability', md: effectSizes },
  anovaFactors:        { doc: 'concepts/variable-types', anchor: 'factors-and-the-reference-level', md: variableTypes },
  targetPower:         { doc: 'concepts/index', md: conceptsIndex },
  tests:               { doc: 'concepts/multiple-testing', anchor: 'choosing-which-tests-to-target', md: multipleTesting },
  advanced:            { doc: 'concepts/simulation-settings', md: simulationSettings },
  uploadedType:        { doc: 'tutorial-app/upload-data', anchor: 'matched-predictors-lock-to-the-data', md: uploadApp },

  // --- Per-knob tips (model + run controls) ---
  correction:               { doc: 'concepts/multiple-testing', anchor: 'available-corrections', md: multipleTesting },
  uploadMode:               { doc: 'concepts/upload-data', anchor: 'three-modes', md: uploadConcepts },
  heteroskedasticityDriver: { doc: 'concepts/variable-types', anchor: 'residuals-and-heteroskedasticity', md: variableTypes },
  residualDistribution:     { doc: 'concepts/variable-types', anchor: 'residual-distribution', md: variableTypes },
  icc:                      { doc: 'concepts/mixed-effects', anchor: 'the-icc', md: mixedEffects },
  clusterDimKind:           { doc: 'concepts/mixed-effects', anchor: 'cluster-sizing-regimes', md: mixedEffects },
  randomSlopes:             { doc: 'concepts/mixed-effects', anchor: 'random-slopes', md: mixedEffects },
  clusterLevelVars:         { doc: 'concepts/mixed-effects', anchor: 'cluster-level-predictors', md: mixedEffects },
  contrasts:                { doc: 'concepts/multiple-testing', anchor: 'contrasts-and-post-hoc', md: multipleTesting },
  testFormulaOverride:      { doc: 'concepts/model-specification', anchor: 'test-formula-misspecification', md: modelSpec },
  sampledShares:            { doc: 'concepts/scenario-analysis', anchor: 'factor-allocation', md: scenarioAnalysis },
  outcomeType:              { doc: 'concepts/effect-sizes', anchor: 'continuous-vs-binary-outcomes', md: effectSizes },

  // --- Scenario knobs (Settings → Scenarios tab) ---
  scenHeterogeneity:      { doc: 'concepts/scenario-analysis', anchor: 'heterogeneity', md: scenarioAnalysis },
  scenHeteroskedasticity: { doc: 'concepts/scenario-analysis', anchor: 'heteroskedasticity', md: scenarioAnalysis },
  scenCorrNoise:          { doc: 'concepts/scenario-analysis', anchor: 'correlation-noise', md: scenarioAnalysis },
  scenDistChange:         { doc: 'concepts/scenario-analysis', anchor: 'distribution-swaps', md: scenarioAnalysis },
  scenResidChange:        { doc: 'concepts/scenario-analysis', anchor: 'residual-swaps', md: scenarioAnalysis },
  scenResidDf:            { doc: 'concepts/scenario-analysis', anchor: 'residual-swaps', md: scenarioAnalysis },
  scenReDist:             { doc: 'concepts/scenario-analysis', anchor: 'mixed-model-knobs', md: scenarioAnalysis },
  scenReDf:               { doc: 'concepts/scenario-analysis', anchor: 'mixed-model-knobs', md: scenarioAnalysis },
  scenIccNoise:           { doc: 'concepts/scenario-analysis', anchor: 'mixed-model-knobs', md: scenarioAnalysis },

  // --- Section tips (run cards / tabs) ---
  sampleSizeSearch:    { doc: 'concepts/required-sample-size', anchor: 'the-search-grid', md: requiredSampleSize },
  findPowerN:          { doc: 'tutorial-app/regression', anchor: 'find-power-at-a-sample-size', md: regressionApp },
  exportPlot:          { doc: 'tutorial-app/index', anchor: 'export', md: indexApp },
};
