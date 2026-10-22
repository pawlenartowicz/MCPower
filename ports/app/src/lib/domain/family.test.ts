import { describe, expect, it } from 'vitest';
import { defaultFamilyConfig, EXTRA_GROUPING_DEFAULTS, FAMILIES, FAMILY_LABEL, SELECTABLE_FAMILIES, mixedDefaultSimulations } from './family';

describe('family domain', () => {
  it('reads the nested-grouping default from the shared config', () => {
    expect(EXTRA_GROUPING_DEFAULTS.nestedN).toBe(2);
  });

  it('gives a mixed model its sim default by outcome: 800 Gaussian, 400 binary/Poisson', () => {
    expect(defaultFamilyConfig('mixed').advanced.simulations).toBe(800);
    expect(mixedDefaultSimulations('linear')).toBe(800);
    expect(mixedDefaultSimulations('logit')).toBe(400);
    expect(mixedDefaultSimulations('probit')).toBe(400);
    expect(mixedDefaultSimulations('poisson')).toBe(400);
  });

  it('lists three entrypoints in spec order', () => {
    expect(FAMILIES).toEqual(['anova', 'regression', 'mixed']);
  });

  it('labels every entrypoint', () => {
    // Pin the exact labels, not just truthiness — the loop's `toBeTruthy()` passed
    // for any non-empty value (a mislabel to 'X' survived it).
    expect(FAMILY_LABEL).toEqual({
      anova: 'ANOVA',
      regression: 'Regression',
      mixed: 'Mixed effects',
    });
  });

  it('defaults Regression config with empty fields', () => {
    const cfg = defaultFamilyConfig('regression');
    expect(cfg.formula).toBe('');
    expect(cfg.variables).toEqual([]);
    expect(cfg.effects).toEqual([]);
    expect(cfg.targetPower).toBe(80);
    expect(cfg.alpha).toBe(0.05);
  });

  it('attaches a cluster block to Mixed only', () => {
    expect(defaultFamilyConfig('mixed').cluster).toBeDefined();
    expect(defaultFamilyConfig('regression').cluster).toBeUndefined();
  });

  it('attaches a baselineProbability to Regression only', () => {
    expect(defaultFamilyConfig('regression').baselineProbability).toBe(0.2);
    expect(defaultFamilyConfig('anova').baselineProbability).toBeUndefined();
    expect(defaultFamilyConfig('mixed').baselineProbability).toBeUndefined();
  });

  it('attaches a baselineRate to Regression only', () => {
    expect(defaultFamilyConfig('regression').baselineRate).toBe(2.0);
    expect(defaultFamilyConfig('anova').baselineRate).toBeUndefined();
    expect(defaultFamilyConfig('mixed').baselineRate).toBeUndefined();
  });

  it('defaults the estimation control to Fast (wald_se rx, agq 1)', () => {
    const adv = defaultFamilyConfig('regression').advanced;
    expect(adv.wald_se).toBe('rx');
    expect(adv.agq).toBe(1);
  });

  it('defaults reportOverall to true for all families', () => {
    expect(defaultFamilyConfig('regression').reportOverall).toBe(true);
    expect(defaultFamilyConfig('mixed').reportOverall).toBe(true);
    expect(defaultFamilyConfig('anova').reportOverall).toBe(true);
  });

  it('defaults contrasts to empty array for all families', () => {
    for (const f of FAMILIES) {
      expect(defaultFamilyConfig(f).contrasts).toEqual([]);
    }
  });

  it('defaults Regression to testing all effects', () => {
    expect(defaultFamilyConfig('regression').tests).toEqual({ kind: 'all' });
  });

  it('ANOVA default uses effects-kind tests with no variables', () => {
    const cfg = defaultFamilyConfig('anova');
    expect(cfg.tests).toEqual({ kind: 'effects', names: [] });
    expect(cfg.variables).toEqual([]);
  });

  it('makes Regression, ANOVA, and Mixed selectable', () => {
    expect(SELECTABLE_FAMILIES).toContain('regression');
    expect(SELECTABLE_FAMILIES).toContain('anova');
    expect(SELECTABLE_FAMILIES).toContain('mixed');
  });

  it('AdvancedConfig has no separate alpha (single-sourced on FamilyConfig.alpha)', () => {
    const cfg = defaultFamilyConfig('regression');
    expect('alpha' in cfg.advanced).toBe(false);
    expect(cfg.alpha).toBe(0.05);
  });
});
