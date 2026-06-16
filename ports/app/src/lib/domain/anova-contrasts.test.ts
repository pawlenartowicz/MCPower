import { describe, expect, it } from 'vitest';
import { defaultFamilyConfig } from './family';
import { regenAutoContrasts } from './anova-contrasts';

function cfgWith(factors: Array<{ name: string; nLevels: number }>) {
  return {
    ...defaultFamilyConfig('anova'),
    variables: factors.map((f) => ({ name: f.name, kind: 'factor' as const, role: 'factor' as const, nLevels: f.nLevels })),
  };
}

describe('regenAutoContrasts', () => {
  it('registers every pairwise level contrast of primary factors', () => {
    const cfg = cfgWith([{ name: 'F1', nLevels: 3 }]);
    regenAutoContrasts(cfg, 'F1:3');
    expect(cfg.contrasts.map((c) => [c.positiveName, c.negativeName])).toEqual([
      ['F1[1]', 'F1[2]'], ['F1[1]', 'F1[3]'], ['F1[2]', 'F1[3]'],
    ]);
  });

  it('preserves enable/disable toggles on regeneration', () => {
    const cfg = cfgWith([{ name: 'F1', nLevels: 3 }]);
    regenAutoContrasts(cfg, 'F1:3');
    cfg.contrasts[0]!.enabled = false;
    regenAutoContrasts(cfg, 'F1:3-again');
    expect(cfg.contrasts[0]!.enabled).toBe(false);
  });

  it('drops contrasts whose level no longer exists, keeps manual valid ones', () => {
    const cfg = cfgWith([{ name: 'F1', nLevels: 3 }]);
    regenAutoContrasts(cfg, 'F1:3');
    cfg.contrasts.push({ positiveName: 'F1[1]', negativeName: 'F1[2]', enabled: true });
    (cfg.variables[0] as any).nLevels = 2;
    regenAutoContrasts(cfg, 'F1:2');
    const pairs = cfg.contrasts.map((c) => `${c.positiveName}-${c.negativeName}`);
    expect(pairs).not.toContain('F1[1]-F1[3]');
    expect(pairs).not.toContain('F1[2]-F1[3]');
    expect(pairs).toContain('F1[1]-F1[2]');
  });
});
