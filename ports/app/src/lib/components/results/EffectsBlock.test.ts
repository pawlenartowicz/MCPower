import { render } from '@testing-library/svelte';
import { describe, it, expect } from 'vitest';
import EffectsBlock from './EffectsBlock.svelte';
import type { AppSpec } from '$lib/domain/app-spec';
import type { StandardizedValues } from '$lib/stores/standardized.svelte';

const values = (clustered = false): StandardizedValues => ({
  rows: [
    { name: 'x1', input: 0.5, std_beta: 0.4472, cohen_d: null },
    { name: 't', input: 0.3, std_beta: null, cohen_d: null },
  ],
  r2: 0.2,
  clustered,
});

function headers(container: HTMLElement): string[] {
  return [...container.querySelectorAll('th')].map((th) => th.textContent?.trim() ?? '');
}

describe('EffectsBlock', () => {
  it('lists every effect with dashes for undefined values and the R² line', () => {
    const { container, getByTestId } = render(EffectsBlock, {
      props: { values: values(), spec: { family: 'linear' } as AppSpec },
    });
    expect(headers(container)).toEqual(['Effect', 'Input', 'Std. β', 'd']);
    const cells = [...container.querySelectorAll('tbody tr')].map((tr) =>
      [...tr.querySelectorAll('td')].map((td) => td.textContent?.trim()),
    );
    expect(cells).toEqual([
      ['x1', '0.50', '0.447', '—'],
      ['t', '0.30', '—', '—'],
    ]);
    expect(getByTestId('effects-block-r2').textContent?.trim()).toBe('R² = 0.200');
  });

  it('adds the OR on the logit link only and the RR for a count outcome', () => {
    const logit = render(EffectsBlock, { props: { values: values(), spec: { family: 'logit' } as AppSpec } });
    expect(headers(logit.container)).toContain('OR');
    const probit = render(EffectsBlock, {
      props: { values: values(), spec: { family: 'logit', link: 'probit' } as AppSpec },
    });
    expect(headers(probit.container)).not.toContain('OR');
    const poisson = render(EffectsBlock, { props: { values: values(), spec: { family: 'poisson' } as AppSpec } });
    expect(headers(poisson.container)).toContain('RR');
  });

  it('labels the fixed-effects R² for a clustered model', () => {
    const { getByTestId } = render(EffectsBlock, {
      props: { values: values(true), spec: { family: 'mixed', outcome: { kind: 'gaussian' } } as AppSpec },
    });
    expect(getByTestId('effects-block-r2').textContent?.trim()).toBe('R² (fixed effects) = 0.200');
  });
});
