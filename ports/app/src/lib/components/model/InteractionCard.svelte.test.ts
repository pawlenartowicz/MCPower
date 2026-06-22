import { render } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import InteractionCard from './InteractionCard.svelte';
import type { VariableRow } from '$lib/domain/family';

const variables: VariableRow[] = [
  { name: 'x1', kind: 'continuous' },
  { name: 'group', kind: 'factor', nLevels: 3 },
];

describe('InteractionCard', () => {
  it('expands a factor interaction to per-level rows and shows the hint', () => {
    const group = {
      term: 'x1:group',
      isFactorInteraction: true,
      rows: [
        { name: 'x1:group[2]', isReference: false },
        { name: 'x1:group[3]', isReference: false },
      ],
    };
    const effects = [
      { name: 'x1:group[2]', value: 0 },
      { name: 'x1:group[3]', value: 0 },
    ];
    const { container } = render(InteractionCard, { props: { group, effects, variables } });
    expect(container.querySelector('[data-testid="effect-x1:group[2]"]')).not.toBeNull();
    expect(container.querySelector('[data-testid="effect-x1:group[3]"]')).not.toBeNull();
    expect(container.textContent).toMatch(/how much x1's effect changes at each level of group/);
  });

  it('keeps a continuous interaction a single inline row with no hint', () => {
    const group = {
      term: 'x1:x2',
      isFactorInteraction: false,
      rows: [{ name: 'x1:x2', isReference: false }],
    };
    const effects = [{ name: 'x1:x2', value: 0 }];
    const conts: VariableRow[] = [
      { name: 'x1', kind: 'continuous' },
      { name: 'x2', kind: 'continuous' },
    ];
    const { container } = render(InteractionCard, {
      props: { group, effects, variables: conts },
    });
    expect(container.querySelector('[data-testid="effect-x1:x2"]')).not.toBeNull();
    expect(container.textContent).not.toMatch(/how much/);
  });
});
