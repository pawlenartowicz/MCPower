import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import { FAMILY_LABEL } from '$lib/domain/family';
import FamilyRibbon from './FamilyRibbon.svelte';

describe('FamilyRibbon', () => {
  it('renders three entrypoints, all selectable', () => {
    render(FamilyRibbon);

    // All three families are selectable.
    for (const label of [FAMILY_LABEL.regression, FAMILY_LABEL.anova, FAMILY_LABEL.mixed]) {
      expect(screen.getByRole('radio', { name: label })).toBeEnabled();
    }
  });
});
