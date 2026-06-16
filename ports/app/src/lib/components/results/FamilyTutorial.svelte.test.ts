import { render, fireEvent, screen } from '@testing-library/svelte';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import FamilyTutorial from './FamilyTutorial.svelte';
import { familyStore } from '$lib/stores/family.svelte';

describe('FamilyTutorial', () => {
  beforeEach(() => {
    familyStore.resetAll(); // active back to 'regression'
  });

  it('renders the regression tutorial by default', () => {
    render(FamilyTutorial);
    expect(screen.getByTestId('family-tutorial').textContent).toMatch(/Regression power/i);
  });

  it('renders the ANOVA tutorial when ANOVA is active', () => {
    familyStore.active = 'anova';
    render(FamilyTutorial);
    expect(screen.getByTestId('family-tutorial').textContent).toMatch(/ANOVA power/i);
  });

  it('opens documentation links in the system browser', async () => {
    const openSpy = vi.spyOn(window, 'open').mockReturnValue(null);
    render(FamilyTutorial);
    const link = screen.getByTestId('family-tutorial').querySelector('a');
    expect(link).not.toBeNull();
    await fireEvent.click(link!);
    expect(openSpy).toHaveBeenCalledWith(
      expect.stringContaining('https://docs.mcpower.app/'),
      '_blank',
      expect.any(String),
    );
  });
});
