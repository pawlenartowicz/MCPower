import { render } from '@testing-library/svelte';
import { describe, it, expect, afterEach } from 'vitest';
import EffectControls from './EffectControls.svelte';
import { familyStore } from '$lib/stores/family.svelte';

describe('EffectControls', () => {
  // The active family defaults to 'regression', whose outcome toggle gates the
  // OR readout; reset it so the logit case never leaks into other suites.
  afterEach(() => {
    familyStore.regressionOutcome = 'continuous';
  });

  it('sign-flip and preset buttons use bg-secondary (not bg-card) for contrast', () => {
    const effect = { name: 'x', value: 0.3 };
    const variables = [{ kind: 'continuous' as const, name: 'x' }];
    const { container } = render(EffectControls, { props: { effect, variables } });

    const signFlipBtn = container.querySelector('button[aria-label="Flip sign"]')!;
    expect(signFlipBtn).not.toBeNull();
    expect(signFlipBtn.className).toContain('bg-secondary');
    expect(signFlipBtn.className).not.toContain('bg-card');

    const presetBtns = container.querySelectorAll('button[title*="Cohen"]');
    expect(presetBtns.length).toBeGreaterThan(0);
    const first = presetBtns[0]!;
    expect(first.className).toContain('bg-secondary');
    expect(first.className).not.toContain('bg-card');
  });

  it('continuous outcome shows no OR readout', () => {
    familyStore.regressionOutcome = 'continuous';
    const { queryByTestId } = render(EffectControls, {
      props: { effect: { name: 'x', value: 0.916 }, variables: [{ kind: 'continuous', name: 'x' }] },
    });
    expect(queryByTestId('effect-or-x')).toBeNull();
  });

  it('logit outcome renders OR = exp(β) and a beta badge', () => {
    familyStore.regressionOutcome = 'binary';
    const { getByTestId, container } = render(EffectControls, {
      props: { effect: { name: 'x', value: 0.916 }, variables: [{ kind: 'continuous', name: 'x' }] },
    });
    // exp(0.916) ≈ 2.50 — the medium odds preset.
    expect(getByTestId('effect-or-x').textContent?.trim()).toBe('OR 2.50');
    // Presets are the odds (beta) set: titles cite Chen, not Cohen, and a beta badge shows.
    expect(container.querySelector('button[title*="Chen"]')).not.toBeNull();
    expect(container.querySelector('button[title*="Cohen"]')).toBeNull();
    expect(container.textContent).toContain('beta');
  });
});
