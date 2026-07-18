import { render } from '@testing-library/svelte';
import { describe, it, expect, afterEach } from 'vitest';
import EffectControls from './EffectControls.svelte';
import { familyStore } from '$lib/stores/family.svelte';

describe('EffectControls', () => {
  // The active family defaults to 'regression', whose outcome toggle gates the
  // OR readout; reset it so the logit case never leaks into other suites.
  afterEach(() => {
    familyStore.regressionOutcome = 'linear';
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
    familyStore.regressionOutcome = 'linear';
    const { queryByTestId } = render(EffectControls, {
      props: { effect: { name: 'x', value: 0.916 }, variables: [{ kind: 'continuous', name: 'x' }] },
    });
    expect(queryByTestId('effect-or-x')).toBeNull();
  });

  it('logit outcome renders OR = exp(β) and a beta badge', () => {
    familyStore.regressionOutcome = 'logit';
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

  it('probit outcome shows no OR readout and uses Cohen presets (not odds)', () => {
    familyStore.regressionOutcome = 'probit';
    const { queryByTestId, container } = render(EffectControls, {
      props: { effect: { name: 'x', value: 0.5 }, variables: [{ kind: 'continuous', name: 'x' }] },
    });
    // A probit β is Cohen's-d scale, not log-odds — no ratio readout, no beta badge.
    expect(queryByTestId('effect-or-x')).toBeNull();
    expect(container.textContent).not.toContain('beta');
    expect(container.querySelector('button[title*="Cohen"]')).not.toBeNull();
    expect(container.querySelector('button[title*="Chen"]')).toBeNull();
  });

  it('poisson outcome renders RR = exp(β) and reuses the odds anchor values', () => {
    familyStore.regressionOutcome = 'poisson';
    const { getByTestId, container } = render(EffectControls, {
      props: { effect: { name: 'x', value: 0.916 }, variables: [{ kind: 'continuous', name: 'x' }] },
    });
    // exp(0.916) ≈ 2.50 — the medium rate-ratio anchor (reused from the odds triple).
    expect(getByTestId('effect-or-x').textContent?.trim()).toBe('RR 2.50');
    expect(container.querySelector('button[title*="Chen"]')).not.toBeNull();
    expect(container.querySelector('button[title*="Cohen"]')).toBeNull();
    expect(container.textContent).toContain('beta');
  });
});
