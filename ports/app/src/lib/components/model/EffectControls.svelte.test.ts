import { render } from '@testing-library/svelte';
import { describe, it, expect, afterEach, vi } from 'vitest';
import EffectControls from './EffectControls.svelte';
import { familyStore } from '$lib/stores/family.svelte';
import type { StandardizedValues } from '$lib/stores/standardized.svelte';

// The live standardized values, set per test; null = pending (the default).
const std = vi.hoisted(() => ({ current: null as StandardizedValues | null }));
vi.mock('$lib/stores/standardized.svelte', () => ({ standardizedStore: std }));

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

  it('logit outcome renders OR = exp(β) and the odds presets', () => {
    familyStore.regressionOutcome = 'logit';
    const { getByTestId, container } = render(EffectControls, {
      props: { effect: { name: 'x', value: 0.916 }, variables: [{ kind: 'continuous', name: 'x' }] },
    });
    // exp(0.916) ≈ 2.50 — the medium odds preset.
    expect(getByTestId('effect-or-x').textContent?.trim()).toBe('OR 2.50');
    // Presets are the odds (beta) set: titles cite Chen, not Cohen.
    expect(container.querySelector('button[title*="Chen"]')).not.toBeNull();
    expect(container.querySelector('button[title*="Cohen"]')).toBeNull();
  });

  it('probit outcome shows no OR readout and uses Cohen presets (not odds)', () => {
    familyStore.regressionOutcome = 'probit';
    const { queryByTestId, container } = render(EffectControls, {
      props: { effect: { name: 'x', value: 0.5 }, variables: [{ kind: 'continuous', name: 'x' }] },
    });
    // A probit β is Cohen's-d scale, not log-odds — no ratio readout.
    expect(queryByTestId('effect-or-x')).toBeNull();
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
  });

  describe('standardized readout', () => {
    afterEach(() => {
      std.current = null;
    });
    const values = (rows: StandardizedValues['rows']): StandardizedValues => ({
      rows,
      r2: 0.2,
      clustered: false,
    });

    it('shows a dash while the live value is pending', () => {
      const { getByTestId } = render(EffectControls, {
        props: { effect: { name: 'x', value: 0.3 }, variables: [{ kind: 'continuous', name: 'x' }] },
      });
      expect(getByTestId('effect-std-x').textContent?.trim()).toBe('Std. β —');
    });

    it('shows the standardized β of a continuous effect', () => {
      std.current = values([{ name: 'x', input: 0.3, std_beta: 0.2673, cohen_d: null }]);
      const { getByTestId } = render(EffectControls, {
        props: { effect: { name: 'x', value: 0.3 }, variables: [{ kind: 'continuous', name: 'x' }] },
      });
      expect(getByTestId('effect-std-x').textContent?.trim()).toBe('Std. β 0.267');
    });

    it("shows Cohen's d for a binary predictor and a factor level", () => {
      std.current = values([
        { name: 't', input: 0.5, std_beta: 0.22, cohen_d: 0.457 },
        { name: 'g[B]', input: 0.2, std_beta: 0.08, cohen_d: 0.18 },
      ]);
      const variables = [
        { kind: 'binary' as const, name: 't' },
        { kind: 'factor' as const, name: 'g' },
      ];
      const t = render(EffectControls, { props: { effect: { name: 't', value: 0.5 }, variables } });
      expect(t.getByTestId('effect-std-t').textContent?.trim()).toBe('d 0.457');
      const g = render(EffectControls, { props: { effect: { name: 'g[B]', value: 0.2 }, variables } });
      expect(g.getByTestId('effect-std-g[B]').textContent?.trim()).toBe('d 0.180');
    });
  });
});

