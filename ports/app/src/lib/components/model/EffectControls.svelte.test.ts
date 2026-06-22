import { render } from '@testing-library/svelte';
import { describe, it, expect } from 'vitest';
import EffectControls from './EffectControls.svelte';

describe('EffectControls', () => {
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
});
