// A value typed past a limit must not stay in the box when clamping leaves the
// state unchanged.
import { fireEvent, render } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import NumberInput from './number-input.svelte';

describe('NumberInput', () => {
  it('shows the clamped value when the typed one clamps back to the current value', async () => {
    const { container } = render(NumberInput, { value: 20, min: 20, max: 100 });
    const input = container.querySelector('input') as HTMLInputElement;
    input.value = '5';
    await fireEvent.change(input);
    expect(input.value).toBe('20');
  });
});
