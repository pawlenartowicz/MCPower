import { render } from '@testing-library/svelte';
import { describe, it, expect, beforeEach } from 'vitest';
import GetStartedChecklist from './GetStartedChecklist.svelte';
import { runStore } from '$lib/stores/run.svelte';
import { familyStore } from '$lib/stores/family.svelte';

describe('GetStartedChecklist', () => {
  beforeEach(() => {
    runStore.clearTabs();
    familyStore.resetActive();
  });

  it('shows exactly three readiness steps (no run-the-button step)', () => {
    const { container } = render(GetStartedChecklist);
    expect(container.querySelectorAll('li').length).toBe(3);
    expect(container.textContent).not.toMatch(/Find power or Find sample/);
  });

  it('marks the model step done for ANOVA when a factor is configured', () => {
    familyStore.active = 'anova';
    familyStore.byFamily.anova.variables = [{ name: 'F1', kind: 'factor', role: 'factor' }];
    const { container } = render(GetStartedChecklist);
    const modelItem = container.querySelectorAll('li')[1];
    expect(modelItem.querySelector('.text-green-600')).not.toBeNull();
  });

  it('labels the ANOVA model step as adding a factor, not entering a formula', () => {
    familyStore.active = 'anova';
    const { container } = render(GetStartedChecklist);
    const modelItem = container.querySelectorAll('li')[1];
    expect(modelItem.textContent).toMatch(/factor/i);
    expect(modelItem.textContent).not.toMatch(/formula/i);
  });
});
