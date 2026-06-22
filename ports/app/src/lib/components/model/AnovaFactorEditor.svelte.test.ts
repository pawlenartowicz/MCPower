// Tests for AnovaFactorEditor.svelte — the auto-contrast regen $effect must
// re-fire on level *renames*, not just factor/level-count changes. A stale
// signature leaves contrasts pointing at dead effect names (e.g. 'F1[2]' after
// the level was renamed), which the engine rejects at run time with
// "contrast pair references unknown effect name".
import { describe, expect, it, beforeEach } from 'vitest';
import { render } from '@testing-library/svelte';
import { tick } from 'svelte';
import AnovaFactorEditor from './AnovaFactorEditor.svelte';
import { familyStore } from '$lib/stores/family.svelte';

async function flushEffects() {
    // Two tick rounds: first flushes $derived; second flushes $effect writes.
    await tick();
    await tick();
}

function contrastPairs() {
    return familyStore.byFamily.anova.contrasts.map(
        (c) => `${c.positiveName} ${c.negativeName}`,
    );
}

describe('AnovaFactorEditor — auto-contrast regen', () => {
    beforeEach(() => {
        familyStore.resetAll();
        familyStore.active = 'anova';
        familyStore.byFamily.anova.variables = [
            {
                name: 'F1',
                kind: 'factor',
                role: 'factor',
                nLevels: 2,
                levelProportions: [0.5, 0.5],
            },
        ];
    });

    it('seeds pairwise contrasts on mount', async () => {
        render(AnovaFactorEditor);
        await flushEffects();
        expect(contrastPairs()).toEqual(['F1[1] F1[2]']);
    });

    it('rebuilds contrasts when a level is renamed', async () => {
        render(AnovaFactorEditor);
        await flushEffects();

        // Rename level 2 → 'dddd', as the Advanced dialog's label input does.
        familyStore.byFamily.anova.variables[0]!.levels = ['1', 'dddd'];
        await flushEffects();

        expect(contrastPairs()).toEqual(['F1[1] F1[dddd]']);
    });
});
