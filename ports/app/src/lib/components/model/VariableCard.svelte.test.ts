// Tests for VariableCard.svelte — locked prop disables type/levels/share controls
// but leaves effect inputs editable; unlocked restores controls.
import { render } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import { fireEvent } from '@testing-library/svelte';
import VariableCard from './VariableCard.svelte';
import type { VariableRow } from '$lib/domain/family';
import type { VariableGroup } from '$lib/domain/effect-names';

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function factorRow(): VariableRow {
    return {
        name: 'cyl',
        kind: 'factor',
        levels: ['4', '6', '8'],
        nLevels: 3,
        levelProportions: [0.4, 0.35, 0.25],
        referenceLevel: '4',
    };
}

function binaryRow(): VariableRow {
    return { name: 'treated', kind: 'binary', binaryProportion: 0.6 };
}

function continuousRow(): VariableRow {
    return { name: 'age', kind: 'continuous' };
}

function factorGroup(v: VariableRow): VariableGroup {
    return {
        name: v.name,
        kind: 'factor',
        rows: [
            { name: `${v.name}[4]`, isReference: true },
            { name: `${v.name}[6]`, isReference: false },
            { name: `${v.name}[8]`, isReference: false },
        ],
    };
}

function binaryGroup(v: VariableRow): VariableGroup {
    return {
        name: v.name,
        kind: 'binary',
        rows: [{ name: v.name, isReference: false }],
    };
}

function continuousGroup(v: VariableRow): VariableGroup {
    return {
        name: v.name,
        kind: 'continuous',
        rows: [{ name: v.name, isReference: false }],
    };
}

// ---------------------------------------------------------------------------
// Factor — locked
// ---------------------------------------------------------------------------

describe('VariableCard factor locked', () => {
    it('shows "from data" badge when locked', () => {
        const v = factorRow();
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: factorGroup(v),
                effects: [],
                variables: [v],
                locked: true,
            },
        });
        expect(container.textContent).toMatch(/from data/i);
    });

    it('does NOT render the type Select trigger when locked', () => {
        const v = factorRow();
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: factorGroup(v),
                effects: [],
                variables: [v],
                locked: true,
            },
        });
        // The Select trigger uses an aria role of combobox or button — but easiest
        // to verify by checking it's absent. The locked path renders a plain <span>.
        // Look for a data-slot="select-trigger" which the SelectTrigger puts on the element.
        expect(container.querySelector('[data-slot="select-trigger"]')).toBeNull();
    });

    it('disables the nLevels NumberInput when locked (factor)', () => {
        const v = factorRow();
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: factorGroup(v),
                effects: [],
                variables: [v],
                locked: true,
            },
        });
        // NumberInput renders an <input type="number">; the nLevels input is the first one.
        const inputs = container.querySelectorAll('input[type="number"]');
        expect(inputs.length).toBeGreaterThan(0);
        // Every number input rendered for locked type/share controls must be disabled.
        inputs.forEach((inp) => {
            expect((inp as HTMLInputElement).disabled).toBe(true);
        });
    });

    it('shows the type kind as static text when locked', () => {
        const v = factorRow();
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: factorGroup(v),
                effects: [],
                variables: [v],
                locked: true,
            },
        });
        // The locked path renders variable.kind as plain text in a <span>.
        expect(container.textContent).toContain('factor');
    });
});

// ---------------------------------------------------------------------------
// Binary — locked
// ---------------------------------------------------------------------------

describe('VariableCard binary locked', () => {
    it('disables binary share NumberInput when locked', () => {
        const v = binaryRow();
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: binaryGroup(v),
                effects: [],
                variables: [v],
                locked: true,
            },
        });
        const inputs = container.querySelectorAll('input[type="number"]');
        expect(inputs.length).toBeGreaterThan(0);
        inputs.forEach((inp) => {
            expect((inp as HTMLInputElement).disabled).toBe(true);
        });
    });
});

// ---------------------------------------------------------------------------
// Continuous — locked (no extra controls; just ensures no crash)
// ---------------------------------------------------------------------------

describe('VariableCard continuous locked', () => {
    it('renders without error when locked and continuous', () => {
        const v = continuousRow();
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: continuousGroup(v),
                effects: [],
                variables: [v],
                locked: true,
            },
        });
        expect(container.textContent).toMatch(/from data/i);
        expect(container.textContent).toContain('continuous');
    });

    it('does NOT render the type Select when locked', () => {
        const v = continuousRow();
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: continuousGroup(v),
                effects: [],
                variables: [v],
                locked: true,
            },
        });
        expect(container.querySelector('[data-slot="select-trigger"]')).toBeNull();
    });
});

// ---------------------------------------------------------------------------
// Unlocked — controls are active
// ---------------------------------------------------------------------------

describe('VariableCard factor unlocked', () => {
    it('renders the type Select trigger when not locked', () => {
        const v = factorRow();
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: factorGroup(v),
                effects: [],
                variables: [v],
                locked: false,
            },
        });
        expect(container.querySelector('[data-slot="select-trigger"]')).not.toBeNull();
    });

    it('does NOT show "from data" badge when unlocked', () => {
        const v = factorRow();
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: factorGroup(v),
                effects: [],
                variables: [v],
                locked: false,
            },
        });
        expect(container.textContent).not.toMatch(/from data/i);
    });

    it('nLevels input is NOT disabled when unlocked', () => {
        const v = factorRow();
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: factorGroup(v),
                effects: [],
                variables: [v],
                locked: false,
            },
        });
        const inputs = container.querySelectorAll('input[type="number"]');
        expect(inputs.length).toBeGreaterThan(0);
        // At least one input should be enabled when unlocked.
        const enabled = Array.from(inputs).some((inp) => !(inp as HTMLInputElement).disabled);
        expect(enabled).toBe(true);
    });
});

// ---------------------------------------------------------------------------
// Locked with effects present — effects stay editable (test #2 / plan I1)
// ---------------------------------------------------------------------------
// These tests exercise the gap: previous locked tests all used effects: [],
// so the effect inputs were never rendered. Here we verify that locking
// type/level/share controls does NOT disable the effect size inputs.

describe('VariableCard factor locked — effects stay editable', () => {
    it('factor effect input (non-reference level) is NOT disabled when locked', () => {
        const v = factorRow(); // kind:'factor', levels:['4','6','8']
        const g = factorGroup(v);
        // Provide one non-reference effect (cyl[6]).
        const effects = [{ name: 'cyl[6]', value: 0.3 }, { name: 'cyl[8]', value: 0.5 }];
        const { container } = render(VariableCard, {
            props: { variable: v, group: g, effects, variables: [v], locked: true },
        });

        // Type select must be absent (locked).
        expect(container.querySelector('[data-slot="select-trigger"]')).toBeNull();
        // nLevels NumberInput (first in the header) must be disabled.
        const allInputs = Array.from(container.querySelectorAll('input[type="number"]'));
        expect(allInputs.length).toBeGreaterThan(0);
        // The effect inputs are rendered by EffectControls which never passes disabled.
        // All disabled inputs are type/level/share controls; effect inputs must not be disabled.
        const effectInputs = allInputs.filter((inp) => {
            const testid = inp.closest('[data-testid]')?.getAttribute('data-testid') ?? '';
            return testid.startsWith('effect-');
        });
        expect(effectInputs.length).toBeGreaterThan(0);
        effectInputs.forEach((inp) => {
            expect((inp as HTMLInputElement).disabled).toBe(false);
        });
        // And the type/level inputs (those without an effect- testid parent) are disabled.
        const lockedInputs = allInputs.filter((inp) => {
            const testid = inp.closest('[data-testid]')?.getAttribute('data-testid') ?? '';
            return !testid.startsWith('effect-');
        });
        expect(lockedInputs.length).toBeGreaterThan(0);
        lockedInputs.forEach((inp) => {
            expect((inp as HTMLInputElement).disabled).toBe(true);
        });
    });

    it('binary effect input is NOT disabled when locked', () => {
        const v = binaryRow(); // kind:'binary'
        const g = binaryGroup(v);
        const effects = [{ name: 'treated', value: 0.4 }];
        const { container } = render(VariableCard, {
            props: { variable: v, group: g, effects, variables: [v], locked: true },
        });

        // The binary share NumberInput is disabled (locked).
        const shareInputs = Array.from(container.querySelectorAll('input[type="number"]')).filter(
            (inp) => {
                const testid = inp.closest('[data-testid]')?.getAttribute('data-testid') ?? '';
                return !testid.startsWith('effect-');
            },
        );
        expect(shareInputs.length).toBeGreaterThan(0);
        shareInputs.forEach((inp) => {
            expect((inp as HTMLInputElement).disabled).toBe(true);
        });

        // The effect NumberInput is NOT disabled.
        const effectInput = container.querySelector('[data-testid="effect-treated"] input[type="number"]');
        expect(effectInput).not.toBeNull();
        expect((effectInput as HTMLInputElement).disabled).toBe(false);
    });
});

// ---------------------------------------------------------------------------
// Unlock-on-clear — re-enabling controls when locked becomes false (test #3)
// ---------------------------------------------------------------------------
// When the upload is cleared, lockedNames empties and the row's `locked` prop
// becomes false. This tests that behaviour by toggling the locked prop:
// locked=true → controls disabled; locked=false → Select restored, inputs enabled.

describe('VariableCard — unlock restores controls', () => {
    it('type Select reappears and nLevels input enabled after locked transitions to false', async () => {
        const v = factorRow();
        const g = factorGroup(v);
        const { container, rerender } = render(VariableCard, {
            props: {
                variable: v,
                group: g,
                effects: [{ name: 'cyl[6]', value: 0.25 }, { name: 'cyl[8]', value: 0.25 }],
                variables: [v],
                locked: true,
            },
        });

        // Sanity: starts locked.
        expect(container.querySelector('[data-slot="select-trigger"]')).toBeNull();
        const inputsBefore = Array.from(container.querySelectorAll('input[type="number"]')).filter(
            (inp) => {
                const testid = inp.closest('[data-testid]')?.getAttribute('data-testid') ?? '';
                return !testid.startsWith('effect-');
            },
        );
        inputsBefore.forEach((inp) => {
            expect((inp as HTMLInputElement).disabled).toBe(true);
        });

        // Clear upload: locked becomes false.
        await rerender({ variable: v, group: g, effects: [{ name: 'cyl[6]', value: 0.25 }, { name: 'cyl[8]', value: 0.25 }], variables: [v], locked: false });

        // Type Select trigger must now be present.
        expect(container.querySelector('[data-slot="select-trigger"]')).not.toBeNull();
        // "from data" badge gone.
        expect(container.textContent).not.toMatch(/from data/i);
        // nLevels / share inputs are re-enabled.
        const inputsAfter = Array.from(container.querySelectorAll('input[type="number"]')).filter(
            (inp) => {
                const testid = inp.closest('[data-testid]')?.getAttribute('data-testid') ?? '';
                return !testid.startsWith('effect-');
            },
        );
        expect(inputsAfter.length).toBeGreaterThan(0);
        inputsAfter.forEach((inp) => {
            expect((inp as HTMLInputElement).disabled).toBe(false);
        });
    });
});

// ---------------------------------------------------------------------------
// ANOVA-mode props (Task 7)
// ---------------------------------------------------------------------------

describe('VariableCard nameEditable', () => {
    it('renders an editable name input when nameEditable=true', () => {
        const v = factorRow();
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: factorGroup(v),
                effects: [],
                variables: [v],
                nameEditable: true,
            },
        });
        const input = container.querySelector('input[aria-label="Variable name"]') as HTMLInputElement | null;
        expect(input).not.toBeNull();
        expect(input!.value).toBe('cyl');
    });

    it('renders static name span (no editable input) by default', () => {
        const v = factorRow();
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: factorGroup(v),
                effects: [],
                variables: [v],
            },
        });
        expect(container.querySelector('input[aria-label="Variable name"]')).toBeNull();
        expect(container.textContent).toContain('cyl');
    });
});

describe('VariableCard onRemove', () => {
    it('shows a remove button that calls onRemove when clicked', async () => {
        const v = factorRow();
        const onRemove = vi.fn();
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: factorGroup(v),
                effects: [],
                variables: [v],
                onRemove,
            },
        });
        const btn = container.querySelector('[aria-label="Remove"]') as HTMLButtonElement | null;
        expect(btn).not.toBeNull();
        await fireEvent.click(btn!);
        expect(onRemove).toHaveBeenCalledOnce();
    });

    it('does NOT show a remove button when onRemove is not provided', () => {
        const v = factorRow();
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: factorGroup(v),
                effects: [],
                variables: [v],
            },
        });
        expect(container.querySelector('[aria-label="Remove"]')).toBeNull();
    });
});

describe('VariableCard minLevels', () => {
    it('allows 2-level factors when minLevels=2 (decrement disabled at 2, not 3)', () => {
        const v: VariableRow = {
            name: 'grp',
            kind: 'factor',
            levels: ['a', 'b'],
            nLevels: 2,
            levelProportions: [0.5, 0.5],
            referenceLevel: 'a',
        };
        const group: VariableGroup = {
            name: 'grp',
            kind: 'factor',
            rows: [
                { name: 'grp[a]', isReference: true },
                { name: 'grp[b]', isReference: false },
            ],
        };
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group,
                effects: [],
                variables: [v],
                minLevels: 2,
            },
        });
        // The nLevels NumberInput should show value 2.
        const numberInput = container.querySelector('input[type="number"]') as HTMLInputElement | null;
        expect(numberInput).not.toBeNull();
        expect(Number(numberInput!.value)).toBe(2);
        // Decrement button should be disabled (at the floor of 2).
        const decrementBtn = container.querySelector('button[aria-label="Decrement"]') as HTMLButtonElement | null;
        expect(decrementBtn).not.toBeNull();
        expect(decrementBtn!.disabled).toBe(true);
    });

    it('default minLevels=3 keeps decrement disabled at 3 (existing behavior)', () => {
        const v = factorRow(); // nLevels: 3
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: factorGroup(v),
                effects: [],
                variables: [v],
            },
        });
        const decrementBtn = container.querySelector('button[aria-label="Decrement"]') as HTMLButtonElement | null;
        expect(decrementBtn).not.toBeNull();
        expect(decrementBtn!.disabled).toBe(true);
    });
});

describe('VariableCard fixedKind', () => {
    it('renders a static kind badge when fixedKind=true (no Select trigger)', () => {
        const v = factorRow();
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: factorGroup(v),
                effects: [],
                variables: [v],
                fixedKind: true,
            },
        });
        expect(container.querySelector('[data-slot="select-trigger"]')).toBeNull();
        // Badge text should contain the kind.
        expect(container.textContent).toContain('factor');
    });

    it('renders the Select trigger normally when fixedKind=false (default)', () => {
        const v = factorRow();
        const { container } = render(VariableCard, {
            props: {
                variable: v,
                group: factorGroup(v),
                effects: [],
                variables: [v],
                fixedKind: false,
            },
        });
        expect(container.querySelector('[data-slot="select-trigger"]')).not.toBeNull();
    });
});
