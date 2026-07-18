import { describe, it, expect } from 'vitest';
import { clampAgqNodes, AGQ_DEFAULT_NODES, AGQ_MIN_NODES, AGQ_MAX_NODES } from './estimation-options';

describe('clampAgqNodes', () => {
  it('rounds up to the nearest odd value', () => {
    expect(clampAgqNodes(4)).toBe(5);
    expect(clampAgqNodes(7)).toBe(7);
  });

  it('clamps below MIN and above MAX', () => {
    expect(clampAgqNodes(1)).toBe(AGQ_MIN_NODES);
    expect(clampAgqNodes(100)).toBe(AGQ_MAX_NODES);
  });

  it('falls back to the default instead of propagating NaN (an emptied NumberInput)', () => {
    expect(clampAgqNodes(NaN)).toBe(AGQ_DEFAULT_NODES);
  });

  it('falls back to the default for non-finite input', () => {
    expect(clampAgqNodes(Infinity)).toBe(AGQ_DEFAULT_NODES);
    expect(clampAgqNodes(-Infinity)).toBe(AGQ_DEFAULT_NODES);
  });
});
