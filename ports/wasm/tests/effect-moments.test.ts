// effectMoments: the TS wrapper over the wasm `effect_moments` export. The
// wasm module is not compiled in the test environment, so the vendor module is
// mocked; the numbers are tested in engine-core.
import { describe, it, expect, vi } from 'vitest';

vi.mock('../vendor/engine-wasm/engine_wasm.js', () => ({
  default: vi.fn().mockResolvedValue(undefined), // init()
  effect_moments: vi.fn(),
}));

import * as wasmMod from '../vendor/engine-wasm/engine_wasm.js';
import { effectMoments } from '../src/index';
import type { AppSpec, EffectMoments } from '../src/types';

describe('effectMoments', () => {
  it('passes the JSON-serialized spec and returns the parsed moments, nulls kept', async () => {
    const out: EffectMoments = {
      descriptors: [{ kind: 'continuous', predictor: 'x' }],
      moments: { noise: 1, sd: [1], cov: [1], groups: [null] },
    };
    vi.mocked(wasmMod.effect_moments).mockReturnValue(JSON.stringify(out));
    const spec = { family: 'linear', effects: [{ name: 'x', value: 0.5 }] } as unknown as AppSpec;
    await expect(effectMoments(spec)).resolves.toEqual(out);
    expect(wasmMod.effect_moments).toHaveBeenCalledWith(JSON.stringify(spec));
  });
});
