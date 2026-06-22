import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runHarness } from './capture-screens.mjs';

test('validate mode reports both fixtures green', async () => {
  const { results } = await runHarness({ mode: 'validate', ids: ['ols-01', 'glmm-01'] });
  assert.equal(results['ols-01'].ok, true);
  assert.equal(results['glmm-01'].ok, true);
});
