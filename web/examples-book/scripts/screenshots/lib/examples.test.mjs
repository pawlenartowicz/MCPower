import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { discoverIds, driveFor } from './examples.mjs';

const CHUNKS = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../../chunks',
);

test('discoverIds finds every example id from chunks/*.py', () => {
  const ids = discoverIds(CHUNKS);
  assert.ok(ids.includes('ols-01'));
  assert.ok(ids.includes('glmm-01'));
  assert.ok(ids.length >= 40, `expected >=40 ids, got ${ids.length}`);
  assert.deepEqual(ids, [...ids].sort()); // stable order
});

test('driveFor maps each book family to its drive plan', () => {
  assert.deepEqual(driveFor('ols-03'),  { entrypoint: 'regression', outcomeKind: 'continuous', familyKey: 'regression' });
  assert.deepEqual(driveFor('glm-02'),  { entrypoint: 'regression', outcomeKind: 'binary',     familyKey: 'regression' });
  assert.deepEqual(driveFor('anova-01'),{ entrypoint: 'anova',      outcomeKind: 'continuous', familyKey: 'anova' });
  assert.deepEqual(driveFor('lmm-04'),  { entrypoint: 'mixed',      outcomeKind: 'continuous', familyKey: 'mixed' });
  assert.deepEqual(driveFor('glmm-05'), { entrypoint: 'mixed',      outcomeKind: 'binary',     familyKey: 'mixed' });
});

test('factorial ANOVA ids are driven through the regression entrypoint', () => {
  for (const id of ['anova-04', 'anova-05', 'anova-06']) {
    assert.deepEqual(driveFor(id), { entrypoint: 'regression', outcomeKind: 'continuous', familyKey: 'regression' });
  }
});
