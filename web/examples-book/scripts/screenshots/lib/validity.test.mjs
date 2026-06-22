import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { bootContext, seedAndReload, driveToFamily, readValidity } from './page-driver.mjs';
import { captureSnapshot } from './snapshot.mjs';
import { driveFor } from './examples.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const load = (id) => JSON.parse(readFileSync(path.join(here, `../${id}.config.json`)));

test('a valid glmm-01 config gates the run open (Find power enabled)', async () => {
  const { browser, page } = await bootContext();
  after(() => browser.close());
  const { config, outcomeKind } = load('glmm-01');
  const plan = driveFor('glmm-01');
  await seedAndReload(page, { familyKey: plan.familyKey, config, snapshot: captureSnapshot() });
  await driveToFamily(page, { entrypoint: plan.entrypoint, outcomeKind });
  assert.equal((await readValidity(page)).ok, true);
});

test('a config with a broken formula is rejected (Find power disabled)', async () => {
  const { browser, page } = await bootContext();
  after(() => browser.close());
  const { config } = load('ols-01');
  const broken = { ...config, formula: 'y ~ (' }; // unbalanced → parse error → spec null
  await seedAndReload(page, { familyKey: 'regression', config: broken, snapshot: captureSnapshot() });
  await driveToFamily(page, { entrypoint: 'regression', outcomeKind: 'continuous' });
  assert.equal((await readValidity(page)).ok, false);
});
