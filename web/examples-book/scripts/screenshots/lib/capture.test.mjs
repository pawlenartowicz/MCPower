import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync, statSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { bootContext, driveAndShoot } from './page-driver.mjs';
import { captureSnapshot } from './snapshot.mjs';
import { driveFor } from './examples.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const out = path.join(here, '../__tmp__/ols-01-setup.png');

test('driveAndShoot writes a non-trivial ConfigPanel PNG for ols-01', async () => {
  const { browser, page } = await bootContext();
  after(() => { browser.close(); if (existsSync(out)) rmSync(out); });
  const { config, outcomeKind } = JSON.parse(readFileSync(path.join(here, '../ols-01.config.json')));
  const plan = driveFor('ols-01');
  const res = await driveAndShoot(page, {
    id: 'ols-01', ...plan, outcomeKind, config, snapshot: captureSnapshot(), outPath: out,
  });
  assert.equal(res.ok, true);
  assert.ok(existsSync(out));
  assert.ok(statSync(out).size > 5000, 'PNG should be a real render, not blank');
  // The seeded effect size must survive cold-boot hydration (applyEffects fix):
  // the captured form shows years_education = 0.25, not the reconcile-zeroed 0.
  const shown = await page
    .getByTestId('effect-years_education')
    .locator('input[type="number"]')
    .inputValue();
  assert.equal(shown, '0.25', 'effect value should render the seeded 0.25, not 0');
});
