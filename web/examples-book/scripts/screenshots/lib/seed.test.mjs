import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { bootContext, seedAndReload } from './page-driver.mjs';
import { captureSnapshot } from './snapshot.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const { config } = JSON.parse(readFileSync(path.join(here, '../ols-01.config.json')));

test('seeded regression config hydrates: formula text appears in the form', async () => {
  const { browser, page } = await bootContext();
  after(() => browser.close());
  await seedAndReload(page, { familyKey: 'regression', config, snapshot: captureSnapshot() });
  // ols-01 is regression/continuous → boots straight into this family.
  await page.locator('div.flex.flex-col.gap-3\\.5.p-4').waitFor({ state: 'visible' });
  const formulaShown = await page.getByText(config.formula, { exact: false }).first().isVisible();
  assert.equal(formulaShown, true, 'seeded formula should render in the hydrated form');
});
