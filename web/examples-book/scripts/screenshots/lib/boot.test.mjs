import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { bootContext, BASE } from './page-driver.mjs';

test('app boots at /online/ and ConfigPanel mounts', async () => {
  const { browser, page } = await bootContext();
  after(() => browser.close());
  assert.equal(page.url(), BASE);
  await page.locator('div.flex.flex-col.gap-3\\.5.p-4').waitFor({ state: 'visible', timeout: 15000 });
});
