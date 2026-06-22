import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import path from 'node:path';

export const BASE = 'http://localhost:4173/online/';
const VIEWPORT = { width: 1440, height: 900 }; // wide → two-pane layout active

const CONFIG_PANEL = 'div.flex.flex-col.gap-3\\.5.p-4';
const FAMILY_LABEL = { regression: 'Regression', anova: 'ANOVA', mixed: 'Mixed effects' };

export async function bootContext({ colorScheme = 'light' } = {}) {
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: VIEWPORT,
    deviceScaleFactor: 2, // crisp screenshots
    colorScheme, // match the seeded snapshot.theme so prefers-color-scheme CSS + chart theme agree
  });
  const page = await context.newPage();
  await page.goto(BASE, { waitUntil: 'load' });
  return { browser, page };
}

// Runs in-page. Writes the two records the app reads on boot:
//   kv['per-family/<familyKey>.json'] = { config }
//   kv['settings.json']               = { snapshot }
// DB 'mcpower' v1, out-of-line-keyed store 'kv' (browser-store.ts:6-12).
function seedInPage({ familyKey, config, snapshot }) {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('mcpower', 1);
    req.onupgradeneeded = () => req.result.createObjectStore('kv');
    req.onerror = () => reject(req.error);
    req.onsuccess = () => {
      const db = req.result;
      const tx = db.transaction('kv', 'readwrite');
      const kv = tx.objectStore('kv');
      kv.put({ config }, `per-family/${familyKey}.json`);
      kv.put({ snapshot }, 'settings.json');
      tx.oncomplete = () => { db.close(); resolve(); };
      tx.onerror = () => reject(tx.error);
    };
  });
}

export async function seedAndReload(page, payload) {
  // seedInPage's own onupgradeneeded creates the store if boot hasn't yet
  // (browser-store.ts:9-16 — lazy on first open).
  await page.evaluate(seedInPage, payload);
  await page.reload({ waitUntil: 'load' }); // boot now hydrates from seeded IDB
  await page.locator(CONFIG_PANEL).waitFor({ state: 'visible' });
}

export async function driveToFamily(page, { entrypoint, outcomeKind }) {
  // Click the family radio: ANOVA/mixed hydrate on demand; clicking regression
  // (the default) is harmless and makes the step uniform.
  await page.getByRole('radio', { name: FAMILY_LABEL[entrypoint], exact: true }).click();
  // Regression's binary outcome is a store-only toggle (not in FamilyConfig).
  // Mixed's binary outcome rides in the seeded config.cluster.binaryOutcome.
  if (entrypoint === 'regression' && outcomeKind === 'binary') {
    await page.getByRole('button', { name: 'Binary', exact: true }).click();
  }
  await page.locator(CONFIG_PANEL).waitFor({ state: 'visible' });
}

// The adapter parses the formula asynchronously (real WASM parser); while pending,
// spec is null and Find power is disabled — indistinguishable from a genuine parse
// failure, so the stability window below must outlast parse latency. A valid config
// read as invalid here means the parse hadn't settled (widen the window), not a bad
// fixture. Poll until the disabled state is stable.
async function waitSettled(page, timeout = 10000) {
  const btn = page.getByRole('button', { name: 'Find power', exact: true });
  await btn.waitFor({ state: 'visible', timeout });
  let prev = null, stable = 0;
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    const d = await btn.isDisabled();
    if (d === prev) { if (++stable >= 3) return d; } else { stable = 0; prev = d; }
    await page.waitForTimeout(150);
  }
  return prev;
}

export async function readValidity(page) {
  const disabled = await waitSettled(page);
  const warnings = await page.locator('[data-testid="config-warnings"]').allTextContents();
  return { ok: disabled === false, warnings };
}

// Cold-boot hydration zeroes seeded effect values: the formula parses async, and
// PredictorCards' reconcile runs against an empty predictor list before the parse
// lands (getStable has no last-good on a cold load), so it rebuilds every effect
// row at 0 — only the *values* are lost, names/kinds/clusters survive. The fix is
// to re-enter the values through the shipped effect inputs once the parse has
// settled (formula is stable, so this commit does not re-trigger the wipe). Each
// EffectControls NumberInput carries data-testid="effect-<canonical-name>"; the
// authored config.effects use those same names. Reference-level rows render no
// input (skipped). Zero-app-change: drives only existing UI controls.
async function applyEffects(page, config) {
  for (const eff of config.effects ?? []) {
    if (!Number.isFinite(eff.value)) continue;
    const input = page.getByTestId(`effect-${eff.name}`).locator('input[type="number"]');
    if ((await input.count()) === 0) continue; // reference level / no rendered input
    const want = String(eff.value);
    if ((await input.inputValue().catch(() => null)) === want) continue;
    await input.fill(want);
    await input.blur(); // fires the input's onchange → app commit()
  }
}

async function screenshotConfigPanel(page, outPath) {
  mkdirSync(path.dirname(outPath), { recursive: true });
  const panel = page.locator(CONFIG_PANEL);
  await panel.waitFor({ state: 'visible' });
  // A panel taller than the viewport forces Playwright to stitch the element by
  // scrolling, during which the app's sticky top bar bleeds into the frame and
  // covers the Upload/Model sections. Grow the viewport to fit the whole panel so
  // it captures in a single frame. Measure height scroll-independently (applyEffects
  // leaves the pane scrolled down) and reset the scroll container + window to the top
  // so the panel sits fully below the top bar before capture.
  const panelH = await panel.evaluate((el) => el.getBoundingClientRect().height);
  const fitH = Math.ceil(panelH + 160); // headroom for the top bar offset
  const grow = fitH > VIEWPORT.height;
  if (grow) await page.setViewportSize({ width: VIEWPORT.width, height: fitH });
  await page.evaluate(() => {
    window.scrollTo(0, 0);
    document.querySelectorAll('.overflow-auto').forEach((e) => (e.scrollTop = 0));
  });
  await panel.screenshot({ path: outPath }); // element crop → results pane never in frame
  if (grow) await page.setViewportSize(VIEWPORT);
}

export async function driveAndShoot(page, { familyKey, entrypoint, outcomeKind, config, snapshot, outPath }) {
  await seedAndReload(page, { familyKey, config, snapshot });
  await driveToFamily(page, { entrypoint, outcomeKind });
  const verdict = await readValidity(page);
  if (verdict.ok) {
    await applyEffects(page, config); // restore effect values zeroed by cold-boot reconcile
    if (outPath) await screenshotConfigPanel(page, outPath);
  }
  return verdict;
}
