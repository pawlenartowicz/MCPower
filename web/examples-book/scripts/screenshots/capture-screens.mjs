import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { discoverIds, driveFor } from './lib/examples.mjs';
import { captureSnapshot } from './lib/snapshot.mjs';
import { bootContext, driveAndShoot } from './lib/page-driver.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CHUNKS = path.resolve(HERE, '../../chunks');
const ASSETS = path.resolve(HERE, '../../assets');

function loadConfig(id) {
  return JSON.parse(readFileSync(path.join(HERE, `${id}.config.json`)));
}

// One browser, serial over ids, for deterministic output (spec §4 step 4).
// theme ∈ {'light','dark'}: light → canonical `<id>-setup.png`; dark → `<id>-setup-dark.png`.
export async function runHarness({ mode, ids, theme = 'light' }) {
  const list = ids?.length ? ids : discoverIds(CHUNKS);
  const snapshot = captureSnapshot(theme);
  const suffix = theme === 'dark' ? '-setup-dark' : '-setup';
  const results = {};
  const { browser, page } = await bootContext({ colorScheme: theme === 'dark' ? 'dark' : 'light' });
  try {
    for (const id of list) {
      const { config, outcomeKind } = loadConfig(id);
      const plan = driveFor(id);
      const outPath = mode === 'capture' ? path.join(ASSETS, `${id}${suffix}.png`) : null;
      results[id] = await driveAndShoot(page, {
        id, familyKey: plan.familyKey, entrypoint: plan.entrypoint, outcomeKind, config, snapshot, outPath,
      });
    }
  } finally {
    await browser.close();
  }
  if (mode === 'validate') {
    writeFileSync(path.join(HERE, 'validation.json'), JSON.stringify(results, null, 2));
  }
  return { results };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const argv = process.argv.slice(2);
  const mode = argv[argv.indexOf('--mode') + 1];
  const themeArg = argv.includes('--theme') ? argv[argv.indexOf('--theme') + 1] : 'light';
  const ids = argv.reduce((a, t, i) => (argv[i - 1] === '--id' ? [...a, t] : a), []);
  if (mode !== 'validate' && mode !== 'capture') {
    console.error('usage: capture-screens.mjs --mode validate|capture [--theme light|dark|both] [--id <id> ...]');
    process.exit(2);
  }
  const themes = themeArg === 'both' ? ['light', 'dark'] : [themeArg === 'dark' ? 'dark' : 'light'];
  for (const theme of themes) {
    const { results } = await runHarness({ mode, ids, theme });
    const failed = Object.entries(results).filter(([, r]) => !r.ok).map(([id]) => id);
    console.log(`${mode} (${theme}): ${Object.keys(results).length} processed, ${failed.length} failed`);
    if (failed.length) console.log('failed:', failed.join(', '));
  }
}
