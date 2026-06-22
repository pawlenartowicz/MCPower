import { test } from 'node:test';
import assert from 'node:assert/strict';
import { reconcileEmbed } from './embed.mjs';

const LIGHT = '![[assets/ols-01-setup.png|600|theme-light]]';
const DARK = '![[assets/ols-01-setup-dark.png|600|theme-dark]]';
const PAIR = `${LIGHT}\n${DARK}`;

test('rewrites a legacy -setup-1 placeholder to the theme-swapping pair', () => {
  const md = '# Title\n\n![[examples-book/ols/ols-01-setup-1.png|600]]\n\nbody';
  const { text, changed } = reconcileEmbed(md, 'ols-01');
  assert.equal(changed, true);
  assert.ok(text.includes(LIGHT));
  assert.ok(text.includes(DARK));
  assert.ok(!text.includes('setup-1'));
});

test('collapses an already-canonical pair idempotently', () => {
  const md = `# Title\n\n${PAIR}\n`;
  const { text, changed } = reconcileEmbed(md, 'ols-01');
  assert.equal(changed, false);
  assert.equal(text.match(/theme-light/g).length, 1);
  assert.equal(text.match(/theme-dark/g).length, 1);
});

test('upgrades a bare light-only embed to the pair', () => {
  const md = '# Title\n\n![[assets/ols-01-setup.png|600]]\n';
  const { text, changed } = reconcileEmbed(md, 'ols-01');
  assert.equal(changed, true);
  assert.ok(text.includes(LIGHT));
  assert.ok(text.includes(DARK));
});

test('inserts the pair under H1 when no placeholder exists', () => {
  const md = '# Title\n\nbody only\n';
  const { text, changed } = reconcileEmbed(md, 'ols-01');
  assert.equal(changed, true);
  assert.ok(text.includes(PAIR));
  assert.ok(text.indexOf('# Title') < text.indexOf(PAIR));
  assert.ok(text.indexOf(PAIR) < text.indexOf('body only'));
});
