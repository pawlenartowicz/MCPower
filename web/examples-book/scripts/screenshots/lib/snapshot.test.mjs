import { test } from 'node:test';
import assert from 'node:assert/strict';
import { captureSnapshot } from './snapshot.mjs';

test('captureSnapshot fixes light theme and expands every config section', () => {
  const s = captureSnapshot();
  assert.equal(s.theme, 'light');
  assert.equal(s.activePane, 'config');
  assert.equal(s.modelExpanded, true);
  assert.equal(s.runExpanded, true);
  assert.equal(s.correlationsExpanded, true);
  assert.equal(s.uploadExpanded, true);
  assert.equal(s.scenariosEnabled, false);
  assert.equal(typeof s.fontSize, 'number');
  assert.equal(typeof s.splitterFraction, 'number');
});

test('captureSnapshot defaults to light and honours an explicit theme', () => {
  assert.equal(captureSnapshot().theme, 'light');
  assert.equal(captureSnapshot('dark').theme, 'dark');
});
