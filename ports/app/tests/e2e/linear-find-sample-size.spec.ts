import { test, expect } from '@playwright/test';

test('linear find-sample-size happy path renders a sample-size result table', async ({ page }) => {
  await page.goto('/');

  // The page lands on the Linear family by default.
  const formula = page.getByPlaceholder(/Write your formula/i);
  await formula.fill('y ~ x1 + x2');

  // Allow a tick for Svelte to process the formula change and render effect rows.
  await page.waitForTimeout(300);

  await page.getByTestId('effect-x1').locator('input').fill('0.3');
  await page.getByTestId('effect-x2').locator('input').fill('0.2');

  // The "Find sample" button is in StatusBar; it dispatches startFindSampleSize
  // using the pre-configured bounds from familyStore (no separate bounds form).
  // Button text: "Find sample" (StatusBar.svelte line 130).
  await page.getByRole('button', { name: /Find sample/i }).click();

  // The sample-size result renders in a <table data-testid="sample-size-table">
  // (TableView.svelte line 261 — fires when tab.kind === 'find-sample-size').
  await expect(page.getByTestId('sample-size-table')).toBeVisible({ timeout: 10_000 });
  // Ensure the table contains actual numeric data, not just a visible-but-empty shell.
  await expect(page.getByTestId('sample-size-table')).toContainText(/\d/);
});
