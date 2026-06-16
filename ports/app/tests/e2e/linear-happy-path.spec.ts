import { test, expect } from '@playwright/test';

test('linear find-power happy path renders bars', async ({ page }) => {
  await page.goto('/');

  // The page should land on the Linear family by default.
  // Placeholder text: "Write your formula — e.g. y = x1 + x2"
  // The parser (formula-linear.ts) requires tilde syntax: y ~ x1 + x2
  const formula = page.getByPlaceholder(/Write your formula/i);
  await formula.fill('y ~ x1 + x2');

  // Effect rows appear after the formula is parsed (reactive store updates).
  // Allow a tick for Svelte to process the formula change.
  await page.waitForTimeout(300);

  await page.getByTestId('effect-x1').locator('input').fill('0.3');
  await page.getByTestId('effect-x2').locator('input').fill('0.2');

  // Run button is in StatusBar.
  await page.getByRole('button', { name: /Find power/i }).click();

  // Bars view should render within a few seconds (mock has 50ms delay).
  await expect(page.getByTestId('bars-view')).toBeVisible({ timeout: 10_000 });
  await expect(page.getByTestId('bar-x1')).toBeVisible();
  await expect(page.getByTestId('bar-x2')).toBeVisible();
});
