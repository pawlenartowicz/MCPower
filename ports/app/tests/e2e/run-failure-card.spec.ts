import { test, expect } from '@playwright/test';

// Regression guard: a failed run must surface the engine's message in the results error
// card — never the old StatusBar "● Error — see console" dead-end (a packaged app has no
// console). The e2e engine mock rejects any run whose outcome is the sentinel "boom".
test('a failed run shows the error card with the engine message and a working copy button', async ({
  page,
  context,
  browserName,
}) => {
  // Clipboard read-back is a Chromium-only capability in Playwright automation;
  // firefox/webkit don't support the `clipboard-read` permission and will throw.
  if (browserName === 'chromium') {
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  }
  await page.goto('/');

  const formula = page.getByPlaceholder(/Write your formula/i);
  await formula.fill('boom ~ x1 + x2');
  await page.waitForTimeout(300); // let the formula parse + effect rows render

  await page.getByTestId('effect-x1').locator('input').fill('0.3');
  await page.getByTestId('effect-x2').locator('input').fill('0.2');

  await page.getByRole('button', { name: /Find power/i }).click();

  // The run-failure card carries the engine message instead of vanishing into the console.
  await expect(page.getByText('Run failed')).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText(/only one observed level/)).toBeVisible();

  // The dead-end can never come back.
  await expect(page.getByText(/see console/)).toHaveCount(0);

  // Copy details puts the full engine text on the clipboard (the bit once lost to console.error).
  // Button click runs on every browser — verifies it renders and is clickable without crashing.
  await page.getByRole('button', { name: /Copy details/i }).click();
  // Clipboard read-back is a Chromium-only capability in Playwright automation;
  // firefox/webkit don't support the `clipboard-read` permission.
  if (browserName === 'chromium') {
    const clip = await page.evaluate(() => navigator.clipboard.readText());
    expect(clip).toContain('only one observed level');
  }
});
