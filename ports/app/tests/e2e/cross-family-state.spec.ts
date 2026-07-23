import { test, expect } from '@playwright/test';

test('ANOVA ↔ Regression round-trip preserves ANOVA state', async ({ page }) => {
  await page.goto('/');

  // --- Step 1: switch to ANOVA and build a factor ---
  await page.getByRole('radio', { name: 'ANOVA' }).click();

  // Default ANOVA has no factors — add one (named "F1" with 2 levels).
  await page.getByRole('button', { name: 'Add factor' }).click();

  // Rename the default factor to "treatment".
  await page.getByLabel('Variable name').first().fill('treatment');

  // Levels are a numeric count — bump to 3 (dummies treatment[2], treatment[3]).
  await page.getByRole('spinbutton').first().fill('3');

  // Wait for Svelte reactivity to propagate to the effects store.
  await page.waitForTimeout(300);

  // Fill effect sizes for the non-reference dummies.
  await page.getByTestId('effect-treatment[2]').locator('input').fill('0.5');
  await page.getByTestId('effect-treatment[3]').locator('input').fill('0.3');
  await page.waitForTimeout(300);

  // --- Step 2: switch to Regression --- assert ANOVA UI is gone, regression UI is present ---
  await page.getByRole('radio', { name: 'Regression' }).click();

  // FormulaInput is visible in Regression mode.
  const formula = page.getByPlaceholder(/Write your formula/i);
  await expect(formula).toBeVisible();

  // ANOVA factor editor must NOT be present.
  await expect(page.getByRole('button', { name: 'Add factor' })).not.toBeVisible();

  // Set a regression formula to confirm no bleed from ANOVA.
  await formula.fill('y ~ x1');
  await page.waitForTimeout(300);
  await expect(formula).toHaveValue('y ~ x1');

  // --- Step 3: switch BACK to ANOVA --- assert ANOVA state survived ---
  await page.getByRole('radio', { name: 'ANOVA' }).click();

  // Factor name must still be "treatment".
  await expect(page.getByLabel('Variable name').first()).toHaveValue('treatment');

  // Level count must still be 3.
  await expect(page.getByRole('spinbutton').first()).toHaveValue('3');

  // Effect inputs must still hold 0.5 / 0.3.
  await expect(page.getByTestId('effect-treatment[2]').locator('input')).toHaveValue('0.5');
  await expect(page.getByTestId('effect-treatment[3]').locator('input')).toHaveValue('0.3');

  // ANOVA has no formula box — assert isolation the other way.
  await expect(page.getByPlaceholder(/Write your formula/i)).not.toBeVisible();
});

test('cross-family state is isolated per family and survives switching', async ({ page }) => {
  await page.goto('/');

  // Per-family configs are isolated across the three real families
  // (regression / anova / mixed). Logistic is NOT a separate family — it is the
  // Logit outcome toggle inside Regression, sharing the same config object — so
  // isolation here is exercised across Regression ↔ Mixed, plus the persistence
  // of the (global) outcome toggle + baseline across a family round-trip.
  const formula = page.getByPlaceholder(/Write your formula/i);

  // --- Step 1: on Regression, set a formula + flip to Logit + a baseline ---
  await formula.fill('y ~ x1');
  await page.waitForTimeout(300);
  await expect(formula).toHaveValue('y ~ x1');

  await page.getByRole('button', { name: 'Logit' }).click();
  await page.getByLabel(/Baseline probability/i).fill('0.4');

  // --- Step 2: switch to Mixed → its own (empty) formula, isolated from Regression ---
  await page.getByRole('radio', { name: 'Mixed effects' }).click();
  await expect(formula).toHaveValue('');
  await formula.fill('y ~ x2 + (1|school)');
  await page.waitForTimeout(300);

  // --- Step 3: back to Regression → formula, outcome (Logit) and baseline preserved ---
  await page.getByRole('radio', { name: 'Regression' }).click();
  await expect(formula).toHaveValue('y ~ x1');
  await expect(page.getByLabel(/Baseline probability/i)).toHaveValue('0.4');

  // --- Step 4: forward to Mixed → its formula is restored ---
  // The commit-time normalizer in FormulaInput settles `(1|school)` to the spaced
  // `(1 | school)` for legibility, so the restored value carries the spaced pipe.
  await page.getByRole('radio', { name: 'Mixed effects' }).click();
  await expect(formula).toHaveValue('y ~ x2 + (1 | school)');
});
