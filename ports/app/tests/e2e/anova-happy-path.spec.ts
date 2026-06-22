import { test, expect } from '@playwright/test';

test('anova find-power happy path renders bars', async ({ page }) => {
  await page.goto('/');

  // Switch to the ANOVA family. FamilyRibbon renders each family as role="radio"
  // with aria-label={FAMILY_LABEL[f]}; FAMILY_LABEL.anova === "ANOVA".
  await page.getByRole('radio', { name: 'ANOVA' }).click();

  // The default ANOVA config has no factors — click "Add factor" to create one
  // (named "F1" with 2 levels).
  await page.getByRole('button', { name: 'Add factor' }).click();

  // Rename the new factor (default name "F1") to "treatment".
  await page.getByLabel('Variable name').first().fill('treatment');

  // Factor levels are a numeric COUNT (NumberInput), not named/comma-separated
  // values. Bump the count to 3 → dummies treatment[2], treatment[3]
  // (treatment[1] is the auto reference level). The level-count is the first
  // spinbutton in the factor editor.
  await page.getByRole('spinbutton').first().fill('3');
  // Tab to commit the level resize before opening the Advanced dialog, so the
  // dialog sees three share rows rather than the pre-resize two.
  await page.getByRole('spinbutton').first().press('Tab');

  // Shares are weights (auto-rescaled by the adapter), so the run is valid
  // either way — the Advanced-dialog detour exercises the relocated Rescale
  // control: open the card's ⚙ dialog, rescale, close.
  await page.getByTestId('advanced-treatment').click();
  await page.getByRole('button', { name: 'Rescale shares to sum to 100%' }).click();
  await page.keyboard.press('Escape');

  // Wait for Svelte reactivity to propagate the variable changes to the effects store.
  await page.waitForTimeout(300);

  // Fill effect sizes for the non-reference dummies.
  // EffectControls (inside the factor's VariableCard) renders
  // data-testid="effect-treatment[2]" and "effect-treatment[3]".
  // The reference row (treatment[1]) is shown as disabled text — no testid.
  await page.getByTestId('effect-treatment[2]').locator('input').fill('0.5');
  await page.getByTestId('effect-treatment[3]').locator('input').fill('0.3');

  // Run button is in StatusBar.
  await page.getByRole('button', { name: /Find power/i }).click();

  // Bars view should render within a few seconds (mock has 50ms delay).
  await expect(page.getByTestId('bars-view')).toBeVisible({ timeout: 10_000 });

  // BarsView renders an sr-only <span data-testid="bar-{name}"> per effect name.
  // An ANOVA spec is built with estimator family 'linear', so the result's
  // effect_names come from the predictor list — one bar per factor, named by the
  // factor ("treatment"), not per dummy level.
  await expect(page.getByTestId('bar-treatment')).toBeVisible();
});
