import { test, expect } from '@playwright/test';

test('mixed find-power happy path renders bars', async ({ page }) => {
  await page.goto('/');

  // Switch to the Mixed family. FamilyRibbon renders each family as role="radio"
  // with aria-label={FAMILY_LABEL[f]}; FAMILY_LABEL.mixed === "Mixed effects".
  await page.getByRole('radio', { name: 'Mixed effects' }).click();

  // Enter a random-intercept formula. FormulaInput uses placeholder text
  // (no aria-label on the input itself; use getByPlaceholder).
  const formula = page.getByPlaceholder(/Write your formula/i);
  await formula.fill('y ~ x + (1|school)');

  // Wait for Svelte reactivity to parse the formula, derive the cluster name,
  // and render ClusterEditor + effect rows.
  await page.waitForTimeout(300);

  // 3. Cluster name is derived from the (1|school) term and shown read-only.
  await expect(page.getByTestId('cluster-name')).toHaveText('school');

  // 4. ICC + n_clusters — the n_clusters tab is the default dimKind.
  //    NumberInput renders id="cluster-icc" on its inner <input>.
  await page.locator('input#cluster-icc').fill('0.2');
  await page.locator('input#cluster-n').fill('20');

  // 5. Set a non-zero effect for x.
  //    EffectControls renders data-testid="effect-x" on the NumberInput wrapper div;
  //    the actual input is the child <input> inside.
  await page.getByTestId('effect-x').locator('input').fill('0.5');

  // 6. Find power (StatusBar button).
  await page.getByRole('button', { name: /Find power/i }).click();

  // 7. Bars view should render within a few seconds (mock has 50ms delay).
  await expect(page.getByTestId('bars-view')).toBeVisible({ timeout: 10_000 });

  // BarsView renders an sr-only <span data-testid="bar-{name}"> for each effect.
  // For y ~ x + (1|school), predictors = ['x'], interaction_terms = [] → bar-x.
  await expect(page.getByTestId('bar-x')).toBeVisible();

  // The mock always returns convergence_rate = 1.0, which is >= 0.95 threshold,
  // so the convergence notice should NOT appear.
  await expect(page.getByTestId('convergence-notice')).not.toBeVisible();
});

test('a second (1|item) term renders a crossed cluster card and runs', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('radio', { name: 'Mixed effects' }).click();

  const formula = page.getByPlaceholder(/Write your formula/i);
  await formula.fill('y ~ x + (1|school) + (1|item)');
  // Formula commits after the debounce; the assertions below auto-retry.

  // Primary card mirrors the first term; the second term gets its own card
  // with a "crossed" badge, an ICC input, and an n-clusters input.
  await expect(page.getByTestId('cluster-name')).toHaveText('school');
  const extraCard = page.getByTestId('cluster-extra-item');
  await expect(extraCard).toBeVisible();
  await expect(extraCard).toContainText('crossed');

  // Extra groupings force "by n clusters" — the cluster-size toggle is disabled.
  await expect(page.getByTestId('dim-cluster-size')).toBeDisabled();

  await page.getByTestId('effect-x').locator('input').fill('0.5');
  await page.getByRole('button', { name: /Find power/i }).click();
  await expect(page.getByTestId('bars-view')).toBeVisible({ timeout: 10_000 });
});
