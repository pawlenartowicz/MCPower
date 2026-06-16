import { test, expect } from '@playwright/test';

test('logit find-power happy path renders bars and table baseline', async ({ page }) => {
  await page.goto('/');

  // There is no separate "Logistic" family. Logistic regression is the Binary
  // outcome of the Regression family: land on Regression (default) and flip the
  // outcome toggle to Binary (role="group" aria-label="Outcome type"). The
  // StatusBar adapter then builds an AppSpec::Logit. Click before entering the
  // formula so the only "Binary" button on the page is the outcome toggle (the
  // per-variable type toggles appear only once predictors exist).
  await page.getByRole('button', { name: 'Binary' }).click();

  const formula = page.getByPlaceholder(/Write your formula/i);
  await formula.fill('y ~ x1 + x2');

  // Wait for Svelte reactivity to parse the formula and render effect rows.
  await page.waitForTimeout(300);

  await page.getByTestId('effect-x1').locator('input').fill('0.3');
  await page.getByTestId('effect-x2').locator('input').fill('0.2');

  // Baseline probability appears only in Binary mode. The Label "Baseline
  // probability" is programmatically associated via for="baseline-prob" → the
  // inner <input id="baseline-prob">.
  await page.getByLabel(/Baseline probability/i).fill('0.3');

  // Run button is in StatusBar.
  await page.getByRole('button', { name: /Find power/i }).click();

  // Bars view should render within a few seconds (mock has 50ms delay).
  await expect(page.getByTestId('bars-view')).toBeVisible({ timeout: 10_000 });
  await expect(page.getByTestId('bar-x1')).toBeVisible();
  await expect(page.getByTestId('bar-x2')).toBeVisible();

  // TableView is rendered inside the default Summary tab (alongside the bars chart),
  // so the realized baseline is already on screen — no sub-view switch needed. It
  // surfaces "baseline prob (realized) = NN.N%" when the estimator is glm (the adapter
  // maps Binary regression → AppSpec::Logit → glm estimator).
  await expect(page.getByText(/baseline prob \(realized\)/i)).toBeVisible();
});
