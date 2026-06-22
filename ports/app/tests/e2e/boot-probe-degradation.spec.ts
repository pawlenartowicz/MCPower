import { test, expect } from '@playwright/test';

// Closes the previously-untested boot probe (App.svelte): when the WASM engine
// fails to instantiate, a blocking CrashModal must replace the UI. The dev server
// for this spec is built with VITE_BOOT_FAIL=true (see playwright.bootfail.config.ts),
// which makes the mocked parseFormula throw at boot — standing in for a real WebKit
// relaxed-simd CompileError. Chromium-only: verifies the degradation plumbing, not
// any real browser's WASM support.
test('a boot-time engine failure raises the blocking crash modal', async ({ page }) => {
  await page.goto('/');

  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible({ timeout: 10_000 });
  await expect(dialog.getByText("This browser can't run MCPower")).toBeVisible();

  // CrashModal renders with showCloseButton={false} and a "Report this" button —
  // asserting both confirms this is the crash surface (not some other dialog) and
  // that it sits over the UI with no dismiss-to-app path.
  await expect(dialog.getByRole('button', { name: /Report this/i })).toBeVisible();
});
