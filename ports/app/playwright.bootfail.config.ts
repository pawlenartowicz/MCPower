import { defineConfig, devices } from '@playwright/test';

// Dedicated config for the boot-probe degradation spec: the dev server is built
// with VITE_BOOT_FAIL=true so App.svelte's engine probe throws at boot and raises
// the blocking CrashModal. Kept separate from playwright.config.ts because
// Playwright's webServer env is global — one config can't serve both a normal and
// a boot-failing build. Distinct port (1422) so it never collides with the normal
// e2e server. Chromium-only: this verifies the degradation plumbing, not a real
// browser's WASM behavior.
export default defineConfig({
  testDir: './tests/e2e',
  testMatch: ['**/boot-probe-degradation.spec.ts'],
  timeout: 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: false,
  forbidOnly: !!process.env['CI'],
  retries: 0,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:1422',
    trace: 'on-first-retry',
  },
  webServer: {
    command: 'pnpm dev --port 1422 --strictPort',
    url: 'http://127.0.0.1:1422',
    reuseExistingServer: !process.env['CI'],
    stdout: 'pipe',
    stderr: 'pipe',
    timeout: 60_000,
    env: { VITE_E2E: 'true', VITE_BOOT_FAIL: 'true' },
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});
