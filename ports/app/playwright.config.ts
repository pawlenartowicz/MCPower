import { defineConfig, devices } from '@playwright/test';

// E2E_BROWSERS selects which browser projects run (comma-separated subset of
// chromium,firefox,webkit). Default chromium-only keeps every existing caller —
// ci.yml's `app` job and app-cross-platform — byte-unaffected; release-web sets
// E2E_BROWSERS=firefox,webkit for its cross-browser UI gate.
const E2E_BROWSERS = (process.env['E2E_BROWSERS'] ?? 'chromium').split(',');

const allProjects = [
  { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
  { name: 'webkit', use: { ...devices['Desktop Safari'] } },
];

export default defineConfig({
  testDir: './tests/e2e',
  // The boot-probe degradation spec only validates under a VITE_BOOT_FAIL server,
  // so it runs solely via playwright.bootfail.config.ts — never the normal run.
  testIgnore: ['**/boot-probe-degradation.spec.ts'],
  timeout: 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: false,
  forbidOnly: !!process.env['CI'],
  retries: 0,
  workers: 1,
  reporter: 'list',
  use: {
    // Dedicated e2e port (NOT the dev/tauri :1420): reuseExistingServer would
    // otherwise silently bind to a running `pnpm dev:tauri` on :1420 — a non-E2E
    // build where the store calls the real Tauri `invoke()` (no runtime in a
    // plain Playwright tab), making every formula-driven spec fail.
    baseURL: 'http://127.0.0.1:1421',
    trace: 'on-first-retry',
  },
  webServer: {
    // VITE_E2E comes from the `env` block below (OS-neutral); a leading
    // `VITE_E2E=true ` shell prefix would not parse on Windows runners.
    command: 'pnpm dev --port 1421 --strictPort',
    url: 'http://127.0.0.1:1421',
    reuseExistingServer: !process.env['CI'],
    stdout: 'pipe',
    stderr: 'pipe',
    timeout: 60_000,
    env: { VITE_E2E: 'true' },
  },
  projects: allProjects.filter((p) => E2E_BROWSERS.includes(p.name)),
});
