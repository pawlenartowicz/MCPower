import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
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
    command: 'VITE_E2E=true pnpm dev --port 1421 --strictPort',
    url: 'http://127.0.0.1:1421',
    reuseExistingServer: !process.env['CI'],
    stdout: 'pipe',
    stderr: 'pipe',
    timeout: 60_000,
    env: { VITE_E2E: 'true' },
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});
