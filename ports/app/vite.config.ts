/// <reference types="vitest/config" />
import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import tailwindcss from '@tailwindcss/vite';
import path from 'node:path';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));

// App version for the bug-report link (port=app|wasm&version=…). Each shell owns
// its own version (§5): wasm reads ports/wasm/package.json, Tauri reads
// tauri.conf.json. Inlined as __MCPOWER_VERSION__ via define below.
const versionFile =
  process.env['VITE_TARGET'] === 'wasm'
    ? path.resolve(__dirname, '../wasm/package.json')
    : path.resolve(__dirname, 'src-tauri/tauri.conf.json');
const appVersion = JSON.parse(readFileSync(versionFile, 'utf8')).version as string;

export default defineConfig({
  // Web deploy serves the app at mcpower.app/online/ (Cloudflare Pages); Tauri stays at /.
  base: process.env['VITE_TARGET'] === 'wasm' ? '/online/' : '/',
  plugins: [tailwindcss(), svelte()],
  resolve: {
    alias: [
      ...(process.env['VITE_E2E'] === 'true' ? [
        {
          find: '$lib/api/engine',
          replacement: path.resolve(__dirname, 'src/lib/api/engine-e2e-mock.ts'),
        },
      ] : []),
      // Browser/WASM build: swap the engine seam for the worker-pool client and
      // resolve the standalone @mcpower/engine-wasm package from source (no pnpm
      // workspace — consumed via this alias). Both must precede the generic $lib.
      ...(process.env['VITE_TARGET'] === 'wasm' ? [
        {
          find: '$lib/api/engine',
          replacement: path.resolve(__dirname, 'src/lib/api/engine-wasm.ts'),
        },
        {
          find: '@mcpower/engine-wasm',
          replacement: path.resolve(__dirname, '../wasm/src/index.ts'),
        },
      ] : []),
      // Vitest: make @mcpower/engine-wasm resolvable for vi.mock (VITE_TARGET not set in test runs).
      ...(process.env['VITEST'] ? [
        {
          find: '@mcpower/engine-wasm',
          replacement: path.resolve(__dirname, '../wasm/src/index.ts'),
        },
      ] : []),
      { find: '$lib', replacement: path.resolve(__dirname, 'src/lib') },
      { find: '$configs', replacement: path.resolve(__dirname, '../../configs') },
      { find: '$docs', replacement: path.resolve(__dirname, '../../web/documentation') },
    ],
    conditions: process.env['VITEST'] ? ['browser', 'module', 'import', 'default'] : undefined,
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/tests/setup.ts'],
    include: ['src/**/*.{test,spec}.{js,ts,svelte}'],
    exclude: ['node_modules', 'e2e', 'src-tauri'],
    server: {
      deps: {
        inline: ['svelte'],
      },
    },
  },
  define: {
    __MCPOWER_VERSION__: JSON.stringify(appVersion),
  },
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: '127.0.0.1',
    fs: { allow: [path.resolve(__dirname, '../..')] },
  },
  envPrefix: ['VITE_', 'TAURI_'],
  // Module workers (the WASM worker pool spawns `new Worker(url, { type: 'module' })`).
  worker: { format: 'es' },
  build: {
    target: 'esnext',
    minify: !process.env['TAURI_ENV_DEBUG'],
    sourcemap: !!process.env['TAURI_ENV_DEBUG'],
  },
});
