import { defineConfig } from 'vite';
import path from 'node:path';

export default defineConfig({
  resolve: {
    alias: { $configs: path.resolve(__dirname, '../../configs') },
  },
  build: {
    lib: { entry: path.resolve(__dirname, 'src/index.ts'), formats: ['es'], fileName: 'index' },
    target: 'es2022',
  },
  test: { environment: 'node' },
});
