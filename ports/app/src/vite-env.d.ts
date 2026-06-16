// Vite + Svelte ambient type declarations; extends ImportMetaEnv with the VITE_TARGET build flag.
/// <reference types="svelte" />
/// <reference types="vite/client" />

declare module '*.svelte' {
  import type { Component } from 'svelte';

  const component: Component;
  export default component;
}

interface ImportMetaEnv {
  readonly VITE_TARGET?: 'tauri' | 'wasm';
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
