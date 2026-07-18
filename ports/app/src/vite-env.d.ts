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

// App version inlined at build time (vite `define`) for the bug-report link.
declare const __MCPOWER_VERSION__: string;

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

// sav-reader ships types at dist/index.d.ts but its `exports` map has no `types`
// condition, so TS can't resolve them. Declare the minimal surface the uploader uses.
declare module 'sav-reader' {
  interface SavSysVar {
    name: string;
    label: string;
    __is_child_string_var: boolean;
  }
  export class SavBufferReader {
    constructor(buffer: unknown);
    // header.n_cases is the file's declared row count (-1 if unknown), read
    // right after open() to reject an oversized file before readAllRows().
    meta: { sysvars: SavSysVar[]; header: { n_cases: number } };
    open(): Promise<void>;
    readAllRows(): Promise<Record<string, unknown>[]>;
  }
}
