// Theme types and system-theme detection utilities for the Tauri app; ResolvedTheme is the canonical set of concrete themes.
export type ResolvedTheme = 'light' | 'dark';

export function getSystemTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return 'light';
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
