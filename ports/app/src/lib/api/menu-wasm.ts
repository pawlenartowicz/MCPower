// Browser file download via an anchor + object URL.
// Replaces the Tauri plugin-dialog save() + plugin-fs write() seam in the WASM build.
export function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  // Firefox only fires the download if the anchor is in the document (mirrors
  // wasm/src/index.ts downloadPlot).
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function downloadJson(filename: string, payload: unknown): void {
  downloadBlob(
    filename,
    new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }),
  );
}
