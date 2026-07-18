// Behaviour tests for UploadDialog: the store is only touched on confirm, and a
// row-count failure blocks the commit. Parsing runs through the real parser layer
// (PapaParse in jsdom); the real uploadStore singleton verifies commit-gating.
import { fireEvent, render, waitFor } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import UploadDialog from './UploadDialog.svelte';
import { uploadStore } from '$lib/stores/upload.svelte';

// A comma CSV with `n` data rows (min upload is 20).
function csv(n: number): File {
  const body = Array.from({ length: n }, (_, i) => `${i},${i * 2}`).join('\n');
  return new File([`a,b\n${body}`], 'data.csv', { type: 'text/csv' });
}

describe('UploadDialog', () => {
  beforeEach(() => {
    uploadStore.clear();
  });

  it('commits to the store only on "Use this data"', async () => {
    const onOpenChange = vi.fn();
    const { getByText } = render(UploadDialog, { file: csv(25), open: true, onOpenChange });

    // Wait until the parse finished and confirm is enabled.
    const useBtn = await waitFor(() => {
      const b = getByText('Use this data') as HTMLButtonElement;
      expect(b.disabled).toBe(false);
      return b;
    });
    expect(uploadStore.csvData).toBeNull(); // untouched before confirm

    await fireEvent.click(useBtn);
    await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false));
    expect(uploadStore.csvData).not.toBeNull();
    expect(uploadStore.csvData?.columns.map((c) => c.name)).toEqual(['a', 'b']);
    expect(uploadStore.filename).toBe('data.csv');
  });

  it('leaves the store untouched on Cancel', async () => {
    const onOpenChange = vi.fn();
    const { getByText } = render(UploadDialog, { file: csv(25), open: true, onOpenChange });
    await waitFor(() => expect((getByText('Use this data') as HTMLButtonElement).disabled).toBe(false));

    await fireEvent.click(getByText('Cancel'));
    expect(onOpenChange).toHaveBeenCalledWith(false);
    expect(uploadStore.csvData).toBeNull();
  });

  it('blocks commit and shows an error when under the row minimum', async () => {
    const onOpenChange = vi.fn();
    const { getByText } = render(UploadDialog, { file: csv(3), open: true, onOpenChange });

    await waitFor(() => expect(getByText(/at least 20/)).toBeTruthy());
    expect((getByText('Use this data') as HTMLButtonElement).disabled).toBe(true);
    expect(uploadStore.csvData).toBeNull();
  });
});
