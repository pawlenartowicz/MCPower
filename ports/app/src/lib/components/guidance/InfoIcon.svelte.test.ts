import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import InfoIcon from './InfoIcon.svelte';
import { DOCS_BASE_URL } from '$lib/content/render-doc';

describe('InfoIcon', () => {
  it('shows the backing first paragraph and a see-more link that opens externally', async () => {
    const openSpy = vi.spyOn(window, 'open').mockReturnValue(null);
    // baselineProbability backs concepts/effect-sizes#baseline-probability.
    render(InfoIcon, { props: { tipKey: 'baselineProbability' } });

    await fireEvent.click(screen.getByLabelText('Help for baselineProbability'));

    // The popover renders the section's first paragraph.
    expect(await screen.findByText(/baseline probability/i)).toBeTruthy();

    // Clicking "see more" routes the anchored URL through the system browser.
    await fireEvent.click(await screen.findByText(/see more/i));
    expect(openSpy).toHaveBeenCalledWith(
      `${DOCS_BASE_URL}/concepts/effect-sizes#baseline-probability`,
      '_blank',
      expect.any(String),
    );
    openSpy.mockRestore();
  });
});
