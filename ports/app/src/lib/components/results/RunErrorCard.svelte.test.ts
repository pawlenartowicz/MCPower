import { render, fireEvent } from '@testing-library/svelte';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import RunErrorCard from './RunErrorCard.svelte';
import { runStore } from '$lib/stores/run.svelte';

describe('RunErrorCard', () => {
  beforeEach(() => {
    runStore.lastError = null;
  });
  afterEach(() => {
    runStore.lastError = null;
  });

  it('renders the run failure title and the engine message', () => {
    runStore.lastError = {
      severity: 'run',
      title: 'Run failed',
      message: "factor 'group' has only one observed level",
    };
    const { getByText } = render(RunErrorCard);
    expect(getByText('Run failed')).toBeTruthy();
    expect(getByText(/only one observed level/)).toBeTruthy();
  });

  it('Copy details writes the full detail text to the clipboard', async () => {
    const writeText = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    runStore.lastError = {
      severity: 'run',
      title: 'Run failed',
      message: 'short message',
      detail: 'FULL STACK TEXT',
    };
    const { getByText } = render(RunErrorCard);
    await fireEvent.click(getByText('Copy details'));
    expect(writeText).toHaveBeenCalledWith('FULL STACK TEXT');
  });

  it('omits Copy details when there is no detail', () => {
    runStore.lastError = { severity: 'run', title: 'Run failed', message: 'no detail here' };
    const { queryByText } = render(RunErrorCard);
    expect(queryByText('Copy details')).toBeNull();
  });

  it('Dismiss clears the run error', async () => {
    runStore.lastError = { severity: 'run', title: 'Run failed', message: 'msg' };
    const { getByText } = render(RunErrorCard);
    await fireEvent.click(getByText('Dismiss'));
    expect(runStore.lastError).toBeNull();
  });

  it('frames a cluster_setup error with the dedicated lead-in and no redundant Show toggle', () => {
    const engineMsg = 'FixedSize cluster regime needs cluster_size >= 2; got 1';
    runStore.lastError = {
      severity: 'run',
      kind: 'cluster_setup',
      title: "Cluster and sample-size settings don't fit together",
      message: engineMsg,
      detail: engineMsg, // detail === message → nothing extra to preview
    };
    const { getByText, queryByText } = render(RunErrorCard);
    expect(getByText(/settings don't fit together/i)).toBeTruthy();
    expect(getByText(/adjust the cluster size/i)).toBeTruthy();
    expect(getByText(engineMsg)).toBeTruthy();
    expect(queryByText('Show details')).toBeNull();
  });

  it('reveals extra technical detail only after clicking Show details', async () => {
    runStore.lastError = {
      severity: 'run',
      kind: 'generic',
      title: 'Run failed',
      message: 'short',
      detail: 'LONG STACK TRACE',
    };
    const { getByText, queryByText, findByText } = render(RunErrorCard);
    expect(queryByText('LONG STACK TRACE')).toBeNull();
    await fireEvent.click(getByText('Show details'));
    expect(await findByText('LONG STACK TRACE')).toBeTruthy();
  });
});
