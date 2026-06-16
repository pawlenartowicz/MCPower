// Fake (demo) RunAdapter that replays fixture JSON with a simulated progress ticker.
import type { DemoPowerResult } from '$lib/domain/result';
import linearFixture from '$lib/fixtures/linear-result.json';
import sampleSizeFixture from '$lib/fixtures/sample-size-result.json';
import type { RunAdapter } from './run';

const TOTAL_SIMS = 10_000;
const DURATION_MS = 2_500;
const TICK_MS = 25;

export const fakeAdapter: RunAdapter = {
  start(kind, { onProgress, onDone }) {
    const fixture = (kind === 'find-power'
      ? linearFixture
      : sampleSizeFixture) as unknown as DemoPowerResult;

    const ticks = Math.floor(DURATION_MS / TICK_MS);
    const per = Math.floor(TOTAL_SIMS / ticks);
    let elapsed = 0;
    let completed = 0;
    onProgress({ completed: 0, total: TOTAL_SIMS });

    const id = window.setInterval(() => {
      elapsed += TICK_MS;
      completed = Math.min(TOTAL_SIMS, completed + per);
      onProgress({ completed, total: TOTAL_SIMS });
      if (elapsed >= DURATION_MS) {
        window.clearInterval(id);
        onProgress({ completed: TOTAL_SIMS, total: TOTAL_SIMS });
        onDone(fixture);
      }
    }, TICK_MS);

    return { cancel: () => window.clearInterval(id) };
  },
};
