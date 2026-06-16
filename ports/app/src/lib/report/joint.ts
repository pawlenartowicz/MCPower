// Joint-significance distribution helpers: derive per-sim "exactly k / at-least k" counts from the engine histogram.

export interface JointDistribution {
  exactly: number[];
  atLeast: number[];
}

/** Derive 'exactly k' and 'at least k' from the engine histogram. Returns null on
 * empty histogram / n=0 so callers omit the section. */
export function jointDistribution(
  histogram: number[] | undefined,
  nSimsUsed: number
): JointDistribution | null {
  if (!histogram || histogram.length === 0 || nSimsUsed === 0) return null;
  const n = nSimsUsed;
  const exactly = histogram.map((h) => h / n);
  const atLeast: number[] = [];
  let running = histogram.reduce((a, b) => a + b, 0);
  for (const h of histogram) {
    atLeast.push(running / n);
    running -= h;
  }
  return { exactly, atLeast };
}

