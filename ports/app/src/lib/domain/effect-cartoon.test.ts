import { describe, expect, it } from 'vitest';
import {
  bernoulliScatter,
  continuousScatter,
  fitRange,
  gridColumns,
  groupedColumns,
  groupOutcomeBars,
  logisticCurve,
  logit,
  POOL,
  POOL_SIZE,
  slopeScatter,
  X_CLAMP,
} from './effect-cartoon';

describe('logit', () => {
  it('maps 0.5 to 0 and is monotone', () => {
    expect(logit(0.5)).toBeCloseTo(0, 10);
    expect(logit(0.8)).toBeGreaterThan(logit(0.2));
  });
});

describe('logisticCurve', () => {
  it('produces a monotone curve bounded in (0,1) spanning ±X_CLAMP', () => {
    const pts = logisticCurve(logit(0.3), 0.7);
    expect(pts[0]!.x).toBeCloseTo(-X_CLAMP);
    expect(pts[pts.length - 1]!.x).toBeCloseTo(X_CLAMP);
    for (const { p } of pts) {
      expect(p).toBeGreaterThan(0);
      expect(p).toBeLessThan(1);
    }
    for (let i = 1; i < pts.length; i++) {
      expect(pts[i]!.p).toBeGreaterThanOrEqual(pts[i - 1]!.p); // positive beta => increasing
    }
  });

  it('a flat (beta=0) curve sits at the baseline probability', () => {
    const pts = logisticCurve(logit(0.4), 0);
    for (const { p } of pts) expect(p).toBeCloseTo(0.4, 6);
  });
});

describe('groupOutcomeBars', () => {
  it('one bar per group; pOne is logistic(intercept+beta) and pOne+pZero==1', () => {
    const bars = groupOutcomeBars(logit(0.2), [0, 1.0, 2.0]);
    expect(bars).toHaveLength(3);
    expect(bars[0]!.pOne).toBeCloseTo(0.2, 6); // reference = baseline
    for (const b of bars) expect(b.pOne + b.pZero).toBeCloseTo(1, 12);
  });

  it('a higher level beta lifts the colored share, monotone in beta', () => {
    const bars = groupOutcomeBars(logit(0.3), [0, 0.5, 1.5]);
    expect(bars[1]!.pOne).toBeGreaterThan(bars[0]!.pOne);
    expect(bars[2]!.pOne).toBeGreaterThan(bars[1]!.pOne);
  });

  it('a higher baseline lifts every group equally toward 1', () => {
    const lo = groupOutcomeBars(logit(0.1), [0, 0.5]);
    const hi = groupOutcomeBars(logit(0.6), [0, 0.5]);
    expect(hi[0]!.pOne).toBeGreaterThan(lo[0]!.pOne);
    expect(hi[1]!.pOne).toBeGreaterThan(lo[1]!.pOne);
  });
});

describe('bernoulliScatter', () => {
  it('resamples with sampleIndex and wraps modulo POOL_SIZE', () => {
    const i0 = bernoulliScatter(logit(0.3), 0.8, 0).points;
    const i1 = bernoulliScatter(logit(0.3), 0.8, 1).points;
    expect(i0).not.toEqual(i1);
    expect(bernoulliScatter(logit(0.3), 0.8, POOL_SIZE).points).toEqual(i0);
  });

  it('a higher baseline yields more 1s (the cloud lifts toward y=1)', () => {
    const ones = (p: number) =>
      bernoulliScatter(logit(p), 0.5, 0).points.filter((q) => q.outcome === 1).length;
    expect(ones(0.8)).toBeGreaterThan(ones(0.15));
  });

  it('a positive slope puts more 1s at high x than at low x', () => {
    const pts = bernoulliScatter(logit(0.5), 1.5, 0).points;
    const hiX = pts.filter((p) => p.x > 0.5);
    const loX = pts.filter((p) => p.x < -0.5);
    const rate = (s: typeof pts) => s.filter((p) => p.outcome === 1).length / Math.max(1, s.length);
    expect(rate(hiX)).toBeGreaterThan(rate(loX));
  });
});

describe('fitRange', () => {
  it('pads the data span by ~12% each side', () => {
    const { lo, hi } = fitRange([0, 1]);
    // span 1 -> 0.12 padding each side
    expect(lo).toBeCloseTo(-0.12, 10);
    expect(hi).toBeCloseTo(1.12, 10);
  });

  it('expands a near-zero span so the axis does not collapse', () => {
    const { lo, hi } = fitRange([0.1, 0.1]);
    // span < 0.5 -> widened by 0.5 each side to [-0.4, 0.6], then padded 12% of 1.0
    expect(lo).toBeCloseTo(-0.52, 10);
    expect(hi).toBeCloseTo(0.72, 10);
  });

  it('falls back to [-1,1] (then padded) on empty / non-finite input', () => {
    const { lo, hi } = fitRange([]);
    expect(lo).toBeCloseTo(-1.24, 10);
    expect(hi).toBeCloseTo(1.24, 10);
  });
});

describe('resampling pool', () => {
  it('survivors are screened for outliers (no far point past the penalty band)', () => {
    // The maxAbs penalty kicks in past 3.3; the cleanest 20 of 100 should sit
    // comfortably below it on every gaussian channel, so no frame squashes.
    for (const b of POOL) {
      const xs = b.cont.map((p) => Math.abs(p.x));
      const ns = b.cont.map((p) => Math.abs(p.n));
      expect(Math.max(...xs)).toBeLessThan(3.5);
      expect(Math.max(...ns)).toBeLessThan(3.5);
    }
  });
});

describe('continuousScatter', () => {
  it('resamples: a different sampleIndex yields a different cloud', () => {
    const a = continuousScatter(0.4, 0).points;
    const b = continuousScatter(0.4, 1).points;
    expect(a).not.toEqual(b);
  });

  it('wraps the index modulo POOL_SIZE (rotation cycles)', () => {
    expect(continuousScatter(0.4, POOL_SIZE).points).toEqual(continuousScatter(0.4, 0).points);
    expect(continuousScatter(0.4, -1).points).toEqual(continuousScatter(0.4, POOL_SIZE - 1).points);
  });

  it('trend rises for a positive slope and falls for a negative slope', () => {
    const up = continuousScatter(0.5).trend;
    expect(up.y2).toBeGreaterThan(up.y1);
    const down = continuousScatter(-0.5).trend;
    expect(down.y2).toBeLessThan(down.y1);
  });

  it('residual spread is the same at every beta (fixed unit error, not standardized)', () => {
    const resid = (beta: number) => {
      const ys = continuousScatter(beta).points.map((p) => p.y - beta * p.x);
      const mean = ys.reduce((a, b) => a + b, 0) / ys.length;
      return ys.reduce((a, b) => a + (b - mean) ** 2, 0) / ys.length;
    };
    // y = beta*x + n with n ~ N(0,1): subtracting beta*x recovers n exactly, so
    // the residual variance is identical regardless of beta — the cloud never
    // collapses onto the line (the bug the standardized sqrt(1-beta^2) had).
    expect(resid(0.95)).toBeCloseTo(resid(0.2), 10);
    expect(resid(0.2)).toBeGreaterThan(0.5); // genuine spread, ~unit variance
  });
});

describe('groupedColumns', () => {
  it('returns one column per mean with the supplied means', () => {
    const { columns } = groupedColumns([0, 0.4, 0.8]);
    expect(columns).toHaveLength(3);
    expect(columns.map((c) => c.mean)).toEqual([0, 0.4, 0.8]);
    for (const c of columns) expect(c.points.length).toBeGreaterThan(0);
  });

  it('shifts a column cloud up with its mean', () => {
    const ref = groupedColumns([0, 0]).columns[1]!;
    const lifted = groupedColumns([0, 1]).columns[1]!;
    const meanY = (pts: { y: number }[]) => pts.reduce((a, p) => a + p.y, 0) / pts.length;
    expect(meanY(lifted.points)).toBeGreaterThan(meanY(ref.points));
  });

  it('resamples its clouds with sampleIndex (factor/binary predictor too)', () => {
    const a = groupedColumns([0, 0.4, 0.8], 0).columns.map((c) => c.points);
    const b = groupedColumns([0, 0.4, 0.8], 1).columns.map((c) => c.points);
    expect(a).not.toEqual(b);
  });
});

describe('gridColumns', () => {
  it('gridColumns has 4 columns [00, A, B, A+B+int]; only the 4th carries the no-interaction ref = A+B', () => {
    const { columns } = gridColumns(0.3, 0.5, 0.2);
    expect(columns.map((c) => c.mean)).toEqual([0, 0.3, 0.5, 0.3 + 0.5 + 0.2]);
    expect(columns[0]!.ref).toBeNull();
    expect(columns[3]!.ref).toBeCloseTo(0.8);
  });
});

describe('slopeScatter', () => {
  it('group1 line reflects the combined slope and the binary intercept shift', () => {
    const { group0, group1 } = slopeScatter(0.4, 0.3, 0.6);
    // group0: slope 0.4, intercept 0
    expect(group0.line.y2).toBeCloseTo(0.4 * 2.6);
    // group1: slope 0.7, intercept 0.6
    expect(group1.line.y2).toBeCloseTo(0.6 + 0.7 * 2.6);
    expect(group0.points.length).toBeGreaterThan(0);
    expect(group1.points.length).toBeGreaterThan(0);
  });
});
