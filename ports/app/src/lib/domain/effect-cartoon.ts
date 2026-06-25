// Pure math for the EffectCartoon preview. No DOM, no engine. Every function is
// an ILLUSTRATION of a single standardized effect, not the simulation. Cloud
// noise is drawn from a fixed-seed mulberry32 POOL of pre-vetted survivor samples
// (built once at module load): the cloud RESAMPLES as the user nudges a setter,
// but every survivor is screened for normality first so no frame shows an ugly
// outlier-heavy draw. Each cloud helper takes a `sampleIndex` selecting which
// survivor to use (the component advances a counter per value change), so output
// is deterministic + testable yet visibly reshuffles. The x-axis clamps to
// ±X_CLAMP SD; y is auto-fit per chart via fitRange (no fixed ±3 clip), so a
// cloud visibly moves as its effect changes.
import { mulberry32 } from '$lib/util/seeded-prng';

export const X_CLAMP = 5; // was X_MIN/X_MAX = ±3 (#9)

const CLAMP = 1e-6;
export function logit(p: number): number {
  const c = Math.min(1 - CLAMP, Math.max(CLAMP, p));
  return Math.log(c / (1 - c));
}

function logistic(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

// Standard-normal draw from the seeded stream (Box–Muller).
function boxMuller(rng: () => number): number {
  const u = Math.max(CLAMP, rng());
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// --- Resampling pool ---------------------------------------------------------
// Draw many candidate clouds once (seed 2137), score each by departure-from-
// normal, and keep the cleanest as a fixed rotation pool. The component indexes
// it by a per-change counter so the cloud resamples without ever showing a frame
// where a lone far outlier blows up fitRange and squashes everything else.
const CANDIDATE_COUNT = 100;
export const POOL_SIZE = 20; // survivors kept (worst 80 of 100 dropped)
const CONT_N = 70; // points per continuous scatter
const GROUP_COLS = 6; // columns available to grouped/grid clouds
const GROUP_N = 24; // points per column

interface SampleBundle {
  // x = predictor, n = gaussian residual (also reused as the binary cloud's
  // vertical jitter), u = uniform for the Bernoulli outcome draw in the
  // binary-outcome sigmoid scatter.
  cont: { x: number; n: number; u: number }[];
  group: { jit: number; n: number }[][]; // grouped/grid column clouds
}

// Departure-from-normal badness for a set of ~standard-normal draws. |skew| +
// |excess kurtosis| catch asymmetry/heavy tails; the maxAbs term past 3.3 is the
// visually decisive one — a single far point dominates the per-chart auto-fit.
// Lower = tidier. Channels are scored on their own mean/sd (scale-invariant),
// except the maxAbs term, which reads raw values (they are ~standard normal).
function normalBadness(values: number[]): number {
  const n = values.length;
  let m = 0;
  for (const v of values) m += v;
  m /= n;
  let m2 = 0;
  let m3 = 0;
  let m4 = 0;
  let maxAbs = 0;
  for (const v of values) {
    const d = v - m;
    m2 += d * d;
    m3 += d * d * d;
    m4 += d * d * d * d;
    if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
  }
  m2 /= n;
  m3 /= n;
  m4 /= n;
  const sd = Math.sqrt(m2);
  const skew = sd > 0 ? m3 / (sd * sd * sd) : 0;
  const exKurt = m2 > 0 ? m4 / (m2 * m2) - 3 : 0;
  return Math.abs(skew) + Math.abs(exKurt) + Math.max(0, maxAbs - 3.3);
}

// Build CANDIDATE_COUNT bundles from one seeded stream, score by the combined
// normality of every gaussian channel (x, residual, column residuals), and keep
// the POOL_SIZE tidiest. Built once at module load; jit (uniform spread) is not
// scored — only the gaussian channels carry the "looks normal" requirement.
export const POOL: SampleBundle[] = (() => {
  const rng = mulberry32(2137);
  const candidates: { bundle: SampleBundle; score: number }[] = [];
  for (let c = 0; c < CANDIDATE_COUNT; c++) {
    // u drawn AFTER x/n so the continuous-scatter x/n stay byte-identical to
    // before the binary channel was added; only the group stream shifts.
    const cont = Array.from({ length: CONT_N }, () => ({
      x: boxMuller(rng),
      n: boxMuller(rng),
      u: rng(),
    }));
    const group = Array.from({ length: GROUP_COLS }, () =>
      Array.from({ length: GROUP_N }, () => ({ jit: rng() * 2 - 1, n: boxMuller(rng) })),
    );
    const score =
      normalBadness(cont.map((p) => p.x)) +
      normalBadness(cont.map((p) => p.n)) +
      normalBadness(group.flat().map((p) => p.n));
    candidates.push({ bundle: { cont, group }, score });
  }
  candidates.sort((a, b) => a.score - b.score);
  return candidates.slice(0, POOL_SIZE).map((c) => c.bundle);
})();

// Survivor at a rotation index (wraps; tolerates negatives).
function survivor(sampleIndex: number): SampleBundle {
  return POOL[((sampleIndex % POOL_SIZE) + POOL_SIZE) % POOL_SIZE]!;
}

// P(y=1) across the standardized predictor; intercept is logit(baseline prob).
export function logisticCurve(intercept: number, beta: number, n = 60): { x: number; p: number }[] {
  const out: { x: number; p: number }[] = [];
  for (let i = 0; i < n; i++) {
    const x = -X_CLAMP + (2 * X_CLAMP * i) / (n - 1);
    out.push({ x, p: logistic(intercept + beta * x) });
  }
  return out;
}

// Binary outcome, continuous predictor: the actual 0/1 data behind the sigmoid.
// Each survivor point draws a Bernoulli outcome at its own success prob
// p = logistic(intercept + beta*x) via its seeded uniform u, so the cloud is the
// data the curve is fitted to (1s pile toward high p, 0s toward low p). jit is
// the gaussian residual reused as a small vertical spread so dots at the same
// outcome level don't overplot (the component scales it to a few px). The intercept
// = logit(baseline) shifts every prob, so a low baseline pushes the whole cloud to 0.
export function bernoulliScatter(
  intercept: number,
  beta: number,
  sampleIndex = 0,
): { points: { x: number; outcome: 0 | 1; jit: number }[] } {
  const points = survivor(sampleIndex).cont.map((b) => {
    const p = logistic(intercept + beta * b.x);
    return { x: b.x, outcome: (b.u < p ? 1 : 0) as 0 | 1, jit: b.n };
  });
  return { points };
}

// Binary outcome, grouped/factor predictor: one stacked proportion bar per level.
// betas[0] is the reference (0); pOne = P(y=1) for that level through the logistic
// link, pZero = 1 - pOne. Full-height bar (pOne + pZero == 1); the component draws
// the pOne segment over the pZero segment so you read each group's split and compare.
export function groupOutcomeBars(
  intercept: number,
  betas: number[],
): { pOne: number; pZero: number }[] {
  return betas.map((b) => {
    const pOne = logistic(intercept + b);
    return { pOne, pZero: 1 - pOne };
  });
}

// Fit [lo, hi] to the supplied y-values with ~12% padding and a 0.5 minimum
// span (so a near-zero effect doesn't collapse the axis). Ported from the demo's
// scaleY domain logic. The component maps y -> pixels from this range, replacing
// the fixed ±3 clip: nothing is cut off and the cloud visibly moves (#10).
export function fitRange(values: number[]): { lo: number; hi: number } {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    lo = -1;
    hi = 1;
  }
  if (hi - lo < 0.5) {
    hi += 0.5;
    lo -= 0.5;
  }
  const pad = (hi - lo) * 0.12;
  return { lo: lo - pad, hi: hi + pad };
}

// y = beta*x + n over the survivor's points: predictor x has SD 1 and the error
// n has SD 1 FIXED (so var(y) = 1 + beta^2). This is the actual model the engine
// simulates, NOT a standardized/correlation picture — the residual cloud keeps
// its full width at every beta and never collapses onto the line (a standardized
// sqrt(1-beta^2) residual would vanish at beta=1, hiding the scatter). Trend
// endpoints at x=±2.6; the component fits y via fitRange(points.y ++ trend.y).
export function continuousScatter(
  beta: number,
  sampleIndex = 0,
): {
  points: { x: number; y: number }[];
  trend: { x1: number; y1: number; x2: number; y2: number };
} {
  const points = survivor(sampleIndex).cont.map((b) => ({ x: b.x, y: beta * b.x + b.n }));
  return { points, trend: { x1: -2.6, y1: beta * -2.6, x2: 2.6, y2: beta * 2.6 } };
}

// means[0] is the reference (0); means[i] is the beta of level i (SD units). One
// column per group: a dot cloud (mean + survivor column residual) and a mean
// line. x per point = column index + jit-based horizontal spread (component maps
// to pixels).
export function groupedColumns(
  means: number[],
  sampleIndex = 0,
): {
  columns: { mean: number; points: { x: number; y: number }[] }[];
} {
  const base = survivor(sampleIndex).group;
  return {
    columns: means.map((mean, idx) => ({
      mean,
      // Factors can have more levels than the pool has columns (GROUP_COLS); wrap
      // so columns past the pool reuse an earlier cloud instead of rendering bare.
      points: (base[idx % GROUP_COLS] ?? []).map((pt) => ({ x: idx + pt.jit * 0.32, y: mean + pt.n })),
    })),
  };
}

// 2×2 cell means [00, A, B, A+B+int] and the no-interaction reference A+B (the
// gap between the 4th mean and ref is the interaction).
export function twoByTwo(
  betaA: number,
  betaB: number,
  interactionBeta: number,
): { cells: [number, number, number, number]; noInteractionRef: number } {
  return {
    cells: [0, betaA, betaB, betaA + betaB + interactionBeta],
    noInteractionRef: betaA + betaB,
  };
}

// Four columns; cells[i] is the mean, the survivor's column i the cloud. The 4th
// column additionally carries the dashed no-interaction reference (betaA+betaB).
export function gridColumns(
  betaA: number,
  betaB: number,
  interactionBeta: number,
  sampleIndex = 0,
): {
  columns: { mean: number; ref: number | null; points: { x: number; y: number }[] }[];
} {
  const { cells, noInteractionRef } = twoByTwo(betaA, betaB, interactionBeta);
  const base = survivor(sampleIndex).group;
  return {
    columns: cells.map((mean, idx) => ({
      mean,
      ref: idx === 3 ? noInteractionRef : null,
      points: (base[idx] ?? []).map((pt) => ({ x: idx + pt.jit * 0.3, y: mean + pt.n })),
    })),
  };
}

// group0 (binary=0): slope contBeta; group1 (binary=1): slope contBeta+interactionBeta.
export function twoSlopes(
  contBeta: number,
  interactionBeta: number,
): { slope0: number; slope1: number } {
  return { slope0: contBeta, slope1: contBeta + interactionBeta };
}

// group0 (binary=0): slope contBeta, intercept 0.
// group1 (binary=1): slope contBeta+interactionBeta, intercept binShift.
// Half the survivor's points scatter around each line (residual SD 0.7 — tighter
// than the main scatter so two overlapping groups stay legible). Data coords.
export function slopeScatter(
  contBeta: number,
  interactionBeta: number,
  binShift: number,
  sampleIndex = 0,
): {
  group0: {
    line: { x1: number; y1: number; x2: number; y2: number };
    points: { x: number; y: number }[];
  };
  group1: {
    line: { x1: number; y1: number; x2: number; y2: number };
    points: { x: number; y: number }[];
  };
} {
  const { slope0, slope1 } = twoSlopes(contBeta, interactionBeta);
  const line0 = { x1: -2.6, y1: slope0 * -2.6, x2: 2.6, y2: slope0 * 2.6 };
  const line1 = { x1: -2.6, y1: binShift + slope1 * -2.6, x2: 2.6, y2: binShift + slope1 * 2.6 };
  const p0: { x: number; y: number }[] = [];
  const p1: { x: number; y: number }[] = [];
  survivor(sampleIndex).cont.forEach((b, idx) => {
    if (idx % 2 === 0) p0.push({ x: b.x, y: slope0 * b.x + 0.7 * b.n });
    else p1.push({ x: b.x, y: binShift + slope1 * b.x + 0.7 * b.n });
  });
  return { group0: { line: line0, points: p0 }, group1: { line: line1, points: p1 } };
}
