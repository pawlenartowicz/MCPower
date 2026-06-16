#!/usr/bin/env python3
"""Fit + freeze the f32 inverse-CDF normal kernel and emit its Rust consts.

Kernel form — the sanctioned one-`ln` fallback (B3 checkpoint of the plan).
The spec's aspiration (single branchless rational in w=v², no `ln`) is
NUMERICALLY INFEASIBLE in f32: Phi^-1's slope at the 2^-25 uniform floor is
~3e6, so any single rational matching it needs a pole at p≈1+3e-7 whose
monomial expansion has coefficients ~(3e6)^k — they overflow f32 and Horner
cancellation gives errors >5. (Measured: a near-minimax AAA rational reaches
5e-4 in f64 but diverges once its coeffs are stored/evaluated in f32.) The
fold below sidesteps that by splitting at Acklam's breakpoint so no single
polynomial has to cover the steep tail:

    v = u - 0.5,  a = |v|
    central (a <= BP):  z = v * Pc(v^2)                       # plain poly, no ln
    tail    (a >  BP):  q = sqrt(-2 * ln_f32(t)),             # t = max(0.5-a, FLOOR)
                        z = copysign(Pt(q), v)

`ln_f32` is a cheap pure-f32 log (exponent from the bits + a degree-5 mantissa
log2 poly), NOT libm `ln` — a libm log at one-per-tail-draw would regress the
very __ieee754_log slice this break targets and break cross-host generation
identity (F3). Only ~4.85% of draws (the tail) touch it; the central 95% are a
branchless poly.

Why this is f32-safe where the pure rational was not: over the SPLIT ranges
(central w in [0,0.226], tail q in [2.7,5.9]) every fitted polynomial is bounded
and smooth — no near-edge pole — so its monomial coefficients stay O(1e3) and
evaluate cleanly in f32. Each poly is a LINEAR least-squares fit in a scaled
Chebyshev basis (no nonlinear-rational convergence fragility), converted to
monomial-ascending form for the Rust Horner loops.

Grading: emulate the EXACT f32 op sequence (incl. the bit-trick ln) and compare
to scipy norm.ppf over the same reachable grid the Rust L1 test uses
(u = FLOOR + (1-2*FLOOR)*k/n, graded at u-as-f32). Hard gate: max|Δz| < 2e-3.

FROZEN: changing any emitted coefficient is a golden-breaking, version-bumping
event (golden_rng.rs / data_gen.rs goldens pin to them)."""
import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.stats import norm

# --- reachable uniform range (matches u32_to_unit_f32 + the Rust L1 grid) ---
FLOOR = 0.5 * 2.0**-24          # = 2^-25; also the tail-arg clamp in the kernel
CAP = 1.0 - FLOOR
P_LOW = 0.02425                 # Acklam breakpoint: tail when min(u,1-u) < P_LOW
BP = 0.5 - P_LOW                # equivalently |v| > BP  (BP = 0.47575)

LOG2_DEG = 5                    # mantissa log2 poly, m in [1,2)
CENTRAL_DEG = 10                # Pc(w), w = v^2
TAIL_DEG = 9                    # Pt(q), q = sqrt(-2 ln t)

f32 = np.float32
LN2 = f32(np.log(2.0))


def horner_f32(coeffs, x):
    """Ascending-monomial Horner in float32 — mirrors the Rust kernel loops
    (separate mul then add, NO fma, so the op sequence is reproducible)."""
    x = x.astype(np.float32)
    acc = np.full_like(x, f32(coeffs[-1]))
    for c in coeffs[-2::-1]:
        acc = (acc * x + f32(c)).astype(np.float32)
    return acc


# ---- fit 1: cheap f32 log via exponent split + mantissa log2 poly ----------
mm = np.linspace(1.0, 2.0, 20000)
LOG2_POLY = C.Chebyshev.fit(mm, np.log2(mm), LOG2_DEG, domain=[1.0, 2.0]
                            ).convert(kind=np.polynomial.Polynomial).coef


def ln_f32(t):
    """ln(t) for t>0 in pure f32: t = m * 2^e (m in [1,2)) via the IEEE bits,
    ln = (e + log2_poly(m)) * ln2. Bit ops are exact; only the mantissa poly
    approximates — its error (~1e-6 in log2) is negligible after sqrt(-2*.)."""
    t = t.astype(np.float32)
    bits = t.view(np.uint32)
    e = (((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32) - 127).astype(np.float32)
    m = ((bits & np.uint32(0x7FFFFF)) | np.uint32(0x3F800000)).view(np.float32)
    return ((e + horner_f32(LOG2_POLY, m)) * LN2).astype(np.float32)


# ---- fit 2: central polynomial  z = v * Pc(v^2)  over |v| <= BP -------------
uc = np.unique(np.clip(np.concatenate([
    np.linspace(P_LOW, 1.0 - P_LOW, 200_000),
    0.5 + np.linspace(-BP, BP, 50_000),
]), P_LOW, 1.0 - P_LOW))
vc = uc - 0.5
wc = vc * vc
CENTRAL_POLY = C.Chebyshev.fit(wc, norm.ppf(uc) / vc, CENTRAL_DEG,
                               domain=[0.0, BP * BP]
                               ).convert(kind=np.polynomial.Polynomial).coef

# ---- fit 3: tail polynomial  |z| = Pt(q),  q = sqrt(-2 ln t) ----------------
# fit against the EXACT q (np.log); the kernel's ln_f32 only perturbs q by ~1e-7
ut = np.unique(np.clip(np.concatenate([
    np.logspace(np.log10(FLOOR), np.log10(P_LOW), 200_000),
    np.linspace(FLOOR, P_LOW, 50_000),
]), FLOOR, P_LOW))
qt = np.sqrt(-2.0 * np.log(ut))
zt = -norm.ppf(ut)              # |z| on the tail (ut is the small-side mass)
TAIL_POLY = C.Chebyshev.fit(qt, zt, TAIL_DEG, domain=[qt.min(), qt.max()]
                            ).convert(kind=np.polynomial.Polynomial).coef


# ---- the exact f32 kernel (bit-faithful to the Rust we will paste) ---------
def kernel_f32(u):
    v = (u.astype(np.float32) - f32(0.5)).astype(np.float32)
    a = np.abs(v).astype(np.float32)
    # central
    w = (v * v).astype(np.float32)
    zc = (v * horner_f32(CENTRAL_POLY, w)).astype(np.float32)
    # tail (computed for all lanes; selected below — mirrors a SIMD blend)
    t = np.maximum(f32(0.5) - a, f32(FLOOR)).astype(np.float32)
    q = np.sqrt((f32(-2.0) * ln_f32(t)).astype(np.float32)).astype(np.float32)
    zt_ = np.copysign(horner_f32(TAIL_POLY, q), v).astype(np.float32)
    return np.where(a <= f32(BP), zc, zt_).astype(np.float32)


# ---- grade over the ACTUALLY reachable uniforms u_j = (2j+1)/2^24 ----------
# These are exactly the u32_to_unit_f32 outputs (word>>9 = j in [0,2^23)) and
# each is EXACTLY representable in f32 — so kernel(u_j) vs Phi^-1(u_j) measures
# the kernel's own error with NO u-quantization artifact. (Grading instead at
# Phi^-1(f64 u) over an off-grid sweep folds in u's f32 half-ULP, ~3e-3 at 4.7σ
# from the 1/phi(z) tail slope — that is u-representation, not kernel error, and
# would equally sink any accurate kernel incl. the pure rational.)
JMAX = 2**23
j = np.unique(np.r_[
    np.linspace(0, JMAX - 1, 2_000_000).astype(np.int64),
    np.arange(0, 50_000), np.arange(JMAX - 50_000, JMAX),
    np.logspace(0, np.log10(JMAX - 1), 300_000).astype(np.int64),
    JMAX - 1 - np.logspace(0, np.log10(JMAX - 1), 300_000).astype(np.int64),
])
j = np.clip(np.unique(j), 0, JMAX - 1)
ug = (2.0 * j.astype(np.float64) + 1.0) / 2.0**24      # exact in f32
ztrue = norm.ppf(ug)
zgot = kernel_f32(ug.astype(np.float32)).astype(np.float64)
err = np.abs(zgot - ztrue)
i = int(np.argmax(err))

a_all = np.abs(ug - 0.5)
cmask, tmask = a_all <= BP, a_all > BP
# breakpoint continuity: central vs tail evaluated at the same u just inside/out
ubp = np.array([0.5 + BP - 1e-6, 0.5 + BP + 1e-6])
jump = abs(float(kernel_f32(ubp[:1])[0]) - float(kernel_f32(ubp[1:])[0]))

print(f"# FLOOR=2^-25  BP=|v|>{BP}  degrees log2/central/tail = "
      f"{LOG2_DEG}/{CENTRAL_DEG}/{TAIL_DEG}")
print(f"# max|Δz| overall   = {err.max():.3e}  at u={ug[i]:.3e}  z={ztrue[i]:.3f}")
print(f"# max|Δz| central   = {err[cmask].max():.3e}")
print(f"# max|Δz| tail       = {err[tmask].max():.3e}")
print(f"# breakpoint jump   = {jump:.3e}")
print(f"# max|coeff| central={np.abs(CENTRAL_POLY).max():.1f} "
      f"tail={np.abs(TAIL_POLY).max():.1f} log2={np.abs(LOG2_POLY).max():.3f}")

# ---- pre-freeze tail-mass gate (mirrors the Rust inv_cdf_tail_distribution) -
for z, theo in [(4.0, 6.3342e-5), (4.5, 6.7953e-6)]:
    m = 8_000_000
    kk = np.arange(m, dtype=np.float64)
    uu = FLOOR + (1.0 - 2.0 * FLOOR) * ((kk + 0.5) / m)
    emp = np.mean(np.abs(kernel_f32(uu.astype(np.float32))) >= z)
    band = 0.25 * theo + 5.0 / m
    flag = "ok" if abs(emp - theo) < band else "FAIL"
    print(f"# tail mass {z}σ: emp={emp:.3e} theo={theo:.3e} ({flag})")

ok = np.isfinite(err.max()) and err.max() < 2e-3
print("# OK: f32 kernel holds the 2e-3 data-plane budget — ready to freeze."
      if ok else "# FAIL: over budget — bump a degree (do NOT widen the gate).")
print()


def emit(name, arr):
    body = ", ".join(f"{f32(c):+.9e}f32" for c in arr)
    print(f"const {name}: [f32; {len(arr)}] = [{body}];")


print(f"const NORM_INV_BP: f32 = {f32(BP):+.9e}f32;       // |v| breakpoint")
print(f"const NORM_INV_FLOOR: f32 = {f32(FLOOR):+.9e}f32;  // 2^-25 tail clamp")
print(f"const NORM_INV_LN2: f32 = {LN2:+.9e}f32;")
emit("NORM_INV_LOG2", LOG2_POLY)     # log2(m) on m in [1,2), ascending in m
emit("NORM_INV_CENTRAL", CENTRAL_POLY)  # Pc, ascending in w = v^2
emit("NORM_INV_TAIL", TAIL_POLY)     # Pt, ascending in q = sqrt(-2 ln t)
