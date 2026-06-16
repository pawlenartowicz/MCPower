//! Row-stable Philox streams: a sequential per-sim stream plus class-addressed
//! planar sub-streams.
//!
//! For fixed `(base_seed, sim_id)`, every draw is deterministic and independent
//! of `max_n`. Prefix `X_full[:N]` is bit-identical across runs with different
//! `max_n` — see the row-stability test below. Two addressing schemes share the
//! per-sim key: the sequential `SimRng` stream (counter word 2 = 0) serves the
//! scattered draws (cluster REs, categorical, bootstrap row-picks), where fixed
//! per-row consumption gives prefix stability behaviorally; the hot draw
//! classes (`CLASS_XNORM` continuous-X normals, `CLASS_RESID` residuals) are
//! counter-addressed by `(key, class, column, row)` — prefix-stable
//! structurally, and contiguous counter runs are exactly the shape the blocked
//! word fill + SIMD inverse-CDF (`fill_words` / `fill_normal_column`) need.

use crate::philox::philox4x32_10;

/// David Stafford's "Mix13" SplitMix64 finalizer — the same avalanche function
/// used by Java's SplittableRandom. Callers supply their own pre-mix strategy
/// for combining the seed inputs (xor+rotate for sim streams, Weyl add for
/// scenario seeds).
#[inline]
pub fn splitmix64_finalize(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Mix `(base_seed, sim_id)` into a 64-bit Pcg64 stream id.
///
/// Pre-mixes with `sim_id.rotate_left(32)` so adjacent `sim_id` values produce
/// non-adjacent stream seeds, then runs the SplitMix64 finalizer.
pub fn pcg_mix64(base_seed: u64, sim_id: u64) -> u64 {
    splitmix64_finalize(base_seed ^ sim_id.rotate_left(32))
}

/// Per-sim RNG over Philox4x32-10. Sequential stream behind a fixed-consumption
/// draw API: for fixed `(base_seed, sim_id)` the stream is deterministic and
/// independent of `max_n`, so `X_full[:N]` is prefix-stable across `N` (the
/// CRN / curve-quality invariant — see `rng_rows_stable_across_max_n`).
/// This stream serves the scattered/cold draws (cluster REs, categorical,
/// bootstrap row-picks): with the 1-uniform inverse-CDF normal + 1-uniform
/// categorical (no rejection) per-row consumption is fixed for a given spec,
/// which is all prefix-stability needs. The hot block-shaped draws (continuous
/// X, residuals) instead use the class-addressed planar fills below, keyed by
/// `key()` — disjoint from this stream by counter word 2 ≥ 1.
#[derive(Debug, Clone)]
pub struct SimRng {
    key: [u32; 2],
    counter: u64,
    buf: [u32; 4],
    buf_pos: usize, // 0..=4; 4 = exhausted
}

impl SimRng {
    /// Construct from a base seed and sim id. The 64-bit mixed seed becomes the
    /// two Philox key words; the counter starts at 0.
    pub fn new(base_seed: u64, sim_id: u64) -> Self {
        let k = pcg_mix64(base_seed, sim_id);
        Self { key: [k as u32, (k >> 32) as u32], counter: 0, buf: [0; 4], buf_pos: 4 }
    }

    /// Per-sim Philox key — read by the planar batched fills
    /// (`fill_normal_column` / `fill_uniform_column`), which address draws by
    /// (class, column, row) instead of consuming this sequential stream.
    #[inline]
    pub fn key(&self) -> [u32; 2] {
        self.key
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.buf_pos == 4 {
            let c = self.counter;
            self.buf = philox4x32_10([c as u32, (c >> 32) as u32, 0, 0], self.key);
            self.counter = self.counter.wrapping_add(1);
            self.buf_pos = 0;
        }
        let w = self.buf[self.buf_pos];
        self.buf_pos += 1;
        w
    }

    /// Uniform on the open interval (0,1), f32.
    #[inline]
    pub fn next_uniform(&mut self) -> f32 {
        u32_to_unit_f32(self.next_u32())
    }

    /// Standard normal (mean 0, SD 1), f32 — one uniform, inverse-CDF, no cache.
    ///
    /// Part of the reproducibility contract (golden-pinned in
    /// `tests/golden_rng.rs`): the contract is the sequence of `next_normal()`
    /// values. Changing the kernel or the per-draw structure moves results in
    /// every port and requires a coordinated version bump.
    #[inline]
    pub fn next_normal(&mut self) -> f32 {
        norm_inv_cdf_f32(self.next_uniform())
    }

    /// Categorical draw from a CDF-inverse over `probs` (f64 spec values). The
    /// draw is one f32 uniform widened to f64 for the cumulative walk. Final
    /// bucket absorbs any remainder; a zero-prob bucket is never returned.
    pub fn next_categorical(&mut self, probs: &[f64]) -> usize {
        debug_assert!(!probs.is_empty(), "next_categorical: empty probs");
        let u = self.next_uniform() as f64;
        let mut acc = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if u < acc {
                return i;
            }
        }
        probs.len() - 1
    }

}

// ===== Planar class-addressed streams =======================================
// The sequential SimRng stream above runs Philox with counter word 2 = 0.
// Planar classes set word 2 = class ≥ 1, so they are disjoint from the
// sequential stream and from each other for every counter position. Within a
// class, the counter is [row >> 2, column, class, 0] with lane row & 3: each
// (class, column) is a contiguous counter run (block-generatable), and a
// draw's value depends only on (key, class, column, row) — prefix-stable in
// max_n and independent of n_clusters by construction.

/// Continuous-predictor normals; column = predictor index j (0-based,
/// non-factor layout).
pub const CLASS_XNORM: u32 = 1;
/// Residual draws; column = slot. Slot layout is owned by data_gen's residual
/// pass (Normal/Binary: slot 0 only; T/HighKurtosis: slot 0 = z, χ² normals in
/// slots 1..; Right/LeftSkewed: χ² normals in slots 0.. — no z slot).
pub const CLASS_RESID: u32 = 2;

/// Fill `out[i]` with the Philox word at (class, col, row = i): word
/// `philox4x32_10([i >> 2, col, class, 0], key)[i & 3]`. Counters are
/// generated four at a time — the 10-round chain on one counter is serial,
/// but counters are independent, so the unroll feeds the multiplier pipeline
/// (the F/G latency lesson applied to integer code).
pub fn fill_words(key: [u32; 2], class: u32, col: u32, out: &mut [u32]) {
    debug_assert!(class != 0, "class 0 is the sequential SimRng stream");
    let n = out.len();
    let mut i = 0usize;
    // Counter/lane indices fit u32 with huge margin: rows are capped by the
    // shared upload limit (configs/config.json max_rows = 100_000), so
    // ctr = row >> 2 and the tail_idx row casts stay far below u32::MAX.
    let mut ctr = 0u32;
    while i + 16 <= n {
        for u in 0..4u32 {
            let w = philox4x32_10([ctr + u, col, class, 0], key);
            let base = i + 4 * u as usize;
            out[base..base + 4].copy_from_slice(&w);
        }
        ctr += 4;
        i += 16;
    }
    while i < n {
        let w = philox4x32_10([ctr, col, class, 0], key);
        let take = (n - i).min(4);
        out[i..i + take].copy_from_slice(&w[..take]);
        ctr += 1;
        i += take;
    }
}

/// Batched `next_uniform` over a (class, col) word column: `out[i]` is
/// bit-identical to `u32_to_unit_f32(word(class, col, i))`.
/// Precondition: `words.len() >= out.len()`.
pub fn fill_uniform_column(key: [u32; 2], class: u32, col: u32, words: &mut [u32], out: &mut [f32]) {
    debug_assert!(words.len() >= out.len(), "words scratch shorter than out");
    let n = out.len();
    let words = &mut words[..n];
    fill_words(key, class, col, words);
    for i in 0..n {
        out[i] = u32_to_unit_f32(words[i]);
    }
}

/// In-place central-branch inverse-CDF: buf[i] = v·Pc(v²) for v = buf[i].
/// Mirrors `horner_f32` op-for-op — plain mul/add, NO mul_add (the no-fma
/// reproducibility rule), so SIMD lanes are bit-identical to the scalar
/// kernel. Change together with `horner_f32`/`NORM_INV_CENTRAL`.
struct CentralHornerOp<'a> {
    buf: &'a mut [f32],
}
impl pulp::WithSimd for CentralHornerOp<'_> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) {
        let last = NORM_INV_CENTRAL.len() - 1;
        let (head, tail) = S::as_mut_simd_f32s(self.buf);
        for x in head {
            let v = *x;
            let r = simd.mul_f32s(v, v);
            let mut acc = simd.splat_f32s(NORM_INV_CENTRAL[last]);
            let mut k = last;
            while k > 0 {
                k -= 1;
                acc = simd.add_f32s(simd.mul_f32s(acc, r), simd.splat_f32s(NORM_INV_CENTRAL[k]));
            }
            *x = simd.mul_f32s(v, acc);
        }
        for x in tail {
            let v = *x;
            *x = v * horner_f32(&NORM_INV_CENTRAL, v * v);
        }
    }
}

/// Batched `next_normal` over a (class, col) word column: `out[i]` is
/// bit-identical to `norm_inv_cdf_f32(u32_to_unit_f32(word(class, col, i)))`.
/// Central branch (~95.15% of draws) runs as one pulp-dispatched pass over the
/// whole column; the |v| > BP tail lanes are collected during the word→v
/// conversion and recomputed scalar afterwards (their central-poly results are
/// overwritten). Precondition: `words.len() >= out.len()`.
pub fn fill_normal_column(
    key: [u32; 2],
    class: u32,
    col: u32,
    words: &mut [u32],
    tail_idx: &mut Vec<u32>,
    out: &mut [f32],
) {
    debug_assert!(words.len() >= out.len(), "words scratch shorter than out");
    let n = out.len();
    let words = &mut words[..n];
    fill_words(key, class, col, words);
    tail_idx.clear();
    for i in 0..n {
        let v = u32_to_unit_f32(words[i]) - 0.5;
        out[i] = v;
        if v.abs() > NORM_INV_BP {
            tail_idx.push(i as u32);
        }
    }
    pulp::Arch::new().dispatch(CentralHornerOp { buf: out });
    for &i in tail_idx.iter() {
        let i = i as usize;
        out[i] = norm_inv_cdf_f32(u32_to_unit_f32(words[i]));
    }
}

/// Map a Philox 32-bit word to an f32 uniform on the OPEN interval (0,1).
/// 23-bit mantissa (`word >> 9`) centred by +0.5 so the result is never 0 or 1
/// — Φ⁻¹ never sees a saturating argument. Floor ≈ 2⁻²⁴, cap ≈ 1−2⁻²⁴; this
/// construction pins the tail reach (see the L1 reach test).
///
/// 23 bits rather than 24: in f32, values ≥ 2²³ = 8_388_608 have ULP ≥ 1.0,
/// so `(max_24bit as f32) + 0.5` rounds to 2²⁴ and the result hits 1.0 exactly.
/// In the range [2²², 2²³) the ULP is 0.5, so every half-integer is exactly
/// representable and the +0.5 centering is guaranteed to stay below 1.0.
#[inline]
pub fn u32_to_unit_f32(word: u32) -> f32 {
    ((word >> 9) as f32 + 0.5) * (1.0 / 8_388_608.0) // (x+0.5) · 2⁻²³
}

// ===== Frozen f32 inverse-CDF normal kernel ================================
// Fitted + frozen by `scripts/fit_norm_inv_cdf.py` (scipy `norm.ppf`, graded on
// the exactly-reachable f32 grid). Validated f32 max|Δz| = 3.0e-4 over the
// reachable range (gate 2e-3); tail mass at 4.0/4.5σ within band. FROZEN:
// changing any coefficient is a golden-breaking, version-bumping event
// (`tests/golden_rng.rs` / `data_gen.rs` goldens pin to these values).
//
// Form — the sanctioned one-`ln` fallback (plan B3 checkpoint, user-approved).
// The spec's single branchless rational (no `ln`) is f32-INFEASIBLE: Φ⁻¹'s slope
// at the 2⁻²⁵ floor (~3e6) forces a denominator pole at p ≈ 1+3e-7 whose monomial
// coefficients (~(3e6)ᵏ) overflow f32 — measured: a near-minimax rational reaches
// 5e-4 in f64 but diverges to >5 once stored/evaluated in f32. Folding at Acklam's
// breakpoint so no single polynomial spans the steep tail sidesteps that:
//   central |v| ≤ BP : z = v · Pc(v²)               — plain poly, ~95% of draws
//   tail    |v| > BP : z = copysign(Pt(q), v),  q = √(−2·ln_f32(t)),
//                      t = max(0.5 − |v|, FLOOR)
// `ln_f32` is a pure-f32 bit-trick log (exponent + degree-5 mantissa log2 poly),
// NOT libm `ln`: only the ~4.85% tail touches it, so the `__ieee754_log` slice
// this break targets stays gone and generation is cross-host bit-identical (F3).
// Every op is plain f32 mul/add (NO fma) so the sequence reproduces across hosts.
const NORM_INV_BP: f32 = 4.757499993e-01_f32; // |v| breakpoint (Acklam p_low = 0.02425)
const NORM_INV_FLOOR: f32 = 2.980232239e-08_f32; // 2⁻²⁵: tail-arg clamp keeps the kernel total at u→{0,1}
const NORM_INV_LN2: f32 = 6.931471825e-01_f32;
const NORM_INV_LOG2: [f32; 6] = [
    -2.786813021e+00_f32, 5.046875954e+00_f32, -3.492494345e+00_f32,
    1.593901396e+00_f32, -4.048671722e-01_f32, 4.342890903e-02_f32,
];
const NORM_INV_CENTRAL: [f32; 11] = [
    2.506664753e+00_f32, 2.586458445e+00_f32, 1.238584900e+01_f32,
    -4.216123047e+02_f32, 1.463396387e+04_f32, -2.779154375e+05_f32,
    3.215264750e+06_f32, -2.294391600e+07_f32, 9.886485600e+07_f32,
    -2.359132160e+08_f32, 2.401336800e+08_f32,
];
const NORM_INV_TAIL: [f32; 10] = [
    -2.150734663e+00_f32, 2.428994656e+00_f32, -7.639000416e-01_f32,
    2.937270701e-01_f32, -8.018484712e-02_f32, 1.541402005e-02_f32,
    -2.041375730e-03_f32, 1.774382981e-04_f32, -9.117987247e-06_f32,
    2.100489240e-07_f32,
];

/// Ascending-monomial Horner in f32 (plain mul/add, no `fma` — reproducible
/// across hosts). `coeffs[0]` is the constant term, `coeffs[last]` the highest.
#[inline]
fn horner_f32(coeffs: &[f32], x: f32) -> f32 {
    let mut acc = coeffs[coeffs.len() - 1];
    for &c in coeffs.iter().rev().skip(1) {
        acc = acc * x + c;
    }
    acc
}

/// Natural log for `t > 0` in pure f32: decompose `t = m·2ᵉ` (m ∈ [1,2)) via the
/// IEEE bits, then `ln = (e + log2_poly(m))·ln2`. The bit ops are exact; only the
/// degree-5 mantissa poly approximates (~1e-6 in log2, negligible after the
/// √(−2··) downstream). Tail-path only. NOT libm `ln` — see the kernel header.
#[inline]
fn ln_f32(t: f32) -> f32 {
    let bits = t.to_bits();
    let e = (((bits >> 23) & 0xff) as i32 - 127) as f32;
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    (e + horner_f32(&NORM_INV_LOG2, m)) * NORM_INV_LN2
}

/// Standard-normal quantile for a uniform `u ∈ (0,1)`, the f32 data-plane kernel.
/// One uniform → one normal (no pair cache). Central region is a branchless
/// polynomial; the tail (`|v| > BP`, ~4.85% of draws) adds one pure-f32 `ln`.
#[inline]
pub fn norm_inv_cdf_f32(u: f32) -> f32 {
    let v = u - 0.5;
    let a = v.abs();
    if a <= NORM_INV_BP {
        v * horner_f32(&NORM_INV_CENTRAL, v * v)
    } else {
        let t = (0.5 - a).max(NORM_INV_FLOOR);
        let q = (-2.0 * ln_f32(t)).sqrt();
        horner_f32(&NORM_INV_TAIL, q).copysign(v)
    }
}

#[cfg(test)]
pub(crate) mod oracle {
    /// High-precision standard-normal quantile Φ⁻¹(p), p ∈ (0,1). Newton
    /// iteration on Φ(z)=p with Φ via libm erfc and φ the Gaussian pdf.
    /// ~1e-12 over the reachable range — the truth the f32 kernel is graded on.
    pub fn phi_inv_ref(p: f64) -> f64 {
        assert!(p > 0.0 && p < 1.0);
        // Acklam 3-region rational seed — accurate to ~1e-9 before any Newton
        // refinement. A simple tail formula like sqrt(-2*ln(1-p)) seeds from 2.72
        // for p=0.975 and Newton oscillates for 4 iterations without converging.
        let mut z = acklam_seed(p);
        // Newton refine: f(z)=Φ(z)-p, f'(z)=φ(z).
        for _ in 0..4 {
            let cdf = 0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2);
            let pdf = (-(0.5) * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
            z -= (cdf - p) / pdf;
        }
        z
    }

    /// Acklam's rational approximation to Φ⁻¹ — 3 regions, ~1e-9 accuracy.
    /// Seeds Newton in `phi_inv_ref`; not exported as truth itself.
    fn acklam_seed(p: f64) -> f64 {
        // Coefficients from Peter Acklam's public-domain approximation.
        const A: [f64; 6] = [
            -3.969683028665376e+01,  2.209460984245205e+02,
            -2.759285104469687e+02,  1.383577518672690e+02,
            -3.066479806614716e+01,  2.506628277459239e+00,
        ];
        const B: [f64; 5] = [
            -5.447609879822406e+01,  1.615858368580409e+02,
            -1.556989798598866e+02,  6.680131188771972e+01,
            -1.328068155288572e+01,
        ];
        const C: [f64; 6] = [
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
             4.374664141464968e+00,  2.938163982698783e+00,
        ];
        const D: [f64; 4] = [
             7.784695709041462e-03,  3.223671334942036e-01,
             2.445134137142996e+00,  3.754408661907416e+00,
        ];
        let p_lo = 0.02425_f64;
        let p_hi = 1.0 - p_lo;
        if p < p_lo {
            let q = (-2.0 * p.ln()).sqrt();
            (((((C[0]*q+C[1])*q+C[2])*q+C[3])*q+C[4])*q+C[5])
                / ((((D[0]*q+D[1])*q+D[2])*q+D[3])*q+1.0)
        } else if p <= p_hi {
            let q = p - 0.5;
            let r = q * q;
            (((((A[0]*r+A[1])*r+A[2])*r+A[3])*r+A[4])*r+A[5])*q
                / (((((B[0]*r+B[1])*r+B[2])*r+B[3])*r+B[4])*r+1.0)
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            -((((( C[0]*q+C[1])*q+C[2])*q+C[3])*q+C[4])*q+C[5])
                / ((((D[0]*q+D[1])*q+D[2])*q+D[3])*q+1.0)
        }
    }

    #[test]
    fn oracle_matches_known_quantiles() {
        // Φ⁻¹(0.975)=1.959963985, Φ⁻¹(0.95)=1.644853627, Φ⁻¹(0.5)=0.
        assert!((phi_inv_ref(0.975) - 1.959_963_985).abs() < 1e-9);
        assert!((phi_inv_ref(0.95) - 1.644_853_627).abs() < 1e-9);
        assert!(phi_inv_ref(0.5).abs() < 1e-12);
        // Symmetry.
        for &p in &[0.01_f64, 0.2, 0.4, 0.49] {
            assert!((phi_inv_ref(p) + phi_inv_ref(1.0 - p)).abs() < 1e-9);
        }
    }
}

#[cfg(test)]
mod uniform_tests {
    use super::*;
    #[test]
    fn unit_f32_in_open_interval() {
        // Extremes of the 23-bit field map strictly inside (0,1).
        assert!(u32_to_unit_f32(0) > 0.0);
        assert!(u32_to_unit_f32(u32::MAX) < 1.0);
        assert!(u32_to_unit_f32(0) < u32_to_unit_f32(u32::MAX));
        // A dense scan stays in range.
        for k in 0..10_000u32 {
            let w = k.wrapping_mul(0x9E37_79B9);
            let u = u32_to_unit_f32(w);
            assert!(u > 0.0 && u < 1.0, "u={u} out of (0,1)");
        }
    }
}

#[cfg(test)]
mod inv_cdf_tests {
    use super::oracle::phi_inv_ref;
    use super::*;

    /// Max |z_f32 − Φ⁻¹| over the reachable range — the kernel's own error must
    /// hold the data-plane budget (gate 2e-3 in z keeps power well under 0.1pp
    /// common; the edge cells are proven by the Phase E null-calibration grid).
    ///
    /// Graded on the EXACTLY reachable uniforms `u32_to_unit_f32` emits — `u_j =
    /// (2j+1)/2²⁴`, `j ∈ [0, 2²³)`, each exactly representable in f32 — so this
    /// isolates kernel error. (Grading against Φ⁻¹ of a denser off-grid f64 sweep
    /// would fold in `u`'s f32 half-ULP, ≈3e-3 at 4.7σ from the `1/φ(z)` tail
    /// slope: that is `u`-representation, not kernel error, and would equally sink
    /// any accurate kernel — including the pure rational the spec first aimed at.)
    #[test]
    fn inv_cdf_max_error_over_reachable_range() {
        let mut max_err = 0.0_f64;
        for j in 0..(1u32 << 23) {
            let u = u32_to_unit_f32(j << 9); // word>>9 == j ⇒ enumerates every reachable u
            let err = (norm_inv_cdf_f32(u) as f64 - phi_inv_ref(u as f64)).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 2e-3, "inverse-CDF max |Δz| = {max_err:.3e}");
    }

    /// The construction's floor pins the tail reach: the most extreme reachable
    /// |z| sits near 5σ (edge regime). Pin it to the actual construction.
    #[test]
    fn inv_cdf_tail_reach() {
        let zmax = norm_inv_cdf_f32(u32_to_unit_f32(u32::MAX));
        let zmin = norm_inv_cdf_f32(u32_to_unit_f32(0));
        assert!((5.0..6.0).contains(&zmax.abs()), "zmax={zmax}");
        assert!((5.0..6.0).contains(&zmin.abs()), "zmin={zmin}");
        assert!(zmin < 0.0 && zmax > 0.0);
    }

    /// Pre-freeze tail-distribution gate (engine-free): the kernel's own output
    /// tail must track the normal tail near where α=5×10⁻⁷ lives, BEFORE any
    /// golden is pinned to these coefficients. A uniform sweep of `u` maps through
    /// the kernel; the fraction landing past ±zσ equals the true two-sided tail
    /// mass iff the kernel places the tail quantile correctly. Catches tail
    /// distortion the global max|Δz| averages away — the cheap early tripwire for
    /// the otherwise-expensive post-freeze E1 rollback.
    #[test]
    fn inv_cdf_tail_distribution_matches_normal() {
        // True P(|Z| ≥ z) for z = 4.0, 4.5 (standard normal) — no oracle needed.
        for &(z, theo) in &[(4.0_f64, 6.3342e-5_f64), (4.5, 6.7953e-6)] {
            let n = 8_000_000usize;
            let floor = 0.5_f64 * 2f64.powi(-24);
            let mut exceed = 0usize;
            for k in 0..n {
                let u = floor + (1.0 - 2.0 * floor) * ((k as f64 + 0.5) / n as f64);
                if (norm_inv_cdf_f32(u as f32) as f64).abs() >= z {
                    exceed += 1;
                }
            }
            let emp = exceed as f64 / n as f64;
            // Catastrophe tripwire, not the sub-1pp proof (that is E1/E2): a
            // generous 25%-of-mass band catches gross tail mis-placement while
            // tolerating the sweep's ~1/n quantisation.
            assert!(
                (emp - theo).abs() < 0.25 * theo + 5.0 / n as f64,
                "tail mass off at {z}σ: emp={emp:.3e} theo={theo:.3e}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Guards CRN / curve quality, not just determinism: row-stability is what
    // makes `find_sample_size`'s grid datasets nested prefixes (X_full[:N]
    // identical across N), which keeps the power-vs-N curve smooth and the
    // crossing stable. A refactor that breaks this breaks `find_sample_size`,
    // not "a determinism dup".
    #[test]
    fn rng_rows_stable_across_max_n() {
        let mut rng_full = SimRng::new(42, 7);
        let full: Vec<f32> = (0..1000).map(|_| rng_full.next_normal()).collect();

        let mut rng_short = SimRng::new(42, 7);
        let short: Vec<f32> = (0..200).map(|_| rng_short.next_normal()).collect();

        assert_eq!(&full[..200], &short[..]);
    }

    #[test]
    fn rng_uniform_in_range() {
        let mut rng = SimRng::new(1, 1);
        for _ in 0..10_000 {
            let u = rng.next_uniform();
            assert!((0.0..1.0).contains(&u), "u out of range: {u}");
        }
    }

    #[test]
    fn rng_different_sim_ids_diverge() {
        let mut a = SimRng::new(42, 0);
        let mut b = SimRng::new(42, 1);
        let mut diff = 0usize;
        for _ in 0..100 {
            if a.next_normal() != b.next_normal() {
                diff += 1;
            }
        }
        assert!(diff > 90, "streams from different sim_ids should diverge");
    }

    #[test]
    fn pcg_mix_distinct_seeds() {
        let a = pcg_mix64(0, 0);
        let b = pcg_mix64(0, 1);
        let c = pcg_mix64(1, 0);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn fill_words_matches_per_counter_philox() {
        use crate::philox::philox4x32_10;
        let key = [0xdead_beef, 0x1234_5678];
        let mut out = vec![0u32; 1003]; // non-multiple of 16 and 4 → both tails
        fill_words(key, CLASS_XNORM, 7, &mut out);
        for (i, &w) in out.iter().enumerate() {
            let kat = philox4x32_10([(i / 4) as u32, 7, CLASS_XNORM, 0], key);
            assert_eq!(w, kat[i % 4], "word {i}");
        }
    }

    #[test]
    fn fill_words_prefix_stable_and_column_disjoint() {
        let key = [42, 7];
        let mut full = vec![0u32; 1000];
        let mut short = vec![0u32; 237];
        fill_words(key, CLASS_RESID, 3, &mut full);
        fill_words(key, CLASS_RESID, 3, &mut short);
        assert_eq!(&full[..237], &short[..]);
        // A different column is a different stream.
        let mut other = vec![0u32; 237];
        fill_words(key, CLASS_RESID, 4, &mut other);
        assert_ne!(&short[..], &other[..]);
    }

    #[test]
    fn fill_normal_column_bit_matches_scalar_kernel() {
        let key = [0x9e37_79b9, 0xbb67_ae85];
        let n = 4099usize; // odd → exercises the SIMD sub-lane tail too
        let mut words = vec![0u32; n];
        let mut tails = Vec::new();
        let mut out = vec![0.0f32; n];
        fill_normal_column(key, CLASS_XNORM, 0, &mut words, &mut tails, &mut out);
        for i in 0..n {
            let expect = norm_inv_cdf_f32(u32_to_unit_f32(words[i]));
            assert_eq!(out[i].to_bits(), expect.to_bits(), "draw {i}");
        }
        // Both branches must actually be exercised (~4.85% expected tail).
        assert!(!tails.is_empty() && tails.len() < n / 10, "tail lanes = {}", tails.len());
    }

    #[test]
    fn fill_uniform_column_bit_matches_scalar() {
        let key = [1, 2];
        let n = 513usize;
        let mut words = vec![0u32; n];
        let mut out = vec![0.0f32; n];
        fill_uniform_column(key, CLASS_RESID, 0, &mut words, &mut out);
        for i in 0..n {
            assert_eq!(out[i].to_bits(), u32_to_unit_f32(words[i]).to_bits(), "draw {i}");
        }
    }

    #[test]
    fn fill_columns_small_and_empty() {
        let key = [3, 9];
        for n in [0usize, 1, 3, 15] {
            let mut words = vec![0u32; n.max(1)];
            let mut tails = Vec::new();
            let mut out = vec![0.0f32; n];
            fill_normal_column(key, CLASS_XNORM, 1, &mut words, &mut tails, &mut out);
            for i in 0..n {
                let expect = norm_inv_cdf_f32(u32_to_unit_f32(words[i]));
                assert_eq!(out[i].to_bits(), expect.to_bits(), "n={n} draw {i}");
            }
            let mut uout = vec![0.0f32; n];
            fill_uniform_column(key, CLASS_RESID, 1, &mut words, &mut uout);
            for i in 0..n {
                assert_eq!(uout[i].to_bits(), u32_to_unit_f32(words[i]).to_bits(), "n={n} draw {i}");
            }
        }
    }

    #[test]
    fn rng_categorical_routes_by_cdf_and_stays_in_range() {
        // Deterministic CDF-inverse properties (no sampling band). A point mass on
        // a single bucket must always return that bucket regardless of the draw; a
        // zero-probability bucket is never returned; and over a many-draw mixed
        // distribution every returned index is in range. These catch a broken
        // cumsum/threshold; the realized bucket *frequencies* are statistical (L3).
        let mut rng = SimRng::new(123, 1);

        // Point mass on bucket 1 → always returns 1, never the zero-prob buckets.
        for _ in 0..1_000 {
            assert_eq!(rng.next_categorical(&[0.0, 1.0, 0.0]), 1);
        }
        // Point mass on the last bucket.
        for _ in 0..1_000 {
            assert_eq!(rng.next_categorical(&[0.0, 0.0, 1.0]), 2);
        }
        // Mixed distribution: every index is a valid bucket.
        let probs = [0.25, 0.5, 0.25];
        for _ in 0..1_000 {
            let i = rng.next_categorical(&probs);
            assert!(i < probs.len(), "categorical index {i} out of range");
        }
    }
}
