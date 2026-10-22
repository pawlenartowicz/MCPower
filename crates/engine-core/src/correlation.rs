//! Correlation utility — in-place Cholesky and PSD repair via spectral clip.
//!
//! Per-sim use: scenarios perturb the correlation matrix each sim, so the
//! Cholesky factor cannot be cached at engine init. `factor_only` handles the
//! optimistic path (input is PSD); `psd_repair_and_factor` falls back to EVD
//! spectral clipping when LLT fails.

use crate::spec::EngineError;
use faer::reborrow::Reborrow;
use faer::{Mat, MatMut, MatRef, Side};

/// Eigenvalue floor for the spectral-repair path. Clipping negative eigenvalues
/// to exactly 0 leaves the reconstructed matrix on the PSD boundary (singular);
/// the subsequent renormalize/force-unit-diagonal round-off then tips the
/// smallest eigenvalue a hair negative (~1e-16), and faer's zero-tolerance
/// Cholesky rejects it. Flooring at a small positive epsilon keeps the repaired
/// matrix strictly positive-definite so the final factorization always succeeds.
/// 1e-10 is far above the round-off and far below any meaningful correlation
/// eigenvalue, so the perturbed matrix shifts negligibly.
const PSD_EIGENVALUE_FLOOR: f64 = 1e-10;

/// Cholesky in place, optimistic path.
///
/// Reads the lower triangle of `in_out`, writes `L` (lower-triangular Cholesky
/// factor) into the lower triangle of `in_out`. The upper triangle is set to
/// zero so the result is a clean lower-triangular factor.
///
/// Returns `CorrelationNotPSD` if any pivot is non-positive (faer's LLT
/// reports `CholeskyError` in that case).
pub fn factor_only(mut in_out: MatMut<'_, f64>) -> Result<(), EngineError> {
    let n = in_out.nrows();
    assert_eq!(n, in_out.ncols(), "factor_only: input must be square");
    if n == 0 {
        return Ok(());
    }

    // Use faer's high-level Cholesky API (returns a freshly-allocated factor).
    // We copy back into `in_out` to keep the in-place signature.
    let chol = in_out
        .rb()
        .llt(Side::Lower)
        .map_err(|_| EngineError::CorrelationNotPSD)?;
    let l = chol.L();

    for j in 0..n {
        for i in 0..n {
            let v = if i >= j { l[(i, j)] } else { 0.0 };
            in_out[(i, j)] = v;
        }
    }
    Ok(())
}

/// Cholesky with PSD repair fallback.
///
/// Fast path: try `factor_only`. On `CorrelationNotPSD`, run spectral repair:
///   1. EVD of the symmetric input (lower triangle).
///   2. Floor eigenvalues at `PSD_EIGENVALUE_FLOOR` (a small positive epsilon,
///      not exactly 0) so the result is strictly positive-definite.
///   3. Reconstruct `A' = U · diag(λ_floored) · Uᵀ`.
///   4. Renormalize to unit diagonal: `A' / sqrt(diag(A') · diag(A')ᵀ)`.
///   5. Force diagonal to 1.0 exactly.
///   6. Cholesky-factor in place.
pub fn psd_repair_and_factor(mut in_out: MatMut<'_, f64>) -> Result<(), EngineError> {
    let n = in_out.nrows();
    assert_eq!(
        n,
        in_out.ncols(),
        "psd_repair_and_factor: input must be square"
    );
    if n == 0 {
        return Ok(());
    }

    // Fast path: try direct Cholesky on the lower triangle.
    let sym: Mat<f64> = match in_out.rb().llt(Side::Lower) {
        Ok(chol) => {
            let l = chol.L();
            for j in 0..n {
                for i in 0..n {
                    let v = if i >= j { l[(i, j)] } else { 0.0 };
                    in_out[(i, j)] = v;
                }
            }
            return Ok(());
        }
        Err(_) => {
            // Symmetric copy from the lower triangle (cholesky only reads Lower).
            let mut sym = Mat::<f64>::zeros(n, n);
            for j in 0..n {
                for i in 0..n {
                    let v = if i >= j {
                        in_out[(i, j)]
                    } else {
                        in_out[(j, i)]
                    };
                    sym[(i, j)] = v;
                }
            }
            sym
        }
    };

    // Eigendecomposition (lower triangle accessed).
    let evd = sym
        .as_ref()
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| EngineError::CorrelationNotPSD)?;
    let u: MatRef<'_, f64> = evd.U();
    let s_col = evd.S().column_vector();

    // λ_floored[k] = max(s_k, PSD_EIGENVALUE_FLOOR) — strictly positive so the
    // reconstructed matrix survives the final (zero-tolerance) Cholesky.
    let mut lambda = Vec::with_capacity(n);
    for k in 0..n {
        let lam: f64 = s_col[k];
        lambda.push(lam.max(PSD_EIGENVALUE_FLOOR));
    }

    // Reconstruct A' = U · diag(λ_floored) · Uᵀ.
    let mut repaired = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += u[(i, k)] * lambda[k] * u[(j, k)];
            }
            repaired[(i, j)] = acc;
        }
    }

    // Renormalize to unit diagonal.
    let mut d = vec![0.0_f64; n];
    for i in 0..n {
        let v = repaired[(i, i)];
        d[i] = if v > 0.0 { v.sqrt() } else { 1.0 };
    }
    for j in 0..n {
        for i in 0..n {
            repaired[(i, j)] /= d[i] * d[j];
        }
    }
    for i in 0..n {
        repaired[(i, i)] = 1.0;
    }

    // Write back into in_out and factor.
    for j in 0..n {
        for i in 0..n {
            in_out[(i, j)] = repaired[(i, j)];
        }
    }

    let chol = in_out
        .rb()
        .llt(Side::Lower)
        .map_err(|_| EngineError::CorrelationNotPSD)?;
    let l = chol.L();
    for j in 0..n {
        for i in 0..n {
            let v = if i >= j { l[(i, j)] } else { 0.0 };
            in_out[(i, j)] = v;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn mat_from_rows(n: usize, rows: &[&[f64]]) -> Mat<f64> {
        let mut m = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = rows[i][j];
            }
        }
        m
    }

    fn approx_eq(a: f64, b: f64, atol: f64) -> bool {
        (a - b).abs() < atol
    }

    #[test]
    fn factor_only_psd_3x3() {
        // A correlation matrix with corr = 0.3 between adjacent variables.
        let original = mat_from_rows(3, &[&[1.0, 0.3, 0.2], &[0.3, 1.0, 0.4], &[0.2, 0.4, 1.0]]);
        let mut m = original.clone();
        factor_only(m.as_mut()).expect("PSD should succeed");

        // Reconstruct L · Lᵀ and compare to original.
        let mut recon = Mat::<f64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += m[(i, k)] * m[(j, k)];
                }
                recon[(i, j)] = acc;
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(recon[(i, j)], original[(i, j)], 1e-12),
                    "({i},{j}): recon={} orig={}",
                    recon[(i, j)],
                    original[(i, j)]
                );
            }
        }
        // Upper triangle is zero.
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert_eq!(m[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn factor_only_identity() {
        let mut m = Mat::<f64>::identity(4, 4);
        factor_only(m.as_mut()).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(m[(i, j)], expected, "({i},{j})");
            }
        }
    }

    #[test]
    fn psd_repair_fast_path_identical_to_factor_only() {
        let original = mat_from_rows(3, &[&[1.0, 0.3, 0.2], &[0.3, 1.0, 0.4], &[0.2, 0.4, 1.0]]);
        let mut m1 = original.clone();
        let mut m2 = original.clone();
        factor_only(m1.as_mut()).unwrap();
        psd_repair_and_factor(m2.as_mut()).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(m1[(i, j)], m2[(i, j)], "({i},{j})");
            }
        }
    }

    #[test]
    fn factor_only_rejects_non_psd() {
        // A symmetric but indefinite 3x3 (one negative eigenvalue).
        let original = mat_from_rows(3, &[&[1.0, 0.9, 0.9], &[0.9, 1.0, -0.9], &[0.9, -0.9, 1.0]]);
        let mut m = original.clone();
        let err = factor_only(m.as_mut()).unwrap_err();
        assert!(matches!(err, EngineError::CorrelationNotPSD));
    }

    #[test]
    fn psd_repair_fallback_recovers() {
        // Same indefinite matrix; psd_repair should succeed via EVD.
        let original = mat_from_rows(3, &[&[1.0, 0.9, 0.9], &[0.9, 1.0, -0.9], &[0.9, -0.9, 1.0]]);
        let mut m = original.clone();
        psd_repair_and_factor(m.as_mut()).unwrap();

        // Reconstruct L · Lᵀ and compare with the hand-computed repair. The
        // input has eigenvalues {−0.8, 1.9, 1.9}, the negative one along
        // v = (1, −1, −1)/√3. Clipping it to the floor (≈ 0) adds 0.8·vvᵀ:
        // diagonal 1 + 0.8/3, off-diagonals ±(0.9 − 0.8/3). Renormalising to a
        // unit diagonal gives off-diagonals ±0.6333…/1.2666… = ±0.5 (the 1e-10
        // floor moves them by ~4e-11).
        let mut recon = Mat::<f64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += m[(i, k)] * m[(j, k)];
                }
                recon[(i, j)] = acc;
            }
        }
        let expected = [[1.0, 0.5, 0.5], [0.5, 1.0, -0.5], [0.5, -0.5, 1.0]];
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(recon[(i, j)], expected[i][j], 1e-9),
                    "repaired ({i},{j}) = {}, expected {}",
                    recon[(i, j)],
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn psd_repair_identity_bit_exact() {
        let mut m = Mat::<f64>::identity(4, 4);
        psd_repair_and_factor(m.as_mut()).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(m[(i, j)], expected, "({i},{j})");
            }
        }
    }

    /// CORR-11 (regression): a boundary PSD-but-singular matrix must survive the
    /// repair path. `[[1,1,0],[1,1,0],[0,0,1]]` has eigenvalues {0,1,2}: the
    /// direct Cholesky rejects the zero pivot, so the EVD fallback runs. Clipping
    /// the zero eigenvalue to exactly 0 leaves the reconstruction singular and the
    /// final (zero-tolerance) Cholesky fails (the bug that surfaced as
    /// `CorrelationNotPSD` on scenario-perturbed measured correlation matrices).
    /// Flooring at PSD_EIGENVALUE_FLOOR makes it strictly PD so repair succeeds.
    #[test]
    fn psd_repair_boundary_singular_matrix_succeeds() {
        let mut m = mat_from_rows(3, &[&[1.0, 1.0, 0.0], &[1.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
        // Sanity: the strict path rejects this singular matrix.
        assert!(
            matches!(
                factor_only(m.clone().as_mut()),
                Err(EngineError::CorrelationNotPSD)
            ),
            "boundary-singular matrix should fail the strict factor_only path"
        );
        psd_repair_and_factor(m.as_mut())
            .expect("repair must succeed on a boundary PSD-but-singular matrix");
        // Reconstructed L·Lᵀ has unit diagonal and is PSD.
        let mut recon = Mat::<f64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += m[(i, k)] * m[(j, k)];
                }
                recon[(i, j)] = acc;
            }
        }
        for i in 0..3 {
            assert!(
                approx_eq(recon[(i, i)], 1.0, 1e-9),
                "diag {i} = {}",
                recon[(i, i)]
            );
        }
    }

    /// CORR-10: both factoring paths accept a zero-size (n==0) matrix and
    /// return Ok without panicking (degenerate-but-valid input).
    #[test]
    fn zero_size_matrix_is_ok() {
        let mut a = Mat::<f64>::zeros(0, 0);
        assert!(
            factor_only(a.as_mut()).is_ok(),
            "factor_only on n==0 must be Ok"
        );

        let mut b = Mat::<f64>::zeros(0, 0);
        assert!(
            psd_repair_and_factor(b.as_mut()).is_ok(),
            "psd_repair_and_factor on n==0 must be Ok"
        );
    }
}
