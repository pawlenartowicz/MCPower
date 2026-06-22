//! Builds and validates the predictor correlation matrix from user-supplied pairs; rejects non-PSD inputs.
//!
//! Matrix storage: flat `n×n`, `m[j*n + i]` (column-major); symmetry makes row- vs column-major equivalent.

use crate::error::SpecError;
use crate::input::CorrelationPair;

/// Build the flat column-major `n×n` correlation matrix from user-supplied
/// pairs. Missing pairs default to 0.0; the diagonal is 1.0. The result is
/// checked for positive semi-definiteness via a Cholesky-with-pivoting test.
///
/// # Errors
/// - [`SpecError::CorrelationUnknownVar`]: a pair names a variable not in `non_factor`.
/// - [`SpecError::CorrelationOutOfRange`]: a correlation value outside [-1, 1] (or non-finite).
/// - [`SpecError::CorrelationNotPsd`]: the assembled matrix is not positive semi-definite.
pub fn build_correlation_matrix(
    non_factor: &[String],
    pairs: &[CorrelationPair],
) -> Result<Vec<f64>, SpecError> {
    let n = non_factor.len();
    let mut m = vec![0.0_f64; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }

    let index = |name: &str| -> Result<usize, SpecError> {
        non_factor
            .iter()
            .position(|p| p == name)
            .ok_or_else(|| SpecError::CorrelationUnknownVar {
                name: name.to_string(),
            })
    };

    for pair in pairs {
        let v = pair.value;
        if !(-1.0..=1.0).contains(&v) || !v.is_finite() {
            return Err(SpecError::CorrelationOutOfRange {
                a: pair.a.clone(),
                b: pair.b.clone(),
                value: v,
            });
        }
        let i = index(&pair.a)?;
        let j = index(&pair.b)?;
        m[j * n + i] = v;
        m[i * n + j] = v;
    }

    if !is_psd(&m, n) {
        return Err(SpecError::CorrelationNotPsd);
    }
    Ok(m)
}

/// Cholesky-based PSD test. Returns true if successful, false on any
/// non-positive pivot (allowing a small numerical tolerance).
pub(crate) fn is_psd(mat: &[f64], n: usize) -> bool {
    if n == 0 {
        return true;
    }
    // Match Python's tolerance (validators.py) so the Rust builder
    // doesn't reject correlation matrices the existing frontend accepts.
    const EPS: f64 = -1e-8;
    let mut l = vec![0.0_f64; n * n];
    for j in 0..n {
        let mut s = mat[j * n + j];
        for k in 0..j {
            s -= l[k * n + j] * l[k * n + j];
        }
        if s < EPS {
            return false;
        }
        let s = s.max(0.0).sqrt();
        l[j * n + j] = s;
        if s == 0.0 {
            // Zero pivot: column below must also be zero for PSD.
            for i in (j + 1)..n {
                let mut sum = mat[j * n + i];
                for k in 0..j {
                    sum -= l[k * n + i] * l[k * n + j];
                }
                if sum.abs() > 1e-8 {
                    return false;
                }
            }
            continue;
        }
        for i in (j + 1)..n {
            let mut sum = mat[j * n + i];
            for k in 0..j {
                sum -= l[k * n + i] * l[k * n + j];
            }
            l[j * n + i] = sum / s;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_when_no_pairs() {
        let m = build_correlation_matrix(&["x1".into(), "x2".into()], &[]).unwrap();
        assert_eq!(m, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn fills_symmetric_entries() {
        let m = build_correlation_matrix(
            &["a".into(), "b".into(), "c".into()],
            &[CorrelationPair {
                a: "a".into(),
                b: "c".into(),
                value: 0.4,
            }],
        )
        .unwrap();
        // a-c entry: indices (0,2) and (2,0) in column-major 3x3.
        assert_eq!(m[2 * 3 + 0], 0.4);
        assert_eq!(m[0 * 3 + 2], 0.4);
        // Diagonal intact.
        assert_eq!(m[0], 1.0);
        assert_eq!(m[4], 1.0);
        assert_eq!(m[8], 1.0);
    }

    #[test]
    fn out_of_range_value_errors() {
        let err = build_correlation_matrix(
            &["a".into(), "b".into()],
            &[CorrelationPair {
                a: "a".into(),
                b: "b".into(),
                value: 1.5,
            }],
        )
        .unwrap_err();
        assert!(matches!(err, SpecError::CorrelationOutOfRange { .. }));
    }

    #[test]
    fn non_psd_triple_errors() {
        // r(a,b)=0.9, r(a,c)=0.9, r(b,c)=-0.9 → not PSD.
        let err = build_correlation_matrix(
            &["a".into(), "b".into(), "c".into()],
            &[
                CorrelationPair {
                    a: "a".into(),
                    b: "b".into(),
                    value: 0.9,
                },
                CorrelationPair {
                    a: "a".into(),
                    b: "c".into(),
                    value: 0.9,
                },
                CorrelationPair {
                    a: "b".into(),
                    b: "c".into(),
                    value: -0.9,
                },
            ],
        )
        .unwrap_err();
        assert!(matches!(err, SpecError::CorrelationNotPsd));
    }

    // The PSD check uses a Cholesky-with-pivoting pass with a negative
    // tolerance so a matrix that is PSD but singular at the boundary
    // (e.g. perfectly-correlated pair, smallest eigenvalue exactly 0) is
    // accepted rather than rejected as indefinite. A stricter (zero or
    // positive) tolerance would reject this.
    #[test]
    fn boundary_psd_singular_matrix_is_accepted() {
        let m = build_correlation_matrix(
            &["a".into(), "b".into()],
            &[CorrelationPair {
                a: "a".into(),
                b: "b".into(),
                value: 1.0,
            }],
        )
        .expect("a perfectly-correlated (singular but PSD) matrix must be accepted");
        // Symmetric write landed and diagonal intact.
        assert_eq!(m, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn unknown_var_errors() {
        let err = build_correlation_matrix(
            &["a".into()],
            &[CorrelationPair {
                a: "a".into(),
                b: "z".into(),
                value: 0.1,
            }],
        )
        .unwrap_err();
        match err {
            SpecError::CorrelationUnknownVar { name } => assert_eq!(name, "z"),
            other => panic!("got {other:?}"),
        }
    }
}
