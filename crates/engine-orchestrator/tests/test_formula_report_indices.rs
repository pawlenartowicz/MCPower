//! Regression: a `test_formula` that drops the **leading** generation term and
//! reports a later one must surface host-facing `target_indices` in the reduced
//! (design_test term) space, not the generation-kernel space. Mismatched spaces
//! sent the port report's `skeleton[target_indices[i]]` lookup out of range and
//! crashed `.summary()`.

use engine_contract::{ColumnId, DesignSpec, DesignTerm, TestTarget};
use engine_orchestrator::{find_power, CancellationToken};

#[test]
fn reduced_test_formula_dropping_leading_term_reports_skeleton_aligned_indices() {
    // Generation y ~ x1 + x2 (kernel cols: x1 = 1, x2 = 2). The test design drops
    // the leading x1 and keeps x2 → reduced design_test [Const, x2], whose only
    // non-intercept term is position 1 (skeleton length 2).
    let mut c = engine_contract::fixtures::example1_simple_ols();
    c.design_test = Some(DesignSpec {
        terms: vec![
            DesignTerm::Const,
            DesignTerm::Direct {
                column: ColumnId(1),
            },
        ],
    });
    // Test the kept term, which is design_test term position 1 (= x2).
    c.test.targets = vec![TestTarget::Marginal { term: 1 }];

    let cancel = CancellationToken::new();
    let result = find_power(&[c.clone()], 100, 200, 2137, None, &cancel)
        .expect("find_power ok for reduced test_formula");

    assert_eq!(result.scenarios.len(), 1);
    let (_, pr) = &result.scenarios[0];

    // Host-facing index is x2's reduced term position (1), NOT its generation
    // kernel column (2). The reduced effect skeleton has 2 entries (Const + x2);
    // an index of 2 would be out of range and crash `.summary()`.
    let skeleton_len = c.design_test.as_ref().unwrap().terms.len();
    assert_eq!(
        pr.target_indices,
        vec![1],
        "report index = reduced term position"
    );
    assert!(
        pr.target_indices.iter().all(|&i| i < skeleton_len),
        "every reported target index must be in range for the reduced skeleton (len {skeleton_len})"
    );
    // Power is still computed correctly (kernel read used the generation column).
    let power = pr.power_uncorrected[0];
    assert!(
        power.is_finite() && (0.0..=1.0).contains(&power),
        "expected finite power in [0,1], got {power}"
    );
}
