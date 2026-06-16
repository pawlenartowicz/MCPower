//! Resolves user-supplied target names to effect-vector indices for the orchestrator.
//!
//! Column-ordering invariant: see `variables.rs` (owning explanation). Effect index `i` maps to
//! design column `i + 1` because column 0 is the intercept. `"overall"` matching is case-sensitive.

use crate::error::SpecError;

/// Resolve user-supplied target names to indices into the effect-sizes
/// vector. The effect-sizes layout is `[intercept, β_for_effect_1,
/// β_for_effect_2, …]`, so `effect_names[i]` maps to index `i + 1`.
///
/// `"overall"` (case-sensitive, mirrors the Python frontend) expands to
/// every effect. Unknown names raise `UnknownTarget`. The output is
/// deduplicated and sorted ascending.
pub fn resolve_targets(targets: &[String], effect_names: &[String]) -> Result<Vec<u32>, SpecError> {
    use std::collections::BTreeSet;
    let mut out: BTreeSet<u32> = BTreeSet::new();
    let expand_all = targets.iter().any(|t| t == "overall");
    if expand_all {
        for i in 0..effect_names.len() {
            out.insert((i + 1) as u32);
        }
    }
    for t in targets {
        if t == "overall" {
            continue;
        }
        let idx = effect_names
            .iter()
            .position(|n| n == t)
            .ok_or_else(|| SpecError::UnknownTarget { name: t.clone() })?;
        out.insert((idx + 1) as u32);
    }
    Ok(out.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overall_expands_to_all_effects() {
        let effects = vec!["x1".into(), "x2".into(), "x1:x2".into()];
        let r = resolve_targets(&["overall".into()], &effects).unwrap();
        assert_eq!(r, vec![1, 2, 3]);
    }

    #[test]
    fn explicit_names_resolve_to_indices() {
        let effects = vec!["x1".into(), "x2".into()];
        let r = resolve_targets(&["x2".into()], &effects).unwrap();
        assert_eq!(r, vec![2]);
    }

    #[test]
    fn unknown_target_errors() {
        let effects = vec!["x1".into()];
        let err = resolve_targets(&["x9".into()], &effects).unwrap_err();
        assert!(matches!(err, SpecError::UnknownTarget { .. }));
    }

    #[test]
    fn duplicates_collapse() {
        let effects = vec!["x1".into(), "x2".into()];
        let r = resolve_targets(&["x1".into(), "x1".into()], &effects).unwrap();
        assert_eq!(r, vec![1]);
    }

    // "overall" is case-sensitive — "Overall" is not the expansion keyword
    // and is treated as an unknown target name.
    #[test]
    fn overall_is_case_sensitive() {
        let effects = vec!["x1".into(), "x2".into()];
        let err = resolve_targets(&["Overall".into()], &effects).unwrap_err();
        assert!(matches!(err, SpecError::UnknownTarget { .. }));
    }
}
