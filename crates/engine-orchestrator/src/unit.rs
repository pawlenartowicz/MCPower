//! Sample-size unit — a host-facing layer over total N for clustered designs.
//!
//! `find_sample_size` always searches and reports total observations N. A
//! researcher planning a mixed-model study usually thinks in the primary
//! grouping factor instead: participants when trials-per-participant is fixed
//! (`ClusterSizing::FixedSize`), or trials per participant when the number of
//! participants is fixed (`ClusterSizing::FixedClusters`). The unit is a
//! divisor between N and that quantity; hosts multiply bounds in on the way
//! in and divide results out on the way out. The engine grid, the cluster
//! atom and every existing field stay in N.
//!
//! The divisor is the primary sizing's fixed value, **not** `ClusterSpec::atom()`:
//! with 20 subjects (`FixedClusters`) crossed with 12 items the atom is 240,
//! but trials per subject is `N / 20` — counted across all item levels.
//! Extra groupings (`Crossed`, `NestedWithin`) are design constants that never
//! scale with N on their own, so they carry no unit.

use crate::find_sample_size::shared_primary_cluster;
use crate::result::{ByValue, GridMode, OrchestratorError, SampleSizeMethod};
use engine_contract::{ClusterSizing, ClusterSpec, SimulationContract};

/// Unit the host expresses sample-size bounds and results in.
/// `Observations` is the default and leaves every result identical to the
/// unit-less path (divisor 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleSizeUnit {
    /// Total observations N (divisor 1). Allowed for every model.
    Observations,
    /// Number of primary clusters. Requires `ClusterSizing::FixedSize`;
    /// divisor = `cluster_size`.
    Clusters,
    /// Observations per primary cluster, counted across all extra-grouping
    /// levels. Requires `ClusterSizing::FixedClusters`; divisor = `n_clusters`.
    ClusterSize,
}

impl SampleSizeUnit {
    /// Parse the host keyword: `"observations"` | `"clusters"` | `"cluster_size"`.
    pub fn parse(s: &str) -> Result<Self, OrchestratorError> {
        match s {
            "observations" => Ok(Self::Observations),
            "clusters" => Ok(Self::Clusters),
            "cluster_size" => Ok(Self::ClusterSize),
            other => Err(OrchestratorError::UnknownUnit(other.to_string())),
        }
    }

    /// The host keyword this unit parses from.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Observations => "observations",
            Self::Clusters => "clusters",
            Self::ClusterSize => "cluster_size",
        }
    }
}

fn regime_name(sizing: &ClusterSizing) -> &'static str {
    match sizing {
        ClusterSizing::FixedClusters { .. } => "fixed_clusters",
        ClusterSizing::FixedSize { .. } => "fixed_size",
    }
}

fn divisor_of(cluster: &ClusterSpec, unit: SampleSizeUnit) -> Result<usize, OrchestratorError> {
    let mismatch = || OrchestratorError::UnitRegimeMismatch {
        unit: unit.as_str().to_string(),
        regime: regime_name(&cluster.sizing).to_string(),
    };
    match (unit, &cluster.sizing) {
        (SampleSizeUnit::Observations, _) => Ok(1),
        (SampleSizeUnit::Clusters, ClusterSizing::FixedSize { cluster_size }) => {
            Ok((*cluster_size).max(1) as usize)
        }
        (SampleSizeUnit::ClusterSize, ClusterSizing::FixedClusters { n_clusters }) => {
            Ok((*n_clusters).max(1) as usize)
        }
        (SampleSizeUnit::Clusters, ClusterSizing::FixedClusters { .. })
        | (SampleSizeUnit::ClusterSize, ClusterSizing::FixedSize { .. }) => Err(mismatch()),
    }
}

/// Divisor between total N and the requested unit, read from the primary
/// `ClusterSpec.sizing` shared by every contract.
///
/// # Errors
/// `UnitRegimeMismatch` when the unit does not fit the primary sizing regime,
/// `UnitOnUnclustered` for `clusters` / `cluster_size` on a model without a
/// cluster, `UnitDivisorDiffersAcrossScenarios` when two scenarios share an
/// atom but not the primary fixed value (they can, when extras differ), and
/// `MixedClusterAtoms` from the shared primary-cluster lookup.
pub fn resolve_unit_divisor(
    contracts: &[SimulationContract],
    unit: SampleSizeUnit,
) -> Result<usize, OrchestratorError> {
    if unit == SampleSizeUnit::Observations {
        return Ok(1);
    }
    let Some(primary) = shared_primary_cluster(contracts)? else {
        return Err(OrchestratorError::UnitOnUnclustered {
            unit: unit.as_str().to_string(),
        });
    };
    let divisor = divisor_of(primary, unit)?;
    for cs in contracts
        .iter()
        .filter_map(|c| c.generation.cluster.as_ref())
    {
        let d = divisor_of(cs, unit)?;
        if d != divisor {
            return Err(OrchestratorError::UnitDivisorDiffersAcrossScenarios { a: divisor, b: d });
        }
    }
    Ok(divisor)
}

/// Convert `(from, to)` bounds and the grid method from the unit to rows.
/// Bounds are always multiplied. `by` is multiplied only when it is a linear
/// step (`ByValue::Fixed` under `GridMode::Linear`); under `GridMode::Log` a
/// `Fixed` value is a point count, and `Auto { count }` is always a count —
/// both pass through unchanged. Engine atom-snapping still applies on N
/// afterwards, so a `by = 1` step in `cluster_size` with 20 subjects × 12
/// crossed items snaps to 12 (atom 240 / 20) and the grid warning fires.
pub fn bounds_to_rows(
    bounds: (usize, usize),
    method: SampleSizeMethod,
    divisor: usize,
) -> ((usize, usize), SampleSizeMethod) {
    let d = divisor.max(1);
    let (from, to) = bounds;
    let method = match method {
        SampleSizeMethod::Grid {
            by: ByValue::Fixed(step),
            mode: GridMode::Linear,
        } => SampleSizeMethod::Grid {
            by: ByValue::Fixed(step * d),
            mode: GridMode::Linear,
        },
        other => other,
    };
    ((from * d, to * d), method)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_contract::{GroupingRelation, GroupingSpec};

    fn clustered(sizing: ClusterSizing) -> SimulationContract {
        let mut c = engine_contract::fixtures::example1_simple_ols();
        c.generation.cluster = Some(ClusterSpec::intercept_only(sizing, 0.25));
        c
    }

    fn with_extra(mut c: SimulationContract, relation: GroupingRelation) -> SimulationContract {
        c.generation.cluster.as_mut().unwrap().extra_groupings = vec![GroupingSpec {
            relation,
            tau_squared: 0.1,
            slopes: vec![],
        }];
        c
    }

    #[test]
    fn parse_accepts_the_three_keywords_and_rejects_others() {
        assert_eq!(
            SampleSizeUnit::parse("observations").unwrap(),
            SampleSizeUnit::Observations
        );
        assert_eq!(
            SampleSizeUnit::parse("clusters").unwrap(),
            SampleSizeUnit::Clusters
        );
        assert_eq!(
            SampleSizeUnit::parse("cluster_size").unwrap(),
            SampleSizeUnit::ClusterSize
        );
        assert!(matches!(
            SampleSizeUnit::parse("participants"),
            Err(OrchestratorError::UnknownUnit(s)) if s == "participants"
        ));
    }

    #[test]
    fn observations_is_divisor_one_for_every_model() {
        let plain = engine_contract::fixtures::example1_simple_ols();
        assert_eq!(
            resolve_unit_divisor(&[plain], SampleSizeUnit::Observations).unwrap(),
            1
        );
        let c = clustered(ClusterSizing::FixedSize { cluster_size: 8 });
        assert_eq!(
            resolve_unit_divisor(&[c], SampleSizeUnit::Observations).unwrap(),
            1
        );
    }

    #[test]
    fn clusters_unit_divides_by_cluster_size_under_fixed_size() {
        let c = clustered(ClusterSizing::FixedSize { cluster_size: 8 });
        assert_eq!(
            resolve_unit_divisor(&[c], SampleSizeUnit::Clusters).unwrap(),
            8
        );
    }

    #[test]
    fn cluster_size_unit_divides_by_n_clusters_under_fixed_clusters() {
        let c = clustered(ClusterSizing::FixedClusters { n_clusters: 20 });
        assert_eq!(
            resolve_unit_divisor(&[c], SampleSizeUnit::ClusterSize).unwrap(),
            20
        );
    }

    #[test]
    fn regime_mismatch_errors_both_ways() {
        let fixed_size = clustered(ClusterSizing::FixedSize { cluster_size: 8 });
        assert!(matches!(
            resolve_unit_divisor(&[fixed_size], SampleSizeUnit::ClusterSize),
            Err(OrchestratorError::UnitRegimeMismatch { unit, regime })
                if unit == "cluster_size" && regime == "fixed_size"
        ));
        let fixed_clusters = clustered(ClusterSizing::FixedClusters { n_clusters: 20 });
        assert!(matches!(
            resolve_unit_divisor(&[fixed_clusters], SampleSizeUnit::Clusters),
            Err(OrchestratorError::UnitRegimeMismatch { unit, regime })
                if unit == "clusters" && regime == "fixed_clusters"
        ));
    }

    #[test]
    fn derived_units_error_on_unclustered_model() {
        let plain = engine_contract::fixtures::example1_simple_ols();
        assert!(matches!(
            resolve_unit_divisor(std::slice::from_ref(&plain), SampleSizeUnit::Clusters),
            Err(OrchestratorError::UnitOnUnclustered { unit }) if unit == "clusters"
        ));
        assert!(matches!(
            resolve_unit_divisor(&[plain], SampleSizeUnit::ClusterSize),
            Err(OrchestratorError::UnitOnUnclustered { unit }) if unit == "cluster_size"
        ));
    }

    #[test]
    fn cluster_size_divisor_is_primary_value_not_atom_with_crossed_extra() {
        // 20 subjects × 12 crossed items: atom 240, divisor stays 20.
        let c = with_extra(
            clustered(ClusterSizing::FixedClusters { n_clusters: 20 }),
            GroupingRelation::Crossed { n_clusters: 12 },
        );
        assert_eq!(c.generation.cluster.as_ref().unwrap().atom(), 240);
        assert_eq!(
            resolve_unit_divisor(&[c], SampleSizeUnit::ClusterSize).unwrap(),
            20
        );
    }

    #[test]
    fn cluster_size_divisor_is_primary_value_not_atom_with_nested_extra() {
        let c = with_extra(
            clustered(ClusterSizing::FixedClusters { n_clusters: 20 }),
            GroupingRelation::NestedWithin { n_per_parent: 3 },
        );
        assert_eq!(c.generation.cluster.as_ref().unwrap().atom(), 60);
        assert_eq!(
            resolve_unit_divisor(&[c], SampleSizeUnit::ClusterSize).unwrap(),
            20
        );
    }

    #[test]
    fn clusters_divisor_with_nested_extra_is_cluster_size() {
        // NestedWithin is the only extra FixedSize admits (Crossed is rejected
        // by contract validation).
        let c = with_extra(
            clustered(ClusterSizing::FixedSize { cluster_size: 8 }),
            GroupingRelation::NestedWithin { n_per_parent: 2 },
        );
        assert_eq!(c.generation.cluster.as_ref().unwrap().atom(), 16);
        assert_eq!(
            resolve_unit_divisor(&[c], SampleSizeUnit::Clusters).unwrap(),
            8
        );
    }

    #[test]
    fn same_atom_different_primary_across_scenarios_errors() {
        // 20 × crossed 12 (atom 240) vs 240 plain clusters (atom 240).
        let a = with_extra(
            clustered(ClusterSizing::FixedClusters { n_clusters: 20 }),
            GroupingRelation::Crossed { n_clusters: 12 },
        );
        let b = clustered(ClusterSizing::FixedClusters { n_clusters: 240 });
        assert!(matches!(
            resolve_unit_divisor(&[a, b], SampleSizeUnit::ClusterSize),
            Err(OrchestratorError::UnitDivisorDiffersAcrossScenarios { a: 20, b: 240 })
        ));
    }

    #[test]
    fn bounds_to_rows_scales_bounds_and_only_linear_fixed_step() {
        let linear_fixed = SampleSizeMethod::Grid {
            by: ByValue::Fixed(5),
            mode: GridMode::Linear,
        };
        let (b, m) = bounds_to_rows((10, 50), linear_fixed, 8);
        assert_eq!(b, (80, 400));
        assert_eq!(
            m,
            SampleSizeMethod::Grid {
                by: ByValue::Fixed(40),
                mode: GridMode::Linear
            }
        );

        let log_fixed = SampleSizeMethod::Grid {
            by: ByValue::Fixed(5),
            mode: GridMode::Log,
        };
        let (b, m) = bounds_to_rows((10, 50), log_fixed, 8);
        assert_eq!(b, (80, 400));
        assert_eq!(m, log_fixed);

        let auto = SampleSizeMethod::Grid {
            by: ByValue::Auto { count: 12 },
            mode: GridMode::Linear,
        };
        let (b, m) = bounds_to_rows((10, 50), auto, 8);
        assert_eq!(b, (80, 400));
        assert_eq!(m, auto);

        // Divisor 1 is the identity.
        let (b, m) = bounds_to_rows((10, 50), linear_fixed, 1);
        assert_eq!(b, (10, 50));
        assert_eq!(m, linear_fixed);
    }
}
