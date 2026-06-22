//! Shared helper for engine-spec-builder integration tests.
//!
//! Routes the legacy `LinearSpec -> Vec<SimulationSpec>` shape through the
//! contract path (`build_linear_contract` + `contract_to_simulation_spec`).
//! Tests in this crate validate the post-projection kernel layout, which
//! requires running the adapter after the contract builder.

use engine_core::contract_adapter::{contract_to_simulation_spec, AdapterError};
use engine_core::spec::SimulationSpec;
use engine_spec_builder::{build_linear_contract, LinearSpec, SpecError};

#[allow(dead_code)]
pub fn build_linear_spec(spec: &LinearSpec) -> Result<Vec<SimulationSpec>, SpecError> {
    let contracts = build_linear_contract(spec)?;
    contracts
        .into_iter()
        .map(|c| contract_to_simulation_spec(&c).map_err(adapter_to_spec_error))
        .collect()
}

fn adapter_to_spec_error(e: AdapterError) -> SpecError {
    match e {
        AdapterError::JointNotSupported { got } => SpecError::InternalContractValidate(format!(
            "TestTarget::Joint {got:?} unsupported by current engine"
        )),
        AdapterError::ContractInvalid(msg) => SpecError::InternalContractValidate(msg),
    }
}
