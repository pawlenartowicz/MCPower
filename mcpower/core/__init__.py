"""Core components for the MCPower framework.

Re-exports the foundational building blocks:

- ``VariableRegistry``, ``PredictorVar``, ``Effect`` — variable and effect
  management.
- ``SimulationRunner``, ``SimulationMetadata``, ``prepare_metadata`` — Monte
  Carlo simulation execution.
- ``ScenarioRunner``, ``apply_per_simulation_perturbations``,
  ``DEFAULT_SCENARIO_CONFIG`` — robustness / scenario analysis.
- ``ResultsProcessor``, ``build_power_result``, ``build_sample_size_result``
  — power calculation and result formatting.
"""

from .results import ResultsProcessor, build_power_result, build_sample_size_result
from .scenarios import (
    DEFAULT_SCENARIO_CONFIG,
    ScenarioRunner,
    apply_per_simulation_perturbations,
)
from .simulation import SimulationMetadata, SimulationRunner, prepare_metadata
from .variables import Effect, PredictorVar, VariableRegistry

__all__ = [
    # Variables
    "VariableRegistry",
    "PredictorVar",
    "Effect",
    # Simulation
    "SimulationRunner",
    "SimulationMetadata",
    "prepare_metadata",
    # Scenarios
    "ScenarioRunner",
    "apply_per_simulation_perturbations",
    "DEFAULT_SCENARIO_CONFIG",
    # Results
    "ResultsProcessor",
    "build_power_result",
    "build_sample_size_result",
]
