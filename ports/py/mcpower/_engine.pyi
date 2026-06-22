"""Type stubs for the compiled Rust extension ``mcpower._engine``.

The actual module is built by maturin from ``crates/engine-py``. These stubs
exist solely so static type-checkers can see the public surface; nothing
imports ``_engine.pyi`` at runtime.
"""

from typing import Any, Callable


def set_n_threads(n: int, /) -> None: ...


def find_power(
    contracts_bytes: bytes,
    sample_size: int,
    n_sims: int,
    base_seed: int,
    progress: Callable[[int, int], bool] | None = None,
) -> dict[str, Any]: ...


def find_sample_size(
    contracts_bytes: bytes,
    target_power: float,
    lo: int,
    hi: int,
    n_sims: int,
    base_seed: int,
    method: str,
    by: int | None = None,
    by_kind: str | None = None,
    mode: str | None = None,
    tol_n: int | None = None,
    progress: Callable[[int, int], bool] | None = None,
) -> dict[str, Any]: ...


def build_contract_from_spec(
    json: str,
    outcome_kind: str,
    estimator: str,
    intercept: float,
    clusters_json: str,
    /,
) -> tuple[list[str], bytes, str]: ...


def fit_uploaded_data(
    contracts: bytes,
    scenario_index: int,
    seed: int,
    design: list[float],
    nrow: int,
    ncol: int,
    outcome: list[float],
    cluster_ids: list[int] | None,
    /,
) -> dict: ...


# Dev-only: registered behind the engine-py `test-bridge` Cargo feature.
# Release wheels (built without `--features test-bridge`) do NOT export this
# symbol — call sites must guard with `hasattr(_engine, "build_contract_from_json")`.
def build_contract_from_json(json: str, /) -> list[tuple[str, bytes]]: ...


# Dev-only (test-bridge): panics inside the engine boundary so tests can assert
# the internal-error path raises RuntimeError with the report hint. Absent from
# release wheels — guard with `hasattr(_engine, "panic_for_test")`.
def panic_for_test() -> None: ...


def power_plot_set_json(
    result: dict[str, Any],
    *,
    show_ci: bool = ...,
    target_power_line: float | None = ...,
) -> list[tuple[str, str]]: ...


def sample_size_plot_set_json(
    result: dict[str, Any],
    *,
    show_ci: bool = ...,
    target_power_line: float | None = ...,
) -> list[tuple[str, str]]: ...


def plot_html_template() -> str: ...


def parse_formula(input: str) -> Any: ...


def parse_assignments(input: str, kind: str, known: dict[str, Any]) -> Any: ...


def build_recovery_design(spec_json: str) -> tuple[list[float], list[str], int]: ...


def standardize_continuous(values: list[float]) -> list[float]: ...


def split_assignments(input: str) -> list[str]: ...


def plot_theme(name: str, /) -> str: ...


def list_plot_themes() -> list[str]: ...


def config() -> str: ...


def scenarios() -> str: ...


def dist_codes() -> str: ...


def residual_codes() -> str: ...


def re_dist_codes() -> str: ...
