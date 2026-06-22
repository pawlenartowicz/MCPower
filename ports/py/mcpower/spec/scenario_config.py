"""Default scenario configs, sourced from the engine-embedded
``configs/scenarios.json`` via ``_engine.scenarios()``."""

import copy
import json
from functools import lru_cache
from typing import Any, Dict


@lru_cache(maxsize=1)
def _cached() -> Dict[str, Any]:
    from .. import _engine

    return json.loads(_engine.scenarios())


def get_default_scenario_config() -> Dict[str, Any]:
    """Fresh deep copy of the canonical defaults (safe to mutate)."""
    return copy.deepcopy(_cached())
