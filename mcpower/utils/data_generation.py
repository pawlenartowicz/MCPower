# Re-export from mcpower.stats.data_generation for backward compatibility
from ..stats.data_generation import *  # noqa: F401,F403
from ..stats.data_generation import (  # noqa: F401
    _USE_JIT,
    NORM_CDF_TABLE,
    T3_PPF_TABLE,
    _generate_cluster_effects,
    _generate_factors,
    _generate_X,
    _generate_X_core,
    _generate_X_jit,
    _init_tables,
)
