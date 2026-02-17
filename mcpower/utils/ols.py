# Re-export from mcpower.stats.ols for backward compatibility
from ..stats.ols import *  # noqa: F401,F403
from ..stats.ols import (  # noqa: F401
    _USE_JIT,
    _generate_y_core,
    _generate_y_jit,
    _ols_core,
    _ols_jit,
)
