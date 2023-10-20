print("....importing local pysindy copy....")
from .base import BaseOptimizer
from .base import EnsembleOptimizer
from .constrained_sr3 import ConstrainedSR3
from .frols import FROLS

try:  # Waiting on PEP 690 to lazy import gurobipy
    from .miosr import MIOSR
except ImportError:
    pass
try:  # Waiting on PEP 690 to lazy import cvxpy
    from .trapping_sr3 import TrappingSR3
except ImportError:
    pass
try:  # Waiting on PEP 690 to lazy import cvxpy
    from .sindy_pi import SINDyPI
except ImportError:
    pass
try:  # Waiting on PEP 690 to lazy import cvxpy
    from .stable_linear_sr3 import StableLinearSR3
except ImportError:
    pass
from .wrapped_optimizer import WrappedOptimizer
from .sr3 import SR3
from .ssr import SSR
from .stlsq import STLSQ
from .adam_stlsq import adam_STLSQ

__all__ = [
    "BaseOptimizer",
    "EnsembleOptimizer",
    "WrappedOptimizer",
    "SR3",
    "STLSQ",
    "adam_STLSQ",
    "ConstrainedSR3",
    "StableLinearSR3",
    "TrappingSR3",
    "SSR",
    "FROLS",
    "SINDyPI",
    "MIOSR",
]
