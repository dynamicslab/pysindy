from .base import BaseOptimizer
from .base import EnsembleOptimizer
from .frols import FROLS
from .sr3 import SR3
from .ssr import SSR
from .stlsq import STLSQ
from .wrapped_optimizer import WrappedOptimizer

try:
    from .constrained_sr3 import ConstrainedSR3
except (ImportError, NameError):
    pass
try:
    from .miosr import MIOSR
except (ImportError, NameError):
    pass
try:
    from .trapping_sr3 import TrappingSR3
except (ImportError, NameError):
    pass
try:
    from .sindy_pi import SINDyPI
except (ImportError, NameError):
    pass
try:
    from .stable_linear_sr3 import StableLinearSR3
except (ImportError, NameError):
    pass
try:
    from .sbr import SBR
except (ImportError, NameError):
    pass


__all__ = [
    "BaseOptimizer",
    "EnsembleOptimizer",
    "WrappedOptimizer",
    "SR3",
    "STLSQ",
    "ConstrainedSR3",
    "StableLinearSR3",
    "TrappingSR3",
    "SSR",
    "FROLS",
    "SINDyPI",
    "MIOSR",
    "SBR",
]
