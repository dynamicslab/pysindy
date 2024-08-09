from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass

from . import differentiation
from . import feature_library
from . import optimizers
from . import deeptime
from . import utils
from .pysindy import SINDy
from .pysindy import AxesArray
from .differentiation import BaseDifferentiation
from .differentiation import FiniteDifference
from .differentiation import SpectralDerivative
from .differentiation import SINDyDerivative
from .differentiation import SmoothedFiniteDifference
from .feature_library import ConcatLibrary
from .feature_library import TensoredLibrary
from .feature_library import GeneralizedLibrary
from .feature_library import CustomLibrary
from .feature_library import FourierLibrary
from .feature_library import IdentityLibrary
from .feature_library import PolynomialLibrary
from .feature_library import PDELibrary
from .feature_library import WeakPDELibrary
from .feature_library import SINDyPILibrary
from .feature_library import ParameterizedLibrary
from .optimizers import BaseOptimizer
from .optimizers import FROLS
from .optimizers import WrappedOptimizer
from .optimizers import SR3
from .optimizers import SSR
from .optimizers import STLSQ
from .optimizers import EnsembleOptimizer

try:
    from .optimizers import ConstrainedSR3
except (ImportError, NameError):
    pass
try:
    from .optimizers import MIOSR
except (ImportError, NameError):
    pass
try:
    from .optimizers import SINDyPI
except (ImportError, NameError):
    pass
try:
    from .optimizers import TrappingSR3
except (ImportError, NameError):
    pass
try:
    from .optimizers import StableLinearSR3
except (ImportError, NameError):
    pass
try:
    from .optimizers import SBR
except (ImportError, NameError):
    pass


__all__ = ["SINDy", "AxesArray"]
__all__.extend(differentiation.__all__)
__all__.extend(feature_library.__all__)
__all__.extend(optimizers.__all__)
__all__.extend(["utils"])
__all__.extend(["deeptime"])
