from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass


from . import differentiation
from . import feature_library
from . import optimizers
from . import utils
from .pysindy import SINDy
from .differentiation import BaseDifferentiation
from .differentiation import FiniteDifference
from .differentiation import SINDyDerivative
from .differentiation import SmoothedFiniteDifference
from .feature_library import ConcatLibrary
from .feature_library import CustomLibrary
from .feature_library import FourierLibrary
from .feature_library import IdentityLibrary
from .feature_library import PolynomialLibrary
from .optimizers import BaseOptimizer
from .optimizers import SINDyOptimizer
from .optimizers import SR3
from .optimizers import STLSQ


__all__ = ["SINDy"]
__all__.extend(differentiation.__all__)
__all__.extend(feature_library.__all__)
__all__.extend(optimizers.__all__)
__all__.extend(["utils"])
