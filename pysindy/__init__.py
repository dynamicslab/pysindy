from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass


from .pysindy import SINDy
from . import differentiation
from . import feature_library
from . import optimizers
from . import utils


__all__ = ["SINDy"]
__all__.extend(differentiation.__all__)
__all__.extend(feature_library.__all__)
__all__.extend(optimizers.__all__)
__all__.extend(utils.__all__)
