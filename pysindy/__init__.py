from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass


from .pysindy import SINDy
from pysindy.differentiation import *
from pysindy.optimizers import *
from pysindy.feature_library import *
from pysindy.utils import *
