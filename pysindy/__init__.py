from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass


from pysindy.pysindy import SINDy
from pysindy.differentiation import *
from pysindy.optimizers import *
from pysindy.feature_library import *
from pysindy.utils import *
