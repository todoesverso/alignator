from .align import *
from .brightandcontrast import *
from .histogram import *
from .video import *
from .vignette import *
from .resize import *
from .manipulator import Manipulator
import sys
import inspect


def get_manipulators():
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    return [c[1] for c in clsmembers if c[1].__base__ == Manipulator]
