from .hierarchical import *

from . import utils
from . import analyze
from . import step_methods as steps
from . import generate

__version__ = '0.6.2'

try:
    from IPython.core.debugger import Tracer; debug_here = Tracer()
except:
    try:
        from IPython.Debugger import Tracer; debug_here = Tracer()
    except:
        def debug_here(): pass

try:
    from collections import OrderedDict
except:
    from OrderedDict import OrderedDict
