from hierarchical import *

import utils
import analyze
import step_methods as steps
import generate
import distributions
import tests

__version__ = '0.5dev'

try:
    from IPython.core.debugger import Tracer; debug_here = Tracer()
except:
    try:
        from IPython.Debugger import Tracer; debug_here = Tracer()
    except:
        def debug_here(): pass
