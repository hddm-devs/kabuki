from hierarchical import *

import utils
import analyze

try:
    from IPython.core.debugger import Tracer; debug_here = Tracer()
except:
    try:
        from IPython.Debugger import Tracer; debug_here = Tracer()
    except:
        def debug_here(): pass
