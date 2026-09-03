import sys

import numpy.core
import numpy.core.multiarray


sys.modules.setdefault("numpy._core", numpy.core)
sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
