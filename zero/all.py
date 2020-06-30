r"""A single entry point to Zero.

The module simply imports everything (:code:`from ... import *`) from all submodules.
It is neither "better" nor "worse" to use this module instead of explicit imports from
submodules, it is completely up to a user. Just keep in mind that if *all* submodules
you need do not import `torch` (or any other heavy libraries) under the hood, it will
be faster to import them individually then via `zero.all`.

Examples:
    .. testcode::

        import zero.all as zero
        flow = zero.Flow(range(10))
        progress = zero.ProgressTracker(1, 0.0)
        timer = zero.Timer()

    If you just need a `zero.time.Timer`, this is a bit faster:

    .. testcode::

        from zero.time import Timer
"""

from .concat_dmap import *  # noqa
from .data import *  # noqa
from .flow import *  # noqa
from .hardware import *  # noqa
from .io import *  # noqa
from .metrics import *  # noqa
from .model import *  # noqa
from .optim import *  # noqa
from .progress import *  # noqa
from .random import *  # noqa
from .tensor import *  # noqa
from .time import *  # noqa
from .types import *  # noqa
