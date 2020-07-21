"""A single entry point to Zero.

The module simply imports everything from all submodules (:code:`from ... import *`).
Keep in mind that if you don't use `torch` in your project and you need only non-torch
tools from Zero, then it will be faster to import them directly from submodules than
from `zero.all`.

Examples:
    .. testcode::

        from zero.all import Stream, Timer, concat
        # or
        import zero.all as zero  # then use zero.Stream, zero.Timer, zero.concat

    If you just need a `zero.time.Timer` and you don't use PyTorch in your project, this
    is a bit faster:

    .. testcode::

        from zero.time import Timer
"""

from .data import *  # noqa
from .hardware import *  # noqa
from .io import *  # noqa
from .metrics import *  # noqa
from .random import *  # noqa
from .stream import *  # noqa
from .time import *  # noqa
from .training import *  # noqa
from .types import *  # noqa
