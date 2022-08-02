"""Deep Learning Utilities for PyTorch users."""
__version__ = '0.0.14.dev0'

from . import data  # noqa
from . import hardware  # noqa
from . import nn  # noqa
from . import random  # noqa
from ._monitoring import ProgressTracker, Timer  # noqa
from ._tensor_array_ops import concat, iter_batches, to  # noqa
from ._utilities import evaluation, improve_reproducibility  # noqa
from .data import Stream  # noqa (NOTE: deprecated alias)
