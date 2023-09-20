"""Deep Learning Utilities for PyTorch users."""
__version__ = '0.0.19.dev0'

from . import cuda, data, hardware, nn, random, utils
from ._stream import Stream
from ._tensor_ops import cat, concat, iter_batches, to
from ._tools import EarlyStopping, ProgressTracker, Timer
from ._utilities import evaluation, improve_reproducibility
from .data import collate

__all__ = [
    # Modules.
    'cuda',
    'nn',
    'random',
    'utils',
    # Functions and classes (the order is optimized for pdoc).
    'to',
    'cat',
    'iter_batches',
    'EarlyStopping',
    'Timer',
    # Deprecated.
    'data',
    'hardware',
    'collate',
    'concat',
    'evaluation',
    'improve_reproducibility',
    'ProgressTracker',
    'Stream',
]
