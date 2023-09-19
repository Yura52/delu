"""Deep Learning Utilities for PyTorch users."""
__version__ = '0.0.19.dev0'

from . import cuda, data, hardware, nn, random
from ._stream import Stream
from ._tensor_ops import cat, concat, iter_batches, to
from ._tools import EarlyStopping, ProgressTracker, Timer
from ._utilities import evaluation, improve_reproducibility
from .data import collate

__all__ = [
    # Modules.
    'cuda',
    'data',
    'nn',
    'random',
    # Functions and classes (the order is optimized for pdoc).
    'to',
    'cat',
    'iter_batches',
    'EarlyStopping',
    'Timer',
    # Deprecated.
    'hardware',
    'collate',
    'concat',
    'evaluation',
    'improve_reproducibility',
    'ProgressTracker',
    'Stream',
]
