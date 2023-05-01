# This module contains internal private tools imported by other modules.

import functools
import inspect
import warnings

from .exceptions import DeLUDeprecationWarning


def deprecated(message: str):
    def decorator(item):
        assert item.__doc__ is not None
        docstring = f'[**DEPRECATED** | {message}]' + ' ' + item.__doc__

        def warn(item_type):
            warnings.warn(
                f'The {item_type} {item.__qualname__}` is deprecated'
                ' and will be removed in future releases. ' + message,
                DeLUDeprecationWarning,
            )

        if isinstance(item, type):
            wrapper = item
            wrapper.__doc__ = docstring
        else:
            assert inspect.isfunction(item)

            @functools.wraps(item)
            def wrapper(*args, **kwargs):
                warn('function')
                return item(*args, **kwargs)

            wrapper.__doc__ = docstring

        return wrapper

    return decorator


def is_namedtuple(x) -> bool:
    return isinstance(x, tuple) and all(
        hasattr(x, attr) for attr in ['_make', '_asdict', '_replace', '_fields']
    )
