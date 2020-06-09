from collections.abc import Mapping, Sequence
from typing import Callable, Generator, List

from .types import OneOrList, Recursive, S, T


def is_namedtuple(x):
    return isinstance(x, tuple) and all(
        hasattr(x, attr) for attr in ['_make', '_asdict', '_replace', '_fields']
    )


def to_list(x: OneOrList[T]) -> List[T]:
    return x if isinstance(x, list) else [x]


def flatten(data: Recursive[T]) -> Generator[T, None, None]:
    if isinstance(data, (str, bytes)):
        # mypy: NaN
        yield data  # type: ignore
    elif isinstance(data, Sequence):
        for x in data:
            yield from flatten(x)
    elif isinstance(data, Mapping):
        for x in data.values():
            yield from flatten(x)
    else:
        yield data


def traverse(fn: Callable[[T], S], data: Recursive[T]) -> Recursive[S]:
    if isinstance(data, (str, bytes)):
        # mypy: NaN
        return fn(data)  # type: ignore
    elif is_namedtuple(data):
        # mypy: NaN
        return type(data)._make(traverse(fn, x) for x in data)  # type: ignore
    elif isinstance(data, tuple):
        return type(data)(traverse(fn, x) for x in data)
    elif isinstance(data, list):
        return type(data)(traverse(fn, x) for x in data)
    elif isinstance(data, dict):
        return type(data)({k: traverse(fn, v) for k, v in data.items()})
    else:
        return fn(data)
