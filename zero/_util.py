from collections.abc import Mapping, Sequence
from typing import Callable, Generator, List

from .types import OneOrList, Recursive, S, T


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
    elif isinstance(data, tuple):
        is_namedtuple = all(
            hasattr(data, x) for x in ['_make', '_asdict', '_replace', '_fields']
        )
        # mypy doesn't understand the branch for named tuples
        return (type(data)._make if is_namedtuple else type(data))(  # type: ignore
            traverse(fn, x) for x in data
        )
    elif isinstance(data, list):
        return type(data)(traverse(fn, x) for x in data)
    elif isinstance(data, dict):
        return type(data)({k: traverse(fn, v) for k, v in data.items()})
    else:
        return fn(data)
