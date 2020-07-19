"""Shortcuts for input and output."""

__all__ = [
    'load_pickle',
    'dump_pickle',
    'load_json',
    'dump_json',
    'load_jsonl',
    'dump_jsonl',
    'extend_jsonl',
]

import json
import pickle
from typing import Any, Iterable, List

from .types import JSON, PathLike


def load_pickle(path: PathLike, **kwargs) -> Any:
    """Load pickled object from a file.

    Args:
        path (`~zero.types.PathLike`): path to the file. The file must exist.
        kwargs: arguments for `pickle.load`
    Returns:
        The upickled object.
    """
    with open(path, 'rb') as f:
        return pickle.load(f, **kwargs)


def dump_pickle(x: Any, path: PathLike, **kwargs) -> None:
    """Dump an object to a file in Pickle format.

    Args:
        x: the object
        path (`~zero.types.PathLike`): path to the file. If the file doesn't exist,
            it will be created, otherwise, overwritten.
        kwargs: arguments for `pickle.dump`
    """
    with open(path, 'wb') as f:
        pickle.dump(x, f, **kwargs)


def load_json(path: PathLike, **kwargs) -> JSON:
    """Load JSON data from a file.

    Args:
        path (`~zero.types.PathLike`): path to the file. The file must exist.
        kwargs: arguments for `json.load`
    Returns:
        `~zero.types.JSON`: The data.
    """
    with open(path) as f:
        return json.load(f, **kwargs)


def dump_json(x: JSON, path: PathLike, **kwargs) -> None:
    """Dump an object to a file in JSON format.

    Args:
        x (`~zero.types.JSON`): the JSON-compatible object
        path (`~zero.types.PathLike`): path to the file. If the file doesn't exist,
            it will be created, otherwise, overwritten.
        kwargs: arguments for `json.dump`
    """
    with open(path, 'w') as f:
        json.dump(x, f, **kwargs)


def load_jsonl(path: PathLike, **kwargs) -> List[JSON]:
    """Load JSONL data from a file.

    Args:
        path (`~zero.types.PathLike`): path to the file. The file must exist.
        kwargs: arguments for `json.loads`
    Returns:
        List[`~zero.types.JSON`]: The data.
    """
    with open(path) as f:
        return [json.loads(x, **kwargs) for x in f]


def _extend_jsonl(records: Iterable[JSON], path: PathLike, mode: str, **kwargs) -> None:
    with open(path, mode) as f:
        for x in records:
            json.dump(x, f, **kwargs)
            f.write('\n')


def dump_jsonl(records: Iterable[JSON], path: PathLike, **kwargs) -> None:
    """Dump an object to a file in JSONL format.

    Args:
        records (Iterable[`~zero.types.JSON`]): the JSON-compatible records
        path (`~zero.types.PathLike`): path to the file. If the file doesn't exist,
            it will be created, otherwise, overwritten.
        kwargs: arguments for `json.dump`
    """
    _extend_jsonl(records, path, 'w', **kwargs)


def extend_jsonl(records: Iterable[JSON], path: PathLike, **kwargs) -> None:
    """Extend existing JSONL-file with new JSON-records.

    Args:
        records (Iterable[`~zero.types.JSON`]): the JSON-compatible records
        path (`~zero.types.PathLike`): path to the file. If the file doesn't exist,
            it will be created, otherwise, overwritten.
        kwargs: arguments for `json.dump`
    """
    _extend_jsonl(records, path, 'a', **kwargs)
