import json
import pickle
from typing import Any, Iterable, List

from .types import JSON, PathLike


def load_pickle(path: PathLike) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(x, path: PathLike) -> None:
    with open(path, 'wb') as f:
        pickle.dump(x, f)


def load_json(path: PathLike) -> JSON:
    with open(path) as f:
        return json.load(f)


def dump_json(x, path: PathLike, **kwargs) -> None:
    with open(path, 'w') as f:
        json.dump(x, f, **kwargs)


def load_jsonl(path: PathLike) -> List[JSON]:
    with open(path) as f:
        return list(map(json.loads, f))


def _extend_jsonl(records: Iterable[JSON], path: PathLike, mode: str, **kwargs) -> None:
    with open(path, mode) as f:
        for x in records:
            json.dump(x, f, **kwargs)
            f.write('\n')


def extend_jsonl(records: Iterable[JSON], path: PathLike, **kwargs) -> None:
    _extend_jsonl(records, path, 'a', **kwargs)


def dump_jsonl(records: Iterable[JSON], path: PathLike, **kwargs) -> None:
    _extend_jsonl(records, path, 'w', **kwargs)
