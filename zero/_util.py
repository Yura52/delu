from collections.abc import Mapping, Sequence


def to_list(x):
    return x if isinstance(x, list) else [x]


def flatten(data):
    if isinstance(data, (str, bytes)):
        yield data
    elif isinstance(data, Sequence):
        for x in data:
            yield from flatten(x)
    elif isinstance(data, Mapping):
        for x in data.values():
            yield from flatten(x)
    else:
        yield data


def traverse(fn, data):
    if isinstance(data, (str, bytes)):
        return fn(data)
    elif isinstance(data, tuple):
        is_namedtuple = all(
            hasattr(data, x) for x in ['_make', '_asdict', '_replace', '_fields']
        )
        return (type(data)._make if is_namedtuple else type(data))(
            traverse(fn, x) for x in data
        )
    elif isinstance(data, list):
        return type(data)(traverse(fn, x) for x in data)
    elif isinstance(data, dict):
        return type(data)({k: traverse(fn, v) for k, v in data.items()})
    else:
        return fn(data)
