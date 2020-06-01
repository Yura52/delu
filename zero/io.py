import json
import pickle


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(x, path):
    with open(path, 'wb') as f:
        return pickle.dump(x, f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def dump_json(x, path, **kwargs):
    with open(path, 'w') as f:
        json.dump(x, f, **kwargs)


def load_jsonl(path):
    with open(path) as f:
        return list(map(json.loads, f))


def _extend_jsonl(records, path, mode, **kwargs):
    with open(path, mode) as f:
        for x in records:
            json.dump(x, f, **kwargs)
            f.write('\n')


def extend_jsonl(records, path, **kwargs):
    _extend_jsonl(records, path, 'a', **kwargs)


def dump_jsonl(records, path, **kwargs):
    _extend_jsonl(records, path, 'w', **kwargs)
