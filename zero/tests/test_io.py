import tempfile
from pathlib import Path

import zero.io as zio


def test_io():
    data = {'a': ['b', 1], 'c': {'d': 2}, 'e': None}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'whatever'

        zio.dump_pickle(data, path)
        assert zio.load_pickle(path) == data

        zio.dump_json(data, path)
        assert zio.load_json(path) == data

        zio.dump_jsonl([data], path)
        assert zio.load_jsonl(path) == [data]

        zio.dump_jsonl([data], path)
        zio.extend_jsonl([data], path)
        assert zio.load_jsonl(path) == [data, data]
