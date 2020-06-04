import math
import time
from typing import Union

_CONTEXT_MANAGER_HELP = (
    'Timer can be used as a context manager only using the .pause method:\n'
    '    timer = Timer().start()\n'
    '    ...\n'
    '    with timer.pause():\n'
    '        ...'
)


class Timer:
    def __init__(self):
        self.reset()

    def start(self):
        assert self._start_time is None
        self._start_time = time.perf_counter()
        return self  # enables the pattern `timer = Timer().start()`

    def __call__(self, round_up: bool = False) -> Union[float, int]:
        if self._start_time is None:
            return 0 if round_up else 0.0
        now = self._pause_start_time or time.perf_counter()
        result = now - self._start_time + self._shift
        return math.ceil(result) if round_up else result

    def add(self, shift: float):
        self._shift += shift

    def sub(self, shift: float):
        self._shift -= shift

    def reset(self):
        self._start_time = None
        self._pause_start_time = None
        self._shift = 0.0

    def pause(self) -> 'Timer':
        assert self._pause_start_time is None
        self._pause_start_time = time.perf_counter()
        return self

    def resume(self):
        assert self._pause_start_time is not None
        self.sub(time.perf_counter() - self._pause_start_time)
        self._pause_start_time = None

    def __enter__(self):
        assert self._pause_start_time is not None, _CONTEXT_MANAGER_HELP

    def __exit__(self, *args, **kwargs):
        assert self._pause_start_time is not None, _CONTEXT_MANAGER_HELP
        self.resume()
