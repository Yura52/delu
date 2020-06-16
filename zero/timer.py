__all__ = ['Timer']

import math
import time
from typing import Optional, Union


class Timer:
    # mypy cannot infer types from .reset(), so they must be given here
    _start_time: Optional[float]
    _stop_time: Optional[float]
    _shift: float

    def __init__(self) -> None:
        self.reset()

    def start(self) -> 'Timer':
        if self._start_time is None:
            self._start_time = time.perf_counter()
        elif self._stop_time is not None:
            self.sub(time.perf_counter() - self._stop_time)
            self._stop_time = None
        return self  # enables the pattern `timer = Timer().start()`

    def stop(self) -> None:
        if self._stop_time is None:
            self._stop_time = time.perf_counter()

    def reset(self) -> None:
        self._start_time = None
        self._stop_time = None
        self._shift = 0.0

    def add(self, shift: float) -> None:
        self._shift += shift

    def sub(self, shift: float) -> None:
        self._shift -= shift

    def __call__(self, round_up: bool = False) -> Union[float, int]:
        if self._start_time is None:
            return 0 if round_up else 0.0
        now = self._stop_time or time.perf_counter()
        result = now - self._start_time + self._shift
        return math.ceil(result) if round_up else result
