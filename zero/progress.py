__all__ = ['LossScore', 'ProgressTracker']

import enum
from typing import Callable, Optional


class LossScore:
    def __init__(self, loss_fn: Callable[[], float]) -> None:
        self._loss_fn = loss_fn

    def loss(self) -> float:
        return self._loss_fn()

    def score(self) -> float:
        return -self.loss()


class _Status(enum.Enum):
    NEUTRAL = enum.auto()
    SUCCESS = enum.auto()
    FAIL = enum.auto()


class ProgressTracker:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self._patience = patience
        self._min_delta = float(min_delta)
        self._best_score: Optional[float] = None
        self._status = _Status.NEUTRAL
        self._bad_counter = 0

    @property
    def best_score(self) -> Optional[float]:
        return self._best_score

    @property
    def success(self) -> bool:
        return self._status == _Status.SUCCESS

    @property
    def fail(self) -> bool:
        return self._status == _Status.FAIL

    def _set_success(self, score: float) -> None:
        self._best_score = score
        self._status = _Status.SUCCESS
        self._bad_counter = 0

    def update(self, score: float) -> None:
        if self._best_score is None:
            self._set_success(score)
        elif score > self._best_score + self._min_delta:
            self._set_success(score)
        else:
            self._bad_counter += 1
            self._status = (
                _Status.FAIL if self._bad_counter > self._patience else _Status.NEUTRAL
            )

    def forget_bad_updates(self) -> None:
        self._bad_counter = 0
        self._status = _Status.NEUTRAL

    def reset(self) -> None:
        self.forget_bad_updates()
        self._best_score = None
