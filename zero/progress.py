"""Early stopping, tracking the best score, etc."""

__all__ = ['ProgressTracker']

import enum
from typing import Optional


class _Status(enum.Enum):
    NEUTRAL = enum.auto()
    SUCCESS = enum.auto()
    FAIL = enum.auto()


class ProgressTracker:
    """Tracks the best score, facilitates early stopping.

    For `~ProgressTracker`, **the greater score is the better score**.
    At any moment the tracker is in one of the following states:
    - success: the last score updated the best score
    - fail: last :code:`n > patience` updates are not better than the best score
    - neutral: if neither success nor fail

    Args:
        patience: Allowed number of bad updates. For example, if patience is 2, then
            2 bad updates is not a fail, but 3 bad updates is a fail.
        min_delta: minimal improvement over current best score to count it as success.

    Examples:
        .. testcode::

            progress = ProgressTracker(2)
            progress = ProgressTracker(3, 0.1)

    .. rubric:: Tutorial

    .. testcode::

        progress = ProgressTracker(2)
        progress.update(-999999999)
        assert progress.success  # the first update always updates the best score

        progress.update(123)
        assert progress.success
        assert progress.best_score == 123

        progress.update(0)
        assert not progress.success and not progress.fail

        progress.update(123)
        assert not progress.success and not progress.fail
        progress.update(123)
        # patience is 2 and the best score is not updated for more than 2 steps
        assert progress.fail
        assert progress.best_score == 123  # fail doesn't affect the best score
        progress.update(123)
        assert progress.fail  # still no improvements

        progress.forget_bad_updates()
        assert not progress.fail and not progress.success
        assert progress.best_score == 123
        progress.update(0)
        assert not progress.fail  # just 1 bad update (the patience is 2)

        progress.reset()
        assert not progress.fail and not progress.success
        assert progress.best_score is None
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self._patience = patience
        self._min_delta = float(min_delta)
        self._best_score: Optional[float] = None
        self._status = _Status.NEUTRAL
        self._bad_counter = 0

    @property
    def best_score(self) -> Optional[float]:
        """The best score so far.

        If the tracker is just created/reset, return `None`.
        """
        return self._best_score

    @property
    def success(self) -> bool:
        """Check if the tracker is in the 'success' state."""
        return self._status == _Status.SUCCESS

    @property
    def fail(self) -> bool:
        """Check if the tracker is in the 'fail' state."""
        return self._status == _Status.FAIL

    def _set_success(self, score: float) -> None:
        self._best_score = score
        self._status = _Status.SUCCESS
        self._bad_counter = 0

    def update(self, score: float) -> None:
        """Update the tracker's state.

        Args:
            score: the score to use for the update.
        """
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
        """Reset bad updates and status, but not the best score."""
        self._bad_counter = 0
        self._status = _Status.NEUTRAL

    def reset(self) -> None:
        """Reset everything."""
        self.forget_bad_updates()
        self._best_score = None
