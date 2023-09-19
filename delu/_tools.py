import datetime
import enum
import time
from typing import Any, Dict, Literal, Optional

from ._utils import deprecated


class EarlyStopping:
    """Performs early stopping after N consequtive non-improving updates.

    **Usage**

    Preventing overfitting by stopping the training
    when the validation metric stops improving:

    >>> def evaluate_model() -> float:
    ...     # A dummy example.
    ...     return torch.rand(1).item()
    ...
    >>> # If the validation score does not increase (mode='max')
    >>> # for 10 (patience=10) epochs in a row, stop the training.
    >>> early_stopping = delu.EarlyStopping(patience=10, mode='max')
    >>> for epoch in range(1000):
    ...     # Training.
    ...     ...
    ...     # Evaluation
    ...     validation_score = evaluate_model()
    ...     ...
    ...     # Submit the new score.
    ...     early_stopping.update(validation_score)
    ...     # Check whether the training should stop.
    ...     if early_stopping.should_stop():
    ...         break

    Additional technical examples:

    >>> early_stopping = delu.EarlyStopping(2, mode='max')
    >>> # Format: (<the best seen score>, <the number of consequtive fails>)
    >>> early_stopping.update(1.0)  # (1.0, 0)
    >>> early_stopping.should_stop()
    False
    >>> early_stopping.update(0.0)  # (1.0, 1)
    >>> early_stopping.should_stop()
    False
    >>> early_stopping.update(2.0)  # (2.0, 0)
    >>> early_stopping.update(1.0)  # (2.0, 1)
    >>> early_stopping.update(2.0)  # (2.0, 2)
    >>> early_stopping.should_stop()
    True

    Resetting the number of consequtive non-improving updates
    without resetting the best seen score:

    >>> early_stopping.forget_bad_updates()  # (2.0, 0)
    >>> early_stopping.should_stop()
    False
    >>> early_stopping.update(0.0)  # (2.0, 1)
    >>> early_stopping.update(0.0)  # (2.0, 2)
    >>> early_stopping.should_stop()
    True

    The next successfull update resets the number of consequtive fails:

    >>> early_stopping.update(0.0)  # (2.0, 3)
    >>> early_stopping.should_stop()
    True
    >>> early_stopping.update(3.0)  # (3.0, 0)
    >>> early_stopping.should_stop()
    False

    It is possible to completely reset the instance:

    >>> early_stopping.reset()  # (-inf, 0)
    >>> early_stopping.should_stop()
    False
    >>> early_stopping.update(-10.0)   # (-10.0, 0)
    >>> early_stopping.update(-100.0)  # (-10.0, 1)
    >>> early_stopping.update(-10.0)   # (-10.0, 2)
    >>> early_stopping.should_stop()
    True
    """

    def __init__(
        self, patience: int, *, mode: Literal['min', 'max'], min_delta: float = 0.0
    ) -> None:
        """
        Args:
            patience: when the number of the latest consequtive bad updates reaches
                ``patience``, `EarlyStopping.should_stop` starts returning `True`
                until the next good update.
            mode: if "min", then the update rule is "the lower value is the better
                value". For "max", it is the opposite.
            min_delta: a new value must differ from the current best value by more
                than ``min_delta`` to be considered as an improvement.
        """
        if patience < 1:
            raise ValueError(
                f'patience must be a positive integer (the provided value: {patience}).'
            )
        if mode not in ('min', 'max'):
            raise ValueError(
                f'mode must be either "min" or "max" (the provided value: "{mode}").'
            )
        if min_delta < 0.0:
            raise ValueError(
                'min_delta must be a non-negative number'
                f' (the provided value: {min_delta}).'
            )
        self._patience = patience
        self._maximize = mode == 'max'
        self._min_delta = min_delta
        self._best_value: Optional[float] = None
        self._n_consequtive_bad_updates = 0

    def reset(self) -> None:
        """Reset everything."""
        self._best_value = None
        self._n_consequtive_bad_updates = 0

    def forget_bad_updates(self) -> None:
        """Reset the number of consecutive non-improving updates to zero.

        Note that this method does NOT reset the best seen score.
        """
        self._n_consequtive_bad_updates = 0

    def should_stop(self) -> bool:
        """Check whether the early stopping condition is activated.

        See examples in `EarlyStopping`.

        Returns:
            `True` if the number of consequtive bad updates has reached the patience.
            `False` otherwise.
        """
        return self._n_consequtive_bad_updates >= self._patience

    def update(self, value: float) -> None:
        """Submit a new value.

        Args:
            value: the new value.
        """
        success = (
            True
            if self._best_value is None
            else value > self._best_value + self._min_delta
            if self._maximize
            else value < self._best_value - self._min_delta
        )
        if success:
            self._best_value = value
            self._n_consequtive_bad_updates = 0
        else:
            self._n_consequtive_bad_updates += 1


class Timer:
    """A simple pickle-friendly timer.

    - `Timer` measures time, can be paused, resumed and used as a context manager.
    - `Timer` **is pickle-friendly and can saved to / loaded from a checkpoint.**
    - `Timer` can report the elapsed time as a (customizable) human-readable string.

    .. note::
        Time measurements are performed with `time.perf_counter`.

    **Usage**

    Creating and running a timer:

    >>> timer = delu.Timer()
    >>> timer.run()

    Call the timer to get the elapsed time. The elapsed time is
    the time passed since the first ``.run()`` call up to now, minus pauses.

    >>> # elapsed = <now> - <time of the first .run()> - <total pause duration>
    >>> import time
    >>> def job():
    ...     time.sleep(0.01)
    ...
    >>> timer = delu.Timer()
    >>> timer.run()
    >>> job()
    >>> # Check that some time has passed.
    >>> timer() > 0.0
    True

    A timer can be paused:

    >>> timer.pause()
    >>> elapsed = timer()
    >>> job()
    >>> # The elapsed time has not changed,
    >>> # because the timer is on pause.
    >>> timer() == elapsed
    True
    >>> # Resume the timer.
    >>> timer.run()
    >>> job()
    >>> timer() > elapsed
    True

    Measuring time between two events:

    >>> start = timer()
    >>> job()
    >>> end = timer()
    >>> duration = end - start

    Measuring time of a code block:

    >>> # When entering the context, .run() is automatically called.
    >>> with delu.Timer() as timer:
    ...     job()
    >>> elapsed = timer()
    >>> elapsed > 0.0
    True
    >>> # On exit, the timer is set on pause.
    >>> job()
    >>> timer() == elapsed
    True

    Resetting the timer:

    >>> timer.reset()
    >>> timer() == 0.0
    True

    Printing and formatting the elapsed time in a human-readable manner:

    >>> timer = delu.Timer()
    >>> timer.run()
    >>> time.sleep(2.0)
    >>> timer.pause()
    >>> # Instead of '...' there are microseconds,
    >>> # for example "0:00:02.005158"
    >>> print(timer)
    0:00:02...
    >>> str(timer)
    '0:00:02...
    >>> # Equivalent to the above.
    >>> f'{timer}'
    '0:00:02...
    >>> # Custom format string.
    >>> timer.format('%Hh %Mm %Ss')
    '00h 00m 02s'

    Saving (pickling) and loading (unpickling) a timer.

    >>> import pickle
    >>> timer = delu.Timer()
    >>> timer.run()
    >>> job()
    >>> timer.pause()
    >>> elapsed_before_saving = timer()
    >>> # Save the timer (typically, as a part of a checkpoint).
    >>> timer_bytes = pickle.dumps(timer)
    >>> ...
    >>> # Some time has passed...
    >>> ...
    >>> # The loaded timer remembers the previous elapsed time.
    >>> loaded_timer = pickle.loads(timer_bytes)
    >>> loaded_timer() == elapsed_before_saving
    True
    >>> # The loaded timer is not running, call .run() to resume it.
    >>> job()
    >>> loaded_timer() == elapsed_before_saving
    >>> loaded_timer.run()
    >>> loaded_timer() > elapsed_before_saving
    True
    """

    # mypy cannot infer types from .reset(), so they must be given here
    _start_time: Optional[float]
    _pause_time: Optional[float]
    _shift: float

    def __init__(self) -> None:
        """
        Args:
        """
        self.reset()

    def reset(self) -> None:
        """Reset the timer completely.

        To start using the instance again after resetting,
        the timer must be explicitly run with `Timer.run`.
        """
        self._start_time = None
        self._pause_time = None
        self._shift = 0.0

    def run(self) -> None:
        """Start/resume the timer.

        If the timer is on pause, the method resumes the timer.
        If the timer is running, the method does nothing.
        """
        if self._start_time is None:
            self._start_time = time.perf_counter()
        elif self._pause_time is not None:
            self._shift -= time.perf_counter() - self._pause_time
            self._pause_time = None

    def pause(self) -> None:
        """Pause the timer.

        If the timer is running, the method pauses the timer.
        If the timer was never ``.run()`` or is already on pause,
        the method does nothing.
        """
        if self._start_time is not None:
            if self._pause_time is None:
                self._pause_time = time.perf_counter()

    def __call__(self) -> float:
        """Get the time elapsed.

        Returns:
            The elapsed time.
        """
        if self._start_time is None:
            return self._shift
        now = self._pause_time or time.perf_counter()
        return now - self._start_time + self._shift

    def __str__(self) -> str:
        """Convert the timer to a string.

        Returns:
            The string representation of the timer's value.
        """
        return str(datetime.timedelta(seconds=self()))

    def format(self, format_str: str, /) -> str:
        """Format the time elapsed since the start in a human-readable string.

        This is a shortcut for ``time.strftime(format_str, time.gmtime(self()))``.

        Args:
            format_str: the format string passed to `time.strftime`.
        Returns:
            the filled ``format_str``.

        **Usage**

        >>> # xdoctest: +SKIP
        >>> timer = delu.Timer()
        >>> # Let's say that exactly 3661 seconds have passed.
        >>> assert timer.format('%Hh %Mm %Ss') == '01h 01m 01s'
        """
        return time.strftime(format_str, time.gmtime(self()))

    def __enter__(self) -> 'Timer':
        """Measure time within a context.

        The method `Timer.run` is called regardless of the current state.
        On exit, `Timer.pause` is called.
        """
        self.run()
        return self

    def __exit__(self, *args) -> bool:  # type: ignore
        """Leave the context and pause the timer."""
        self.pause()
        return False

    def __getstate__(self) -> Dict[str, Any]:
        return {'_shift': self(), '_start_time': None, '_pause_time': None}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Load the state.

        A time with just loaded state is not running (basically, it is a freshly
        created timer which stores the elapsed time from the loaded state).
        """
        self.__dict__.update(state)


class _ProgressStatus(enum.Enum):
    NEUTRAL = enum.auto()
    SUCCESS = enum.auto()
    FAIL = enum.auto()


@deprecated('Instead, use `delu.EarlyStopping` and manually track the best score.')
class ProgressTracker:
    """Helps with early stopping and tracks the best metric value.

    ⚠️ **DEPRECATED** ⚠️ <DEPRECATION MESSAGE>

    For `~ProgressTracker`, **the greater score is the better score**.
    At any moment the tracker is in one of the following states:

    - *success*: the last update increased the best score
    - *fail*: last ``n > patience`` updates did not improve the best score
    - *neutral*: if neither success nor fail

    .. rubric:: Tutorial

    .. testcode::

        progress = delu.ProgressTracker(2)
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

    def __init__(self, patience: Optional[int], min_delta: float = 0.0) -> None:
        """
        Args:
            patience: Allowed number of unsuccessfull updates. For example, if patience
                is 2, then 2 unsuccessfull updates in a row is not a fail,
                but 3 unsuccessfull updates in a row is a fail.
                `None` means "infinite patience" and the progress tracker is never
                in the "fail" state.
            min_delta: the minimal improvement over the current best score
                to count it as success.

        Examples:
            .. testcode::

                progress = delu.ProgressTracker(2)
                progress = delu.ProgressTracker(3, 0.1)
        """
        self._patience = patience
        self._min_delta = float(min_delta)
        self._best_score: Optional[float] = None
        self._status = _ProgressStatus.NEUTRAL
        self._bad_counter = 0

    @property
    def best_score(self) -> Optional[float]:
        """The best score so far.

        If the tracker is just created/reset, return `None`.
        """
        return self._best_score

    @property
    def success(self) -> bool:
        """Check if the tracker is in the "success" state."""
        return self._status == _ProgressStatus.SUCCESS

    @property
    def fail(self) -> bool:
        """Check if the tracker is in the "fail" state."""
        return self._status == _ProgressStatus.FAIL

    def _set_success(self, score: float) -> None:
        self._best_score = score
        self._status = _ProgressStatus.SUCCESS
        self._bad_counter = 0

    def update(self, score: float) -> None:
        """Submit a new score and update the tracker's state accordingly.

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
                _ProgressStatus.FAIL
                if self._patience is not None and self._bad_counter > self._patience
                else _ProgressStatus.NEUTRAL
            )

    def forget_bad_updates(self) -> None:
        """Reset unsuccessfull update counter and set the status to "neutral".

        Note that this method does NOT reset the best score.
        """
        self._bad_counter = 0
        self._status = _ProgressStatus.NEUTRAL

    def reset(self) -> None:
        """Reset everything."""
        self.forget_bad_updates()
        self._best_score = None
