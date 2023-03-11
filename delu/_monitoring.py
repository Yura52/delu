import datetime
import enum
import time
from typing import Any, Dict, Optional


class _ProgressStatus(enum.Enum):
    NEUTRAL = enum.auto()
    SUCCESS = enum.auto()
    FAIL = enum.auto()


class ProgressTracker:
    """Facilitates early stopping and tracks the best metric value.

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
        """Initialize self.

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

        See also:
            `ProgressTracker.reset`
        """
        self._bad_counter = 0
        self._status = _ProgressStatus.NEUTRAL

    def reset(self) -> None:
        """Reset everything.

        See also:
            `ProgressTracker.forget_bad_updates`
        """
        self.forget_bad_updates()
        self._best_score = None


class Timer:
    """Measures time.

    Measures time elapsed since the first call to `~Timer.run` up to "now" minus pauses.

    Note:
        Measurements are performed via `time.perf_counter`.

    .. rubric:: Tutorial

    .. testcode::

        import time

        assert delu.Timer()() == 0.0

        timer = delu.Timer()
        timer.run()  # start
        time.sleep(0.01)
        assert timer()  # some time has passed

        timer.pause()
        elapsed = timer()
        time.sleep(0.01)
        assert timer() == elapsed  # time didn't change because the timer is on pause

        timer.run()  # resume
        time.sleep(0.01)
        assert timer() > elapsed

        timer.reset()
        assert timer() == 0.0

        with delu.Timer() as timer:
            time.sleep(0.01)
        # timer is on pause and timer() returns the time elapsed within the context

    `Timer` can be printed and formatted in a human-readable manner::

        timer = delu.Timer()
        timer.run()
        <let's assume that 3661 seconds have passed>
        print('Time elapsed:', timer)  # prints "Time elapsed: 1:01:01"
        assert str(timer) == f'{timer}' == '1:01:01'
        assert timer.format('%Hh %Mm %Ss') == '01h 01m 01s'

    `Timer` is pickle friendly:

    .. testcode::

        import pickle

        timer = delu.Timer()
        timer.run()
        time.sleep(0.01)
        timer.pause()
        old_value = timer()
        timer_bytes = pickle.dumps(timer)
        time.sleep(0.01)
        new_timer = pickle.loads(timer_bytes)
        assert new_timer() == old_value
    """

    # mypy cannot infer types from .reset(), so they must be given here
    _start_time: Optional[float]
    _pause_time: Optional[float]
    _shift: float

    def __init__(self) -> None:
        """Initialize self.

        Examples:
            .. testcode::

                timer = delu.Timer()
        """
        self.reset()

    def reset(self) -> None:
        """Reset the timer.

        Resets the timer to the initial state.
        """
        self._start_time = None
        self._pause_time = None
        self._shift = 0.0

    def run(self) -> None:
        """Start/resume the timer.

        If the timer is on pause, the method resumes the timer.
        If the timer is running, the method does nothing (i.e. it does NOT overwrite
        the previous pause time).
        """
        if self._start_time is None:
            self._start_time = time.perf_counter()
        elif self._pause_time is not None:
            self._shift -= time.perf_counter() - self._pause_time
            self._pause_time = None

    def pause(self) -> None:
        """Pause the timer.

        If the timer is running, the method pauses the timer.
        If the timer is already on pause, the method does nothing.

        Raises:
            AssertionError: if the timer is just created or just reset.
        """
        assert self._start_time is not None
        if self._pause_time is None:
            self._pause_time = time.perf_counter()

    def __call__(self) -> float:
        """Get the time elapsed since the start.

        Returns:
            Time elapsed.
        """
        if self._start_time is None:
            return self._shift
        now = self._pause_time or time.perf_counter()
        return now - self._start_time + self._shift

    def __str__(self) -> str:
        """Convert the timer to a string.

        Returns:
            The string representation of the timer's value rounded to seconds.
        """
        return str(datetime.timedelta(seconds=round(self())))

    def format(self, format_str: str) -> str:
        """Format the time elapsed since the start in a human-readable string.

        Args:
            format_str: the format string passed to `time.strftime`
        Returns:
            Filled ``format_str``.

        Example::

            timer = delu.Timer()
            <let's assume that 3661 seconds have passed>
            assert timer.format('%Hh %Mm %Ss') == '01h 01m 01s'
        """
        return time.strftime(format_str, time.gmtime(self()))

    def __enter__(self) -> 'Timer':
        """Measure time within a context.

        The method `Timer.run` is called regardless of the current state. On exit,
        `Timer.pause` is called.

        See also:
            `Timer.__exit__`

        Example:
            ..testcode::

                import time
                with delu.Timer() as timer:
                    time.sleep(0.01)
                elapsed = timer()
                assert elapsed > 0.01
                time.sleep(0.01)
                assert timer() == elapsed  # the timer is paused in __exit__
        """
        self.run()
        return self

    def __exit__(self, *args) -> bool:  # type: ignore
        """Leave the context and pause the timer.

        See `Timer.__enter__` for details and examples.

        See also:
            `Timer.__enter__`
        """
        self.pause()
        return False

    def __getstate__(self) -> Dict[str, Any]:
        return {'_shift': self(), '_start_time': None, '_pause_time': None}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
