"""Time management."""

__all__ = ['Timer', 'format_seconds']

import time
from typing import Optional, Union


class Timer:
    """Measures time.

    Measures time elapsed since the first call to `~Timer.run` up to "now" plus
    shift. The shift accumulates all pauses time and can be manually changed with the
    methods `~Timer.add` and `~Timer.sub`. If a timer is just created/reset, the shift
    is 0.0.

    Examples:
        .. testcode::

            timer = Timer()

    .. rubric:: Tutorial

    .. testcode::

        import time

        assert Timer()() == 0.0

        timer = Timer().run()  # start
        time.sleep(0.01)
        assert timer()  # some time has passed

        timer.pause()
        elapsed = timer()
        time.sleep(0.01)
        assert timer() == elapsed  # time didn't change because the timer is on pause

        timer.add(1.0)
        assert timer() == elapsed + 1.0

        timer.run()  # resume
        time.sleep(0.01)
        assert timer() > elapsed + 1.0

        timer.reset()
        assert timer() == 0.0
    """

    # mypy cannot infer types from .reset(), so they must be given here
    _start_time: Optional[float]
    _pause_time: Optional[float]
    _shift: float

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the timer.

        Resets the timer to the initial state.
        """
        self._start_time = None
        self._pause_time = None
        self._shift = 0.0

    def run(self) -> 'Timer':
        """Start/resume the timer.

        If the timer is on pause, the method resumes the timer.
        If the timer is runnning, the method does nothing (i.e. it does NOT overwrite
        the previous pause time).
        """
        if self._start_time is None:
            self._start_time = time.perf_counter()
        elif self._pause_time is not None:
            self.sub(time.perf_counter() - self._pause_time)
            self._pause_time = None
        return self  # enables the pattern `timer = Timer().run()`

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

    def add(self, delta: float) -> None:
        """Add non-negative delta to the shift.

        Args:
            delta
        Raises:
            AssertionError: if delta is negative

        Examples:
            .. testcode::

                timer = Timer()
                assert timer() == 0.0
                timer.add(1.0)
                assert timer() == 1.0
                timer.add(2.0)
                assert timer() == 3.0
        """
        assert delta >= 0
        self._shift += delta

    def sub(self, delta: float) -> None:
        """Subtract non-negative delta from the shift.

        Args:
            delta
        Raises:
            AssertionError: if delta is negative

        Examples:
            .. testcode::

                timer = Timer()
                assert timer() == 0.0
                timer.sub(1.0)
                assert timer() == -1.0
                timer.sub(2.0)
                assert timer() == -3.0
        """
        assert delta >= 0
        self._shift -= delta

    def __call__(self) -> float:
        """Get time elapsed since the start.

        If the timer is just created/reset, the shift is returned (can be negative!
        ).
        Otherwise, :code:`now - start_time + shift` is returned. The shift includes
        total pause time (including the current pause, if the timer is on pause) and
        all manipulations by `~Timer.add` and `~Timer.sub`.

        Returns:
            Time elapsed.
        """
        if self._start_time is None:
            return self._shift
        now = self._pause_time or time.perf_counter()
        return now - self._start_time + self._shift


def format_seconds(seconds: Union[int, float], format_str: str = '%Hh %Mm %Ss') -> str:
    """Format numeric seconds in a human-readable string.

    Args:
        seconds: seconds to format
        format_str: the format string passed to `time.strftime`
    Returns:
        Filled :code:`format_str`.

    Examples:
        .. testcode::

            assert format_seconds(3661) == '01h 01m 01s'
    """
    return time.strftime(format_str, time.gmtime(seconds))
