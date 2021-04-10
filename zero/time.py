"""Time management."""

__all__ = ['Timer']

import time
from typing import Any, Dict, Optional


class Timer:
    """Measures time.

    Measures time elapsed since the first call to `~Timer.run` up to "now" plus
    shift. The shift accumulates all pauses time and can be manually changed with the
    methods `~Timer.add` and `~Timer.sub`. If a timer is just created/reset, the shift
    is 0.0.

    Note:
        Measurements are performed via `time.perf_counter`.

    Examples:
        .. testcode::

            timer = Timer()

    .. rubric:: Tutorial

    .. testcode::

        import time

        assert Timer()() == 0.0

        timer = Timer()
        timer.run()  # start
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

        with Timer() as timer:
            time.sleep(0.01)
        # timer is on pause and timer() returns the time elapsed within the context

    Note:
        When a Timer instance is pickled, the result of `Timer.__call__` is saved as
        shift (hence, the "time elapsed" is preserved) and all other attributes are
        omitted.
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

    def run(self) -> None:
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

        If the timer is just created/reset, the shift is returned (can be negative!).
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

    def format(self, format_str: str = '%Hh %Mm %Ss') -> str:
        """Format the time elapsed since the start in a human-readable string.

        Args:
            format_str: the format string passed to `time.strftime`
        Returns:
            Filled :code:`format_str`.

        Examples:
            .. testcode::

                timer = Timer()
                timer.add(3661)
                assert timer.format() == '01h 01m 01s'
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
                with Timer() as timer:
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
