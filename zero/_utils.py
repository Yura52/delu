import datetime
import enum
import secrets
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from . import random as zero_random


class _ProgressStatus(enum.Enum):
    NEUTRAL = enum.auto()
    SUCCESS = enum.auto()
    FAIL = enum.auto()


class ProgressTracker:
    """Tracks the best score, helps with early stopping.

    For `~ProgressTracker`, **the greater score is the better score**.
    At any moment the tracker is in one of the following states:

    - *success*: the last update changed the best score
    - *fail*: last :code:`n > patience` updates are not better than the best score
    - *neutral*: if neither success nor fail

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

    def __init__(self, patience: Optional[int], min_delta: float = 0.0) -> None:
        """Initialize self.

        Args:
            patience: Allowed number of bad updates. For example, if patience is 2, then
                2 bad updates is not a fail, but 3 bad updates is a fail. If `None`,
                then the progress tracker never fails.
            min_delta: minimal improvement over current best score to count it as
                success.

        Examples:
            .. testcode::

                progress = ProgressTracker(2)
                progress = ProgressTracker(3, 0.1)
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
        """Check if the tracker is in the 'success' state."""
        return self._status == _ProgressStatus.SUCCESS

    @property
    def fail(self) -> bool:
        """Check if the tracker is in the 'fail' state."""
        return self._status == _ProgressStatus.FAIL

    def _set_success(self, score: float) -> None:
        self._best_score = score
        self._status = _ProgressStatus.SUCCESS
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
                _ProgressStatus.FAIL
                if self._patience is not None and self._bad_counter > self._patience
                else _ProgressStatus.NEUTRAL
            )

    def forget_bad_updates(self) -> None:
        """Reset bad updates and status, but not the best score."""
        self._bad_counter = 0
        self._status = _ProgressStatus.NEUTRAL

    def reset(self) -> None:
        """Reset everything."""
        self.forget_bad_updates()
        self._best_score = None


class Timer:
    """Measures time.

    Measures time elapsed since the first call to `~Timer.run` up to "now" plus
    shift. The shift accumulates all pauses time and can be manually changed with the
    methods `~Timer.add` and `~Timer.sub`. If a timer is just created/reset, the shift
    is 0.0.

    Note:
        Measurements are performed via `time.perf_counter`.

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

    `Timer` can be printed and formatted in a human-readable manner:

    .. testcode::

        timer = Timer()
        timer.add(3661)
        print('Time elapsed:', timer)
        assert str(timer) == f'{timer}' == '1:01:01'
        assert timer.format('%Hh %Mm %Ss') == '01h 01m 01s'

    .. testoutput::

        Time elapsed: 1:01:01

    `Timer` is pickle friendly:

    .. testcode::

        import pickle

        timer = Timer()
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

                timer = Timer()
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
            Filled :code:`format_str`.

        Examples:
            .. testcode::

                timer = Timer()
                timer.add(3661)
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


class evaluation(torch.no_grad):
    """Context-manager & decorator for models evaluation.

    This code... ::

        with evaluation(model):
            ...

        @evaluation(model)
        def f():
            ...

    ...is equivalent to the following: ::

        with torch.no_grad():
            model.eval()
            ...

        @torch.no_grad()
        def f():
            model.eval()
            ...

    Args:
        modules

    Note:
        The training status of modules is undefined once a context is finished or a
        decorated function returns.

    Warning:
        The function must be used in the same way as `torch.no_grad`, i.e. only as a
        context manager or a decorator as shown below in the examples. Otherwise, the
        behaviour is undefined.

    Examples:
        .. testcode::

            a = torch.nn.Linear(1, 1)
            b = torch.nn.Linear(2, 2)
            with evaluation(a):
                ...
            with evaluation(a, b):
                ...

            @evaluation(a)
            def f():
                ...

            @evaluation(a, b)
            def f():
                ...

        .. testcode::

            model = torch.nn.Linear(1, 1)
            for grad_before_context in False, True:
                for train in False, True:
                    torch.set_grad_enabled(grad_before_context)
                    model.train(train)
                    with evaluation(model):
                        assert not model.training
                        assert not torch.is_grad_enabled()
                        ...
                    assert torch.is_grad_enabled() == grad_before_context
                    # model.training is unspecified here
    """

    def __init__(self, *modules: nn.Module) -> None:
        assert modules
        self._modules = modules

    def __enter__(self) -> None:
        result = super().__enter__()
        for m in self._modules:
            m.eval()
        return result


def improve_reproducibility(seed: Optional[int]) -> int:
    """Set seeds in `random`, `numpy` and `torch` and make some cuDNN operations deterministic.

    Do everything possible to improve reproducibility for code that relies on global
    random number generators from the aforementioned modules. See also the note below.

    Sets:

    1. seeds in `random`, `numpy.random`, `torch`, `torch.cuda`
    2. `torch.backends.cudnn.benchmark` to `False`
    3. `torch.backends.cudnn.deterministic` to `True`

    Args:
        seed: the seed for all mentioned libraries. Must be less than
            :code:`2 ** 32 - 1`. If `None`, a high-quality seed is generated instead.

    Returns:
        seed: if :code:`seed` is set to `None`, the generated seed is returned;
            otherwise, :code:`seed` is returned as is

    Note:
        If you don't want to set the seed by hand, but still want to have a chance to
        reproduce things, you can use the following pattern::

            print('Seed:', zero.improve_reproducibility(None))

    Note:
        100% reproducibility is not always possible in PyTorch. See
        `this page <https://pytorch.org/docs/stable/notes/randomness.html>`_ for
        details.

    Examples:
        .. testcode::

            assert zero.improve_reproducibility(0) == 0
            seed = zero.improve_reproducibility(None)
    """
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    if seed is None:
        # See https://numpy.org/doc/1.18/reference/random/bit_generators/index.html#seeding-and-entropy  # noqa
        seed = secrets.randbits(128) % (2 ** 32 - 1)
    else:
        assert seed < (2 ** 32 - 1)
    zero_random.seed(seed)
    return seed
