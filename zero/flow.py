import math
from typing import Any, Iterable, Iterator, Optional, Sized, Union


def _try_len(x):
    return len(x) if isinstance(x, Sized) else None


class Flow:
    class _EpochData:
        def __init__(self, flow, n_iterations, increment_iteration):
            self._flow = flow
            self._start_iteration = flow.iteration
            self._n_iterations = n_iterations
            self._increment_iteration = increment_iteration

        def __iter__(self):
            return self

        def __next__(self):
            flow = self._flow
            if (
                self._n_iterations is not None
                and flow.iteration - self._start_iteration >= self._n_iterations
            ):
                raise StopIteration()
            return flow.next(self._increment_iteration)

    def __init__(self, loader: Iterable) -> None:
        assert _try_len(loader) != 0
        self._epoch = 0
        self._iteration = 0
        self._count = 0
        self._loader = loader
        self._iter: Optional[Iterator] = None

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def count(self) -> int:
        return self._count

    @property
    def loader(self) -> Iterable:
        return self._loader

    def increment_epoch(self, max: Union[None, int, float] = None) -> bool:
        if isinstance(max, float):
            assert math.isinf(max)
        should_increment = max is None or self.epoch < max
        if should_increment:
            self._epoch += 1
        return should_increment

    def increment_iteration(self) -> None:
        self._iteration += 1

    def _increment_count(self):
        self._count += 1

    def data(
        self,
        n_iterations: Union[None, int, float] = None,
        increment_iteration: bool = True,
    ) -> Iterator:
        if isinstance(n_iterations, float):
            assert math.isinf(n_iterations)
        if n_iterations is None:
            if not isinstance(self.loader, Sized):
                raise ValueError()
            n_iterations = len(self.loader)
        return Flow._EpochData(self, n_iterations, increment_iteration)

    def next(self, increment_iteration: bool = True) -> Any:
        if self._iter is None:
            self._iter = iter(self._loader)
        try:
            value = next(self._iter)
        except StopIteration:
            self.reset_iterator()
            # If the following line raises StopIteration too, then the data is over
            # and the exception should be just propagated.
            value = next(self._iter)
        if increment_iteration:
            self.increment_iteration()
        self._increment_count()
        return value

    def reset_iterator(self) -> None:
        self._iter = iter(self._loader)

    def set_loader(self, loader: Iterable) -> None:
        assert _try_len(loader) != 0
        self._loader = loader
        if self._iter is not None:
            self._iter = iter(loader)
