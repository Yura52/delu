"""Smart Python loops."""

__all__ = ['Stream', 'ManualStream']

import math
from typing import Any, Iterable, Iterator, Optional, Sized, Union


def _try_len(x):
    return len(x) if isinstance(x, Sized) else None


class Stream:
    """Smart wrapper for iterables.

    `Stream` simplifies managing loops, especially in typical deep learning scenarios
    (it is usually used to wrap :code:`train_dataloader` or any other data source).

    `Stream`:

    - simplifies management of the "epoch" and "iteration" variables
    - allows to customize the size of epoch
    - allows to change the underlying data loader on the fly
    - enables useful patterns
    - (not implemented: `issue <https://github.com/Yura52/zero/issues/6>`_) allows to
      dump and restore loop's state: epoch, iteration, etc.

    Args:
        loader: any kind of iterable (DataLoader, list, iterator, generator, ...)
    Raises:
        AssertionError: if :code:`loader` is not an iterator and is empty

    Examples:
        .. testcode::

            stream = Stream([0, 1, 2, 3])
            stream = Stream(range(10))
            import itertools
            stream = Stream(itertools.repeat(0))

            from torch.utils.data import DataLoader, TensorDataset
            dataset = TensorDataset(torch.randn(10, 2))
            stream = Stream(DataLoader(dataset, batch_size=3, shuffle=True))

    .. rubric:: Tutorial

    Let's revise the conventional approach without `Stream`:

    .. code-block::

        loader = DataLoader(...)
        iteration = 0
        for epoch in range(n_epoches):
            if need_custom_epoch_size():
                assert False, 'It is possible, but not convenient'

            for x in loader:
                iteration += 1
                print('Epoch:', epoch, 'Iteration:', iteration)
                ...

            if need_new_loader():
                assert False, 'It is possible, but not convenient'

    There are several ways how you can use `Stream` to enhance this loop. Let's start
    with creating a stream:

    .. code-block::

        stream = Stream(DataLoader(...))

    The dataloader is accessible via `Stream.loader`. Now, let's reproduce the loop
    above:

    .. code-block::

        for epoch in range(n_epoches):
            for x in stream.data():
                print('Epoch:', epoch, 'Iteration:', stream.iteration)

        # or

        while stream.increment_epoch(n_epoches):
            for x in stream.data():
                print('Epoch:', stream.epoch, 'Iteration:', stream.iteration)

    Firstly, we see that `Stream.iteration` is created and incremented automatically.
    We also see that :code:`while` loop can be used instead of more "conventional"
    :code:`for`. It brings the following differences:

    - restoring stream's state via the :code:`state_dict` mechanism becomes possible
    - terminating the loop by adding more conditions to the :code:`while` statement
      becomes possible; for example, with `zero.training.ProgressTracker` early stopping
      can look like this:

      .. code-block::

          while not progress.fail and stream.increment_epoch(n_epoches):

    - epoches numeration effectively starts from 1; it is consistent with iterations
      numeration (also starts from 1)

    In order to customize the epoch size, pass the size to `Stream.data`:

    .. code-block::

        while stream.increment_epoch(n_epoches):
            for x in stream.data(custom_epoch_size):
                ...

    Changing the underlying loader on the fly is possible at *any* moment (even in the
    middle of epoch) via `Stream.set_loader`. For example::

        while stream.increment_epoch(n_epoches):
            for x in stream.data(custom_epoch_size):
                ...
                if need_new_loader():
                    stream.set_loader(new_loader)

    Additionally, two new forms of infinite loop become possible:

    .. code-block::

        for x in stream.data(math.inf):
            ...
            if stream.iteration % frequency:
                ...

        while True:
            x = stream.next()
            ...
            if stream.iteration % frequency:
                ...

    Note:
        For better technical understanding, keep in mind that `Stream` simply
        incapsulates an "infinite iterator" that is constantly moving forward. The
        behavior is absolutely the same for both finite and infinite iterables and can
        be expressed with the following loop::

            while True:
                for item in loader:  # loader which is passed in the constructor
                    ...

        Documentation for `Stream.next` and `Stream.data` provide helpful examples.

    See Also:
        `ManualStream`: like `Stream`, but for cases when one logical step (e.g.
        training step) does not correspond to one iteration.
    """

    class _EpochData:
        def __init__(self, stream, n, attr):
            self._stream = stream
            self._n = n
            self._attr = attr
            self._start = self._get_current()

        def _get_current(self):
            return getattr(self._stream, self._attr)

        def __iter__(self):
            return self

        def __next__(self):
            if self._n is not None and self._get_current() - self._start >= self._n:
                raise StopIteration()
            return self._stream.next()

    def __init__(self, loader: Iterable) -> None:
        assert _try_len(loader) != 0
        self._iteration = 0
        self._epoch = 0
        self._loader = loader
        self._iter: Optional[Iterator] = None

    @property
    def iteration(self) -> int:
        """Current iteration.

        Technically, the number of `Stream.next` calls.
        """
        return self._iteration

    @property
    def epoch(self) -> int:
        """Current epoch.

        Technically, the number of "succeeded" `Stream.increment_epoch` calls.
        """
        return self._epoch

    @property
    def loader(self) -> Iterable:
        """The underlying loader."""
        return self._loader

    def _increment_iteration(self):
        self._iteration += 1

    def increment_epoch(self, max: Optional[Union[int, float]] = None) -> bool:
        """(Try to) increment epoch.

        Args:
            max: if `None` or `math.inf` then epoch is incremented; otherwise, epoch is
                incremented only if :code:`self.epoch < max`
        Returns:
            True, if epoch was incremented, otherwise, False.
        Raises:
            AssertionError: if max is float, but not `math.inf`

        Examples:
            .. testcode::

                stream = Stream(range(5))
                assert stream.epoch == 0
                assert stream.increment_epoch()
                assert stream.epoch == 1
                assert stream.increment_epoch(2)
                assert stream.epoch == 2
                assert not stream.increment_epoch(2)
                assert stream.epoch == 2
        """
        if isinstance(max, float):
            assert math.isinf(max)
        should_increment = max is None or self.epoch < max
        if should_increment:
            self._epoch += 1
        return should_increment

    def data(self, n_items: Optional[Union[int, float]] = None) -> Iterator:
        """Iterate over the loader.

        Under the hood, `Stream.next` is called, hence, `Stream.iteration` changes during
        iterations.

        Args:
            n_items: how many items to produce. If `None`, interpreted as
                :code:`len(self.loader)`.
        Raises:
            AssertionError: if :code:`n_items` is float, but not `math.inf`
            ValueError: if :code:`loader` is an iterator and :code:`n_items` is
                `None`

        Examples:
            .. testcode::

                stream = Stream(range(5))
                assert list(stream.data()) == [0, 1, 2, 3, 4]
                assert list(stream.data(3)) == [0, 1, 2]
                # stream doesn't "start over"!
                assert list(stream.data(3)) == [3, 4, 0]
                assert list(stream.data(1)) == [1]
                assert list(stream.data(2)) == [2, 3]
        """
        if isinstance(n_items, float):
            assert math.isinf(n_items)
        if n_items is None:
            if not isinstance(self.loader, Sized):
                raise ValueError()
            n_items = len(self.loader)
        return Stream._EpochData(self, n_items, 'iteration')

    def next(self) -> Any:
        """Get the next item and increment iteration.

        Returns:
            The next item.
        Raises:
            StopIteration: if :code:`loader` is a finite iterator and the data is over

        Examples:
            .. testcode::

                stream = Stream(range(3))
                assert stream.iteration == 0
                assert stream.next() == 0
                assert stream.iteration == 1
                assert stream.next() == 1
                assert stream.next() == 2
                assert stream.next() == 0
                assert stream.iteration == 4
        """
        if self._iter is None:
            self._iter = iter(self._loader)
        try:
            value = next(self._iter)
        except StopIteration:
            self.reload_iterator()
            # If the following line raises StopIteration too, then the data is over
            # and the exception should be just propagated.
            value = next(self._iter)
        self._increment_iteration()
        return value

    def reload_iterator(self) -> None:
        """Set the underlying iterator to `iter(self.loader)`.

        If the underlying loader is a finite iterable, the method can be used to
        interrupt and skip the current epoch (i.e. skip its data). If the loader is an
        iterator, the method does nothing.

        Examples:
            .. testcode::

                stream = Stream(range(5))
                assert stream.next() == 0
                assert stream.next() == 1
                stream.reload_iterator()
                assert stream.next() == 0

                stream = Stream(iter(range(5)))
                assert stream.next() == 0
                assert stream.next() == 1
                stream.reload_iterator()
                assert stream.next() == 2
        """
        self._iter = iter(self.loader)

    def set_loader(self, loader: Iterable) -> None:
        """Set new loader.

        Args:
            loader:
        Raises:
            AssertionError: if :code:`loader` is not an iterator and is empty.

        Examples:
            .. testcode::

                from itertools import repeat
                stream = Stream(repeat(0))
                for x in stream.data(5):
                    print(stream.iteration, x)
                    if stream.iteration == 2:
                        stream.set_loader(repeat(1))

            .. testoutput::

                1 0
                2 0
                3 1
                4 1
                5 1
        """
        assert _try_len(loader) != 0
        self._loader = loader
        if self._iter is not None:
            self._iter = iter(loader)


class ManualStream(Stream):
    """Like `Stream`, but with additional fine-graded control.

    `ManualStream` can be useful when one logical step does not correspond to one
    iteration (for example, you collect data from several iterations to build one
    training batch). The class inherits from `Stream` and adds some features (see
    documentation for details).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mstep = 0

    @property
    def mstep(self) -> int:
        """Current manual step.

        Technically, the number of `ManualStream.increment_mstep` calls.
        """
        return self._mstep

    def increment_mstep(self) -> None:
        """Increment manual step."""
        self._mstep += 1

    # mypy doesn't approve the signature change
    def data(  # type: ignore
        self,
        # The star is a protection against stream.data(n_msteps). Don't remove it,
        # especially if you don't understand why the given example is problematic.
        *,
        n_iterations: Optional[Union[int, float]] = None,
        n_msteps: Optional[Union[int, float]] = None
    ) -> Iterator:
        """Iterate over the loader.

        Exactly one of the arguments must be given.

        Args:
            n_iterations: if not None, the method behaves like `Stream.data`.
            n_msteps: if not None, items are produced until `ManualStream.mstep` increases
                by this value
        Raises:
            AssertionError: if both :code:`n_iterations` and :code:`n_msteps` are given
                or both of them are omitted.
            AssertionError: if :code:`n_iterations` is float, but not `math.inf`
            AssertionError: if :code:`n_msteps` is float, but not `math.inf`

        Examples:
            .. testcode::

                stream = ManualStream(range(5))
                data = stream.data(n_msteps=1)
                assert next(data) == 0
                assert next(data) == 1
                assert next(data) == 2
                assert stream.iteration == 3
                stream.increment_mstep()
                try:
                    next(data)
                except StopIteration:
                    print('StopIteration')

            .. testoutput::

                StopIteration
        """
        assert (n_iterations is None) ^ (n_msteps is None)
        if isinstance(n_iterations, float):
            assert math.isinf(n_iterations)
        if isinstance(n_msteps, float):
            assert math.isinf(n_msteps)
        if n_iterations is None:
            n = n_msteps
            attr = 'mstep'
        else:
            n = n_iterations
            attr = 'iteration'
        return Stream._EpochData(self, n, attr)
