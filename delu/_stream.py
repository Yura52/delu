import math
from typing import Any, Dict, Iterable, Iterator, Optional, Sized, Union

from typing_extensions import Literal


def _try_len(x):
    try:
        return len(x)
    except (TypeError, NotImplementedError):
        return None


class Stream:
    """A handy wrapper for data loaders and iterables."""

    class _NextN:
        def __init__(self, stream: 'Stream', n_items: Optional[int]) -> None:
            self._stream = stream
            self._total = n_items
            self._current = 0

        def __len__(self) -> int:
            if self._total is None:
                raise ValueError(
                    'The iterable is of the infinite size, so it has no "len"'
                )
            return self._total - self._current

        def __iter__(self):
            return self

        def __next__(self) -> Any:
            if self._total is None or self._current < self._total:
                self._current += 1
                return self._stream.next()
            else:
                raise StopIteration()

    def __init__(self, loader: Iterable) -> None:
        """Initialize self.

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
        """
        assert _try_len(loader) != 0
        self._step = 0
        self._epoch = 0
        self._loader = loader
        self._iter: Optional[Iterator] = None

    @property
    def step(self) -> int:
        """The current step (technically, the number of `Stream.next` calls)."""
        return self._step

    @property
    def epoch(self) -> int:
        """The current epoch.

        Technically, the number of `Stream.increment_epoch` calls.
        """
        return self._epoch

    @property
    def loader(self) -> Iterable:
        """The underlying loader."""
        return self._loader

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
                for x in stream.next_n(5):
                    print(stream.step, x)
                    if stream.step == 2:
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

    def _increment_step(self):
        self._step += 1

    def increment_epoch(self) -> None:
        """Increment `Stream.epoch`.

        Examples:
            .. testcode::

                stream = Stream(range(5))
                assert stream.epoch == 0
                stream.increment_epoch()
                assert stream.epoch == 1
                stream.increment_epoch()
                assert stream.epoch == 2
        """
        self._epoch += 1

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

    def next(self) -> Any:
        """Get the next item and increment ``self.step``.

        Returns:
            The next item.
        Raises:
            StopIteration: if :code:`loader` is a finite iterator and the data is over

        Examples:
            .. testcode::

                stream = Stream(range(3))
                assert stream.step == 0
                assert stream.next() == 0
                assert stream.step == 1
                assert stream.next() == 1
                assert stream.next() == 2
                assert stream.next() == 0
                assert stream.step == 4

            .. code-block::

                while True:
                    x = stream.next()
                    ...
                    if stream.step % frequency:
                        ...
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
        self._increment_step()
        return value

    def next_n(self, n_items: Union[None, int, Literal['inf']] = None) -> Iterator:
        """Iterate over the next N items.

        An iterator is returned, and items are fetched (i.e. `Stream.next` is called)
        during actual iterations, NOT in advance.

        Args:
            n_items: the number of items to iterate over. If `None`, then
                ``len(self.loader)`` is used instead. Otherwise, must be either a
                non-negative integer or "inf" (in the latter case, the endless iterator
                is returned).

        Examples:
            .. testcode::

                stream = Stream(range(5))
                assert list(stream.next_n()) == [0, 1, 2, 3, 4]
                assert list(stream.next_n(3)) == [0, 1, 2]
                # stream doesn't "start over"!
                assert list(stream.next_n(3)) == [3, 4, 0]
                assert list(stream.next_n(1)) == [1]
                assert list(stream.next_n(2)) == [2, 3]

            .. code-block::

                for x in stream.next_n('inf'):
                    ...
                    if stream.step % frequency:
                        ...
        """
        if isinstance(n_items, str):
            assert n_items == 'inf'
        if n_items is None:
            if not isinstance(self.loader, Sized):
                raise ValueError()
            n_items = len(self.loader)
        return Stream._NextN(self, None if n_items == 'inf' else n_items)

    def epochs(
        self, max_epoch: Union[int, float], epoch_size: Optional[int] = None
    ) -> Iterator[Iterator[Any]]:
        """Iterate over data epochs.

        A shortcut for what is probably the most popular form of a training loop in Deep
        Learning::

            for epoch in stream.epochs(max_epoch, epoch_size):
                for batch in epoch:
                    ...

            # is equivalent to:

            while stream.epoch < max_epoch:
                stream.increment_epoch()
                for batch in stream.next_n(epoch_size):
                    ...

        Args:
            max_epoch: defines the number of epochs. The loop keeps running while
                :code:`self.epoch < max_epoch`. If `float`, must be :code:`float('inf')`
                or `math.inf`.
            epoch_size: the number of data items in one epoch
                (is forwarded to `Stream.next_n`).

        Returns:
            Iterator over iterators over data from `Stream.loader`.
        Raises:
            AssertionError: if :code:`max_epoch` is a finite float or nan.

        Examples:
            .. testcode::

                stream = Stream(range(3))
                for epoch in stream.epochs(2):
                    for x in epoch:
                        print(x)
                    print('-')

            .. testoutput::

                0
                1
                2
                -
                0
                1
                2
                -

            .. testcode::

                stream = Stream(range(3))
                for epoch in stream.epochs(3, 2):
                    for x in epoch:
                        print(x)
                    print('-')

            .. testoutput::

                0
                1
                -
                2
                0
                -
                1
                2
                -
        """
        if isinstance(max_epoch, float):
            assert math.isinf(max_epoch)
        while self.epoch < max_epoch:
            self.increment_epoch()
            yield self.next_n(epoch_size)

    def state_dict(self) -> Dict[str, Any]:
        """Get the stream's state.

        The result can be passed to `Stream.load_state_dict`. The result includes:

        - step
        - epoch

        Note:
            Fields related to data (loader, iterator etc.) are **NOT** included in the
            state. If you want to save the "state of data stream" then you have to save
            the state of corresponding random number generators separately.

        Returns:
            state

        See also:
            `Stream.load_state_dict`

        Examples:
            .. testcode::

                stream = Stream(range(10))
                assert stream.state_dict() == {'step': 0, 'epoch': 0}
                stream.next()
                stream.next()
                stream.increment_epoch()
                assert stream.state_dict() == {'step': 2, 'epoch': 1}
        """
        return {'step': self.step, 'epoch': self.epoch}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary.

        Args:
            state_dict: state. Must be produced by `Stream.state_dict`.

        Note:
            The method does not affect data that is produced by `Stream.epochs`,
            `Stream.next_n`, `Stream.next` (see the examples below), i.e. the method
            only sets some "metadata" such as step, epoch etc. If you want to
            load the "state of data stream", you have to load the state of corresponding
            random number generators separately.

        See also:
            `Stream.state_dict`

        Examples:

            .. testcode::

                stream = Stream(range(10))
                stream.next()
                stream.increment_epoch()
                assert stream.state_dict() == {'step': 1, 'epoch': 1}

                new_stream = Stream(range(10))
                new_stream.load_state_dict(stream.state_dict())
                assert new_stream.state_dict() == {'step': 1, 'epoch': 1}
                assert new_stream.next() == 0
                assert new_stream.state_dict() == {'step': 2, 'epoch': 1}
        """
        self._step = state_dict['step']
        self._epoch = state_dict['epoch']
