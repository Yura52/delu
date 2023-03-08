from typing import Any, Iterable, Iterator, Literal, Optional, Union


def _try_len(x):
    try:
        return len(x)
    except (TypeError, NotImplementedError):
        return None


class Stream:
    """TODO"""

    class _NextN:
        def __init__(self, stream: 'Stream', n_items: Optional[int]) -> None:
            self._stream = stream
            self._total = n_items
            self._current = 0

        def __len__(self) -> int:
            if self._total is None:
                raise RuntimeError(
                    'The "iterator over N next objects" has N == infinity,'
                    ' so it has no "len"'
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

    def __init__(self, data: Iterable) -> None:
        """Initialize self.

        Args:
            data: any kind of iterable (DataLoader, list, iterator, generator, ...)
        Raises:
            AssertionError: if ``data`` is empty

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
        assert _try_len(data) != 0
        self._data = data
        self._iter: Optional[Iterator] = None

    @property
    def data(self) -> Iterable:
        """The underlying iterable data source."""
        return self._data

    def set_data(self, data: Iterable) -> None:
        """Set new data.

        Args:
            data: the new data

        Examples:
            .. testcode::

                from itertools import repeat
                stream = Stream(repeat(0))
                for i, x in enumerate(stream.next_n(5)):
                    print(i, x)
                    if i == 1:
                        stream.set_data(repeat(1))

            .. testoutput::

                0 0
                1 0
                2 1
                3 1
                4 1
        """
        assert _try_len(data) != 0
        self._data = data
        self._iter = None

    def reload_iterator(self) -> None:
        """Set the underlying iterator to `iter(self.data)`.

        Note:
            If the data source is an iterator, the method is a no-op.

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
        self._iter = iter(self.data)

    def next(self) -> Any:
        """Get the next item.

        Returns:
            The next item.
        Raises:
            StopIteration: if `Stream.data` is a fully consumed finite iterator.

        Examples:
            .. testcode::

                stream = Stream(range(3))
                assert stream.next() == 0
                assert stream.next() == 1
                assert stream.next() == 2
                assert stream.next() == 0

            .. code-block::

                while True:
                    x = stream.next()
                    ...
        """
        if self._iter is None:
            self.reload_iterator()
        assert self._iter is not None
        try:
            return next(self._iter)
        except StopIteration:
            self.reload_iterator()
            # If the following line raises StopIteration too, then the data is over
            # and the exception should be just propagated.
            return next(self._iter)

    def next_n(self, n_items: Union[int, Literal['inf']]) -> Iterator:
        """Make a lazy iterator over the next N items.

        Warning:
            the returned iterator fetches items (i.e. calls `Stream.next`)
            during actual iterations, NOT in advance (see examples below).

        Args:
            n_items: the number of items to iterate over. Must be either a
                non-negative integer or the string "inf" (in the latter case, the
                endless iterator is returned).

        Examples:
            .. testcode::

                stream = Stream(range(4))
                assert list(stream.next_n(3)) == [0, 1, 2]
                assert list(stream.next_n(2)) == [3, 0]

            **The returned iterator is lazy**:

            .. testcode::

                stream = Stream(range(5))
                a = stream.next_n(2)
                b = stream.next_n(2)
                assert next(a) == 0
                assert next(b) == 1
                assert next(a) == 2
                assert next(b) == 3

            If you want to skip the items, you should explicitly consume the iterator:

            .. testcode::

                stream = Stream(range(4))
                for epoch in range(2):
                    epoch_items = stream.next_n(2)
                    if epoch == 0:
                        # Let's say you want to skip epoch=0 for some reason.
                        # If you want to skip the items from this epoch,
                        # you must explicitly consume the iterator:
                        for _ in epoch_items:
                            pass
                        continue
                    for x in epoch_items:
                        print(x)

            .. testoutput::

                2
                3
        """
        if isinstance(n_items, str):
            assert n_items == 'inf'
        return Stream._NextN(self, None if n_items == 'inf' else n_items)
