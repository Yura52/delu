from typing import (
    Generic,
    Iterable,
    Iterator as typing_Iterator,
    Literal,
    Optional,
    TypeVar,
    Union,
)

T = TypeVar('T')


class Iterator(Generic[T]):
    """Wrapper for DataLoaders and other iterables for building custom (training) loops.

    `Iterator` turns an iterable (e.g. a DataLoader) into an infinite iterator
    and allows:

    - training with custom epoch size with `Iterator.next_n`
    - iterating step-by-step with `Iterator.next`
    - changing the data source on the fly with `Iterator.set_source`

    Note:
        If the wrapped data source is a finite iterator, then `Iterator`
        is finite as well.

    .. rubric:: Tutorial

    Let's say we have a data loader (in fact, it can be any iterable,
    e.g. a list of integers or an iterator over file lines)::

        loader = DataLoader(...)

    `Iterator` can be created by wrapping the data loader::

        dataiter = Iterator(loader)

    The original data source is available through `Iterator.source`::

        assert dataiter.source is loader

    Now, we can build a training loop with custom epoch size::

        for epoch in range(n_epochs):
            for batch in dataiter.next_n(custom_epoch_size):
                train(batch)
            evaluate(...)

    Or we can build a "step-based" training loop::

        for step, batch in enumerate(dataiter.next_n(n_steps)):
            train(batch)
            if step % epoch_size == 0:
                evaluate(...)

    In particular, we can build an infinite training loop::

        for step, batch in enumerate(dataiter.next_n('inf')):
            train(batch)
            if step % epoch_size == 0:
                evaluate(...)

    It is also possible to change the original data source at any moment
    (even in the middle of an epoch)::

        for step in range(n_steps):
            if should_change_data_loader():
                new_loader = DataLoader(...)
                dataiter.set_source(new_loader)
            batch = dataiter.next()
            train(batch)
            if step % custom_epoch_size == 0:
                evaluate(...)

    Note:
        For better technical understanding, keep in mind that `Iterator` simply
        stores the provided data source and an iterator over it.
        The behavior is absolutely the same for both finite and infinite iterables
        and can be expressed with the following loop::

            while True:
                # `source` is the data source passed to the constructor
                # or via set_source
                for item in source:
                    ...

        In fact, `Iterator` is not tied to neither PyTorch nor deep learning, but
        it turns out to be useful in this context.
    """

    class _NextN:
        def __init__(self, iter_: 'Iterator[T]', n: Optional[int]) -> None:
            self._iter = iter_
            self._n_total = n
            self._n_consumed = 0

        def __len__(self) -> int:
            if self._n_total is None:
                raise RuntimeError(
                    'The "iterator over N next objects" has N == infinity,'
                    ' so it has no "len"'
                )
            return self._n_total - self._n_consumed

        def __iter__(self):
            return self

        def __next__(self) -> T:
            if self._n_total is None or self._n_consumed < self._n_total:
                self._n_consumed += 1
                return self._iter.next()
            else:
                raise StopIteration()

    def __init__(self, source: Iterable[T]) -> None:
        """Initialize self.

        Args:
            source: any kind of iterable (DataLoader, list, iterator, generator, ...)
        Raises:
            AssertionError: if ``source`` is empty

        Examples:
            .. testcode::

                dataiter = Iterator([0, 1, 2, 3])
                dataiter = Iterator(range(10))
                import itertools
                dataiter = Iterator(itertools.repeat(0))

                from torch.utils.data import DataLoader, TensorDataset
                dataset = TensorDataset(torch.randn(10, 2))
                dataiter = Iterator(DataLoader(dataset, batch_size=3, shuffle=True))
        """
        self.set_source(source)

    @property
    def source(self) -> Iterable[T]:
        """The underlying iterable data source."""
        return self._source

    def set_source(self, source: Iterable[T]) -> None:
        """Set new data source.

        Args:
            data: the new data

        Examples:
            .. testcode::

                from itertools import repeat
                dataiter = Iterator(repeat(0))
                for i, x in enumerate(dataiter.next_n(5)):
                    print(i, x)
                    if i == 1:
                        dataiter.set_source(repeat(1))

            .. testoutput::

                0 0
                1 0
                2 1
                3 1
                4 1
        """
        try:
            assert len(source) > 0  # type: ignore
        except (TypeError, NotImplementedError):
            pass
        self._source = source
        self._iter: Optional[typing_Iterator[T]] = None

    def reload_iterator(self) -> None:
        """Set the underlying iterator to `iter(self.source)`.

        Note:
            If the current source is an iterator, the method is a no-op.

        Examples:
            .. testcode::

                dataiter = Iterator(range(5))
                assert dataiter.next() == 0
                assert dataiter.next() == 1
                dataiter.reload_iterator()
                assert dataiter.next() == 0

                dataiter = Iterator(iter(range(5)))
                assert dataiter.next() == 0
                assert dataiter.next() == 1
                dataiter.reload_iterator()
                assert dataiter.next() == 2
        """
        self._iter = iter(self.source)

    def next(self) -> T:
        """Get the next item.

        Returns:
            The next item.
        Raises:
            StopIteration: if `Iterator.source` is a fully consumed finite iterator.

        Examples:
            .. testcode::

                dataiter = Iterator(range(3))
                assert dataiter.next() == 0
                assert dataiter.next() == 1
                assert dataiter.next() == 2
                assert dataiter.next() == 0

            .. code-block::

                while True:
                    x = dataiter.next()
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

    def next_n(self, n_items: Union[int, Literal['inf']]) -> Iterable[T]:
        """Make a lazy iterator over the next N items.

        Warning:
            The returned iterator fetches items (i.e. calls `Iterator.next`)
            *lazily*, i.e. during actual iterations,
            NOT in advance (see examples below).

        Args:
            n_items: the number of items to iterate over. Must be either a
                non-negative integer or the string "inf" (in the latter case, the
                endless iterator is returned).

        Examples:
            .. testcode::

                dataiter = Iterator(range(4))
                assert list(dataiter.next_n(3)) == [0, 1, 2]
                assert list(dataiter.next_n(2)) == [3, 0]

            **The returned iterator is lazy**:

            .. testcode::

                dataiter = Iterator(range(5))
                a = dataiter.next_n(2)
                b = dataiter.next_n(2)
                assert next(a) == 0
                assert next(b) == 1
                assert next(a) == 2
                assert next(b) == 3

            If you want to skip the items, you should explicitly consume the iterator:

            .. testcode::

                dataiter = Iterator(range(4))
                for epoch in range(2):
                    epoch_items = dataiter.next_n(2)
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
        return Iterator._NextN(self, None if n_items == 'inf' else n_items)
