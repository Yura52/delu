"""Smart Python loops."""

__all__ = ['Stream']

import math
from typing import Any, Dict, Iterable, Iterator, Optional, Sized, Union

from tqdm import tqdm


def _try_len(x):
    try:
        return len(x)
    except (TypeError, NotImplementedError):
        return None


class Stream:
    """Smart wrapper for iterables.

    `Stream` simplifies managing loops, especially in typical deep learning scenarios
    (it is usually used to wrap :code:`train_dataloader` or any other data source).

    `Stream`:

    - simplifies management of the "epoch" and "iteration" variables
    - allows to dump and restore loop's state: epoch, iteration, etc.
    - allows to customize the size of epoch
    - allows to change the underlying data loader on the fly
    - enables useful patterns

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
        for epoch in range(n_epochs):
            for x in loader:
                iteration += 1
                print('Epoch:', epoch, 'Iteration:', iteration)
                ...

    There are several ways how you can use `Stream` to enhance this loop. Let's start
    with creating a stream::

        stream = Stream(DataLoader(...))

    The dataloader is accessible via `Stream.loader`. Now, let's reproduce the loop
    above::

        for epoch in stream.epochs(n_epochs):
            for x in epoch:
                print('Epoch:', stream.epoch, 'Iteration:', stream.iteration)

    We see that `Stream.epoch` and `Stream.iteration` are managed automatically.
    Additionally, a progress bar is displayed while the loop is running.

    Saving the loop's state and resuming the loop is possible with the methods
    `Stream.state_dict`, `Stream.load_state_dict`. In practice, it may look like this::

        model = ...
        optimizer = ...
        stream = Stream(DataLoader(...))
        if load_from_checkpoint:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            ...
            stream.load_state_dict(checkpoint['stream'])
        ...
        for epoch in stream.epochs(...):
            for batch in epoch:
                ...
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': model.state_dict(),
                    'stream': stream.state_dict(),
                },
                f'checkpoint_{stream.epoch}.pt'
            )

    Note:
        Stream's state does not include the loader's state. See `Stream.state_dict` and
        `Stream.load_state_dict` for details.

    In order to customize the epoch size, pass the size as the second argument::

        for epoch in stream.epochs(n_epochs, custom_epoch_size):
            for x in epoch:
                ...

    Changing the underlying loader on the fly is possible at *any* moment (even in the
    middle of epoch) via `Stream.set_loader`. For example::

        for epoch in stream.epochs(n_epochs, custom_epoch_size):
            for x in epoch:
                ...
                if need_new_data():
                    stream.set_loader(new_loader)

    If the method `Stream.epochs` does not fit your workflow and you want more control
    over the loop, there are more "low-level" methods (in fact, `Stream.epochs` is just
    a thin wrapper around them):

    - `Stream.increment_epoch`
    - `Stream.data`
    - `Stream.next`

    Note:
        For better technical understanding, keep in mind that `Stream` simply
        encapsulates an "infinite iterator" that is constantly moving forward. The
        behavior is absolutely the same for both finite and infinite iterables and can
        be expressed with the following loop::

            while True:
                for item in loader:  # loader which is passed in the constructor
                    ...

        Documentation for `Stream.next` and `Stream.data` provide helpful examples.
    """

    class _EpochData:
        def __init__(self, stream, size):
            self._stream = stream
            self._size = size
            self._start = self._stream.iteration

        def __iter__(self):
            return self

        def __next__(self):
            if (
                self._size is not None
                and self._stream.iteration - self._start >= self._size
            ):
                raise StopIteration()
            return self._stream.next()

    def __init__(self, loader: Iterable) -> None:
        assert _try_len(loader) != 0
        self._iteration = 0
        self._epoch = 0
        self._loader = loader
        self._iter: Optional[Iterator] = None
        self._pbar: Optional[tqdm] = None

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

    def _increment_iteration(self):
        self._iteration += 1

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

            .. code-block::

                while True:
                    x = stream.next()
                    ...
                    if stream.iteration % frequency:
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
        self._increment_iteration()
        if self._pbar is not None:
            self._pbar.update()
        return value

    def data(self, n_items: Optional[Union[int, float]] = None) -> Iterator:
        """Iterate over the loader.

        Under the hood, `Stream.next` is called, hence, `Stream.iteration` changes
        during iterations.

        Args:
            n_items: how many items to produce. If `None`, interpreted as
                :code:`len(self.loader)`. If `float`, must be `math.inf`.
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

            .. code-block::

                for x in stream.data(math.inf):
                    ...
                    if stream.iteration % frequency:
                        ...
        """
        if isinstance(n_items, float):
            assert math.isinf(n_items)
        if n_items is None:
            if not isinstance(self.loader, Sized):
                raise ValueError()
            n_items = len(self.loader)
        return Stream._EpochData(self, n_items)

    def epochs(
        self,
        n_epochs: Union[int, float],
        epoch_size: Optional[Union[int, float]] = None,
        progress_bar: bool = True,
    ) -> Iterator[Iterator[Any]]:
        """Iterate over data epochs.

        A shortcut for what is probably the most popular form of a training loop in Deep
        Learning (plus a progress bar)::

            for epoch in stream.epochs(n_epochs, epoch_size):
                for x in epoch:
                    ...

            # is equivalent to:

            while stream.epoch < n_epochs:
                stream.increment_epoch()
                for x in stream.data(epoch_size):
                    ...

        Args:
            n_epochs: the number of epochs.  If `float`, must be `math.inf`.
            epoch_size: the number of data items in one epoch (is forwarded to
                `Stream.data`)
            progress_bar: show the progress bar for iterations. The initial value is set
                to `Stream.iteration`. See also the note below.
        Returns:
            Iterator over iterators over data from `Stream.loader`.
        Raises:
            AssertionError: if :code:`n_epochs` if `float`, but not `math.inf`.

        Note:
            If :code:`progress_bar` is True, *the progress bar is updated on yielding
            every item* which means that the progress bar should be interpreted as "what
            iteration is in progress" instead of "how many iterations are done". The
            percentage will be displayed only if the total number of planned iterations
            can be inferred from the arguments and/or from `Stream.loader`.

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
        if isinstance(n_epochs, float):
            assert math.isinf(n_epochs)
        if progress_bar:
            pbar_epoch_size = (
                _try_len(self.loader) if epoch_size is None else epoch_size
            )
            self._pbar = tqdm(
                initial=self.iteration,
                total=None
                if (pbar_epoch_size is None or math.isinf(n_epochs))
                else n_epochs * pbar_epoch_size,
            )
        while self.epoch < n_epochs:
            self.increment_epoch()
            yield self.data(epoch_size)

    def state_dict(self) -> Dict[str, Any]:
        """Get the stream's state.

        The result can be passed to `Stream.load_state_dict`. The result includes:

        - epoch
        - iteration

        Note:
            Fields related to data (loader, iterator etc.) are **NOT** included in the
            state. If you want to save the "state of data stream" then you have to save
            the state of corresponding random number generators separately.

        Returns:
            state

        Examples:
            .. testcode::

                stream = Stream(range(10))
                assert stream.state_dict() == {'epoch': 0, 'iteration': 0}
                stream.next()
                stream.next()
                stream.increment_epoch()
                assert stream.state_dict() == {'epoch': 1, 'iteration': 2}

        See also:
            `Stream.load_state_dict`
        """
        return {'iteration': self.iteration, 'epoch': self.epoch}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary.

        Args:
            state_dict: state. Must be produced by `Stream.state_dict`.

        Note:
            The method does not affect data that is produced by `Stream.epochs`,
            `Stream.data`, `Stream.next` (see the examples below), i.e. the method
            only sets some "metadata" such as epoch, iteration etc. If you want to
            load the "state of data stream", you have to load the state of corresponding
            random number generators separately.

        Examples:

            .. testcode::

                stream = Stream(range(10))
                stream.next()
                stream.increment_epoch()
                assert stream.state_dict() == {'epoch': 1, 'iteration': 1}

                new_stream = Stream(range(10))
                new_stream.load_state_dict(stream.state_dict())
                assert new_stream.state_dict() == {'epoch': 1, 'iteration': 1}
                assert new_stream.next() == 0
                assert new_stream.state_dict() == {'epoch': 1, 'iteration': 2}

        See also:
            `Stream.state_dict`
        """
        self._iteration = state_dict['iteration']
        self._epoch = state_dict['epoch']
