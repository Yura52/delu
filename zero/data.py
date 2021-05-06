"""Missing batteries from `torch.utils.data`."""

import itertools
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

T = TypeVar('T')


class Enumerate(Dataset):
    """Make dataset return both indices and items.

    .. rubric:: Tutorial

    .. testcode::

        from torch.utils.data import DataLoader, TensorDataset
        X, y = torch.randn(9, 2), torch.randn(9)
        dataset = TensorDataset(X, y)
        for batch_idx, batch in DataLoader(Enumerate(dataset), batch_size=3):
            print(batch_idx)

    .. testoutput::

        tensor([0, 1, 2])
        tensor([3, 4, 5])
        tensor([6, 7, 8])
    """

    def __init__(self, dataset: Dataset) -> None:
        """Initialize self.

        Args:
            dataset
        """
        self._dataset = dataset

    @property
    def dataset(self) -> Dataset:
        """Access the underlying dataset.

        Returns:
            The dataset.
        """
        return self._dataset

    def __len__(self) -> int:
        """Get the length of the underlying dataset."""
        return len(self._dataset)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        """Return index and the corresponding item from the underlying dataset.

        Args:
            index
        Returns:
            (index, item)
        """
        return index, self._dataset[index]


class FnDataset(Dataset):
    """A thin wrapper around a loader function and its arguments.

    `FnDataset` allows to avoid implementing Dataset-classes (well, at least in simple
    cases). Below you can find the full tutorial and typical use cases, but here is a
    quick example:

    Without `FnDataset`::

        class ImagesList(Dataset):
            def __init__(self, filenames, transform):
                self.filenames = filenames
                self.transform = transform

            def __len__(self):
                return len(self.filenames)

            def __getitem__(self, index):
                return self.transform(Image.open(self.filenames[index]))

        dataset = ImagesList(filenames, transform)

    With `FnDataset`::

        dataset = FnDataset(Image.open, filenames, transform)

    .. rubric:: Tutorial

    With vanilla PyTorch, in order to create a dataset you have to inherit from
    `torch.utils.data.Dataset` and implement three methods:

    - :code:`__init__`
    - :code:`__len__`
    - :code:`__getitem__`

    With `FnDataset` the only thing you *may need* to implement is the :code:`fn`
    argument that will power :code:`__getitem__`. The easiest way to learn
    `FnDataset` is to go through examples below.

    A list of images::

        dataset = FnDataset(Image.open, filenames)
        # dataset[i] returns Image.open(filenames[i])

    A list of images that are cached after the first load::

        from functools import lru_cache
        dataset = FnDataset(lru_cache(None)(Image.open), filenames)

    `pathlib.Path` is very useful when you want to create a dataset that reads from
    files. For example::

        images_dir = Path(...)
        dataset = FnDataset(Image.open, images_dir.iterdir())

    If there are many files, but you need only those with specific extensions, use
    `pathlib.Path.glob`::

        dataset = FnDataset(Image.open, images_dir.glob(*.png))

    If there are many files in many subfolders, but you need only those with specific
    extensions and that satisfy some condition, use `pathlib.Path.rglob`::

        dataset = FnDataset(
            Image.open, (x for x in images_dir.rglob(*.png) if condition(x))
        )

    A segmentation dataset::

        image_filenames = ...
        gt_filenames = ...

        def get(i):
            return Image.open(image_filenames[i]), Image.open(gt_filenames[i])

        dataset = FnDataset(get, len(image_filenames))

    A dummy dataset that demonstrates that `FnDataset` is a very general thing:

    .. testcode::

        def f(x):
            return x * 10

        def g(x):
            return x * 2

        dataset = FnDataset(f, 3, g)
        # dataset[i] returns g(f(i))
        assert len(dataset) == 3
        assert dataset[0] == 0
        assert dataset[1] == 20
        assert dataset[2] == 40

    """

    def __init__(
        self,
        fn: Callable[..., T],
        args: Union[int, Iterable],
        transform: Optional[Callable[[T], Any]] = None,
    ) -> None:
        """Initialize self.

        Args:
            fn: the function that produces values based on arguments from :code:`args`
            args: arguments for :code:`fn`. If an iterable, but not a list, then is
                casted to a list. If an integer, then the behavior is the same as for
                :code:`list(range(args))`. The size of :code:`args` defines the return
                value for `FnDataset.__len__`.
            transform: if presented, is applied to the return value of `fn` in
                `FnDataset.__getitem__`

        Examples:
            .. code-block::

                import PIL.Image as Image
                import torchvision.transforms as T

                dataset = FnDataset(Image.open, filenames, T.ToTensor())
        """
        self._fn = fn
        if isinstance(args, Iterable):
            if not isinstance(args, list):
                args = list(args)
        self._args = args
        self._transform = transform

    def __len__(self) -> int:
        """Get the dataset size.

        See `FnDataset` for details.

        Returns:
            size
        """
        return len(self._args) if isinstance(self._args, list) else self._args

    def __getitem__(self, index: int) -> Any:
        """Get value by index.

        See `FnDataset` for details.

        Args:
            index
        Returns:
            value
        Raises:
            IndexError: if :code:`index >= len(self)`
        """
        if isinstance(self._args, list):
            x = self._args[index]
        elif index < self._args:
            x = index
        else:
            raise IndexError(f'Index {index} is out of range')
        x = self._fn(x)
        return x if self._transform is None else self._transform(x)


class _IndicesDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> int:
        return i


class IndexLoader:
    """Like `~torch.utils.data.DataLoader`, but over indices instead of data.

    **The shuffling logic is delegated to the native PyTorch DataLoader**, i.e. no
    custom logic is performed under the hood. The data loader which actually generates
    indices is available as `IndexLoader.loader`.

    Examples:
        Usage for training:

        .. code-block::

            train_loader = IndexLoader(len(train_dataset), batch_size, shuffle=True)
            for epoch in epochs:
                for batch_idx in train_loader:
                    ...

        Other examples:

        .. testcode::

            dataset_size = 10  # len(dataset)
            for batch_idx in IndexLoader(dataset_size, batch_size=3):
                print(batch_idx)

        .. testoutput::

            tensor([0, 1, 2])
            tensor([3, 4, 5])
            tensor([6, 7, 8])
            tensor([9])

        .. testcode::

            dataset_size = 10  # len(dataset)
            for batch_idx in IndexLoader(dataset_size, 3, drop_last=True):
                print(batch_idx)

        .. testoutput::

            tensor([0, 1, 2])
            tensor([3, 4, 5])
            tensor([6, 7, 8])

    See also:
        `zero.iter_batches`
    """

    def __init__(
        self, size: int, *args, device: Union[int, str, torch.device] = 'cpu', **kwargs
    ) -> None:
        """Initialize self.

        Args:
            size: the number of items (for example, :code:`len(dataset)`)
            *args: positional arguments for `torch.utils.data.DataLoader`
            device: if not CPU, then all indices are materialized and moved to the
                device at the beginning of every loop. It can be useful when the indices
                are applied to non-CPU data (e.g. CUDA-tensors) and moving data between
                devices takes non-negligable time (which can happen in the case of
                simple and fast models like MLPs).
            **kwargs: keyword arguments for `torch.utils.data.DataLoader`
        Raises:
            AssertionError: if size is not positive
        """
        assert size > 0
        self._batch_size = args[0] if args else kwargs.get('batch_size', 1)
        self._loader = DataLoader(_IndicesDataset(size), *args, **kwargs)
        if isinstance(device, (int, str)):
            device = torch.device(device)
        self._device = device

    @property
    def loader(self) -> DataLoader:
        """The underlying DataLoader."""
        return self._loader

    def __len__(self) -> int:
        """Get the size of the underlying DataLoader."""
        return len(self.loader)

    def __iter__(self):
        return iter(
            self._loader
            if self._device.type == 'cpu'
            else torch.cat(list(self.loader)).to(self._device).split(self._batch_size)
        )


def _is_namedtuple(x) -> bool:
    return isinstance(x, tuple) and all(
        hasattr(x, attr) for attr in ['_make', '_asdict', '_replace', '_fields']
    )


def iter_batches(
    data: Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Dict[Any, torch.Tensor],
        TensorDataset,
    ],
    *args,
    **kwargs,
) -> Iterator:
    """*Efficiently* iterate over data (tensor, tuple of tensors, dict of tensors etc.)
    in a batchwise manner.

    The function is useful when you want to *efficiently* iterate **once** over
    tensor-based data in a batchwise manner. See examples below for typical use cases.

    The function is a more efficient alternative to `torch.utils.data.DataLoader` when
    it comes to in-memory data, because it uses batch-based indexing instead of
    item-based indexing (DataLoader's behavior). **The shuffling logic is delegated to
    the native PyTorch DataLoader**, i.e. no custom logic is performed under the hood.

    Args:
        data:
        *args: positional arguments for `IndexLoader`
        **kwargs: keyword arguments for `IndexLoader`
    Returns:
        Iterator over batches.

    Warning:
        Numpy-arrays are not supported because of how they behave when indexed by a
        torch tensor of the size 1. For details, see
        `the issue <https://github.com/numpy/numpy/issues/16543>`_

    Note:
        If you want to infititely iterate over batches, wrap the function in
        :code:`while True:`.

    See also:
        - `zero.data.IndexLoader`
        - `concat`

    Examples:
        Besides loops over batches, the function can be used in combination with
        `concat`:

        .. code-block::

            result = concat(map(fn, iter_batches(dataset_or_tensors_or_whatever, ...)))

        The function can also be used for training:

        .. code-block::

            for epoch in epochs:
                for batch in iter_batches(data, batch_size, shuffle=True)):
                    ...
    """
    # mypy understands very little about this function
    if _is_namedtuple(data):
        assert data
        f = lambda idx: type(data)._make(x[idx] for x in data)  # type: ignore # noqa
        size = len(data[0])  # type: ignore # noqa
    elif isinstance(data, tuple):
        assert data
        f = lambda idx: type(data)(x[idx] for x in data)  # type: ignore # noqa
        size = len(data[0])  # type: ignore # noqa
    elif isinstance(data, dict):
        assert data
        f = lambda idx: type(data)({k: v[idx] for k, v in data.items()})  # type: ignore # noqa
        size = len(next(iter(data.values())))  # type: ignore # noqa
    else:
        f = data.__getitem__
        size = len(data)  # type: ignore # noqa
    return map(f, IndexLoader(size, *args, **kwargs))


def concat(iterable: Iterable[T]) -> T:
    """Concatenate items (tensors, numpy-arrays, tuples, dicts etc.) along the first
    dimension.

    `concat` is a more general version of :code:`torch.cat(..., dim=0)`. It works not
    only with sequences of tensors, but also with sequences of containers (tuples,
    dicts etc.) of different types of data (tensors, numpy-arrays, primitive types). See
    the tutorial and the examples below to understand what the function does.

    Args:
        iterable: items **of the same structure** (for example, "an iterable of tensors"
            OR "an iterable of tuples of tensors where all the tuples are of the same
            length" OR "an iterable of dicts of tensors and numpy-arrays where all the
            dicts have the same keys" etc.)
    Returns:
        Concatenated items of the iterable.

    Note:
        The concatenation algorithm is fully determined by the first item of the
        iterable. If there are items of different structure, then the function is
        likely to fail or produce incorrect results, hence the requirement of the
        same structure for all items of the iterable.

    Warning:
        The function starts with conversion of the iterable to a list. Make sure that
        you have enough memory for such operation, otherwise, memory limit may be
        exceeded. Note that in most cases manual implementation would involve the same
        conversion, just keep this in mind when using the function.

    See also:
        `iter_batches`

    .. rubric:: Tutorial

    For usage examples, scroll further.

    If you have an iterable that contains/produces batches of some kind (tensors,
    numpy-arrays, tuples/dictionaries thereof and other not-too-specific content), then
    use `concat` to concatenate all the items. A prominent case is application of models
    and functions to batches (e.g. to :code:`DataLoader`)::

        whole_result = concat(map(model_or_fn, batches))
        # or
        whole_result = concat(expression(x) for x in batches)

    For example::

        dataset = ...  # PyTorch dataset
        loader = DataLoader(dataset, batch_size)

        def step(batch):
            X, y = batch
            return model(X), y

        y_pred, y = concat(map(step, loader))
        assert len(y_pred) == len(dataset) and len(y) == len(dataset)

        # or
        def step(batch):
            X, y = batch
            return {'y_pred': model(X), 'y': y}

        result = concat(map(step, loader))  # no changes
        assert result['y_pred'] == len(dataset) and len(result['y']) == len(dataset)

    The function can be used in combination with `iter_batches`. For example, this is
    how pairwise dot products can be calculated in a batchwise manner if full matrix
    multiplication does not fit into memory:

    .. testcode::

        n_objects = 100
        n_features = 16
        batch_size = 20
        data = torch.randn(n_objects, n_features)
        result = concat(
            batch.matmul(data.T).to('cpu') for batch in iter_batches(data, batch_size)
        )
        assert result.shape == (n_objects, n_objects)

    Or even like this:

    .. testcode::

        n_objects = 100
        n_features = 16
        batch_size = 20
        data = torch.randn(n_objects, n_features)
        result = concat(
            concat(b.matmul(a.T).to('cpu') for b in iter_batches(data, batch_size)).T
            for a in iter_batches(data, batch_size)
        )
        assert result.shape == (n_objects, n_objects)

    Examples:
        How to read the examples:

        - the mental model for understanding the following examples is "concatenating
          data for 3 batches of sizes (2, 2, 3)". Note that sizes of batches are
          allowed to vary, but the structure is always the same
        - in all examples there is :code:`data` - a list of batches; in fact, it can be
          any "iterable of batches", including iterators and generators; the list is
          chosen to simplify the demonstration

        1-D example:

        .. testcode::

            result = concat([
                torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4, 5, 6])
            ])
            assert torch.equal(result, torch.tensor([0, 1, 2, 3, 4, 5, 6]))

        2-D example:

        .. testcode::

            result = concat([
                torch.tensor([
                    [0, 0],
                    [1, 1]
                ]),
                torch.tensor([
                    [2, 2],
                    [3, 3]
                ]),
                torch.tensor([
                    [4, 4],
                    [5, 5],
                    [6, 6],
                ]),
            ])
            assert torch.equal(
                result,
                torch.tensor([
                    [0, 0],
                    [1, 1],
                    [2, 2],
                    [3, 3],
                    [4, 4],
                    [5, 5],
                    [6, 6]
                ])
            )

        N-D example: <the same>.

        The following examples demonstrate support for different kinds of input data;
        data is 1-D everywhere just for simplicity (i.e. dimensions can be arbitrary).

        .. testcode::

            array = np.array
            tensor = torch.tensor
            l = [0, 1, 2, 3, 4, 5, 6]
            a = array([0, 1, 2, 3, 4, 5, 6])
            t = tensor([0, 1, 2, 3, 4, 5, 6])

            data = [[0, 1], [2, 3], [4, 5, 6]]
            assert concat(data) == l

            data = [array([0, 1]), array([2, 3]), array([4, 5, 6])]
            assert np.array_equal(concat(data), a)

            data = [tensor([0, 1]), tensor([2, 3]), tensor([4, 5, 6])]
            assert torch.equal(concat(data), t)

            # If items are not lists, arrays nor tensors, the data is returned in a form
            # of a list. It makes sense since the list of such items is already
            # a result for all batches.
            data = ['three batches, hence three items', 0, 1.0]
            assert concat(data) == data

            data = [
                ([0, 1], array([0, 1]), tensor([0, 1])),
                ([2, 3], array([2, 3]), tensor([2, 3])),
                ([4, 5, 6], array([4, 5, 6]), tensor([4, 5, 6])),
            ]
            result = concat(data)
            assert isinstance(result, tuple) and len(result) == 3
            assert (
                result[0] == l
                and np.array_equal(result[1], a)
                and torch.equal(result[2], t)
            )

            data = [
                {'l': [0, 1], 'a': array([0, 1]), 't': tensor([0, 1])},
                {'l': [2, 3], 'a': array([2, 3]), 't': tensor([2, 3])},
                {'l': [4, 5, 6], 'a': array([4, 5, 6]), 't': tensor([4, 5, 6])},
            ]
            result = concat(data)
            assert isinstance(result, dict) and list(result) == ['l', 'a', 't']
            assert (
                result['l'] == l
                and np.array_equal(result['a'], a)
                and torch.equal(result['t'], t)
            )
    """
    data = iterable if isinstance(iterable, list) else list(iterable)
    assert data, 'iterable must be non-empty'

    def concat_fn(sequence):
        if not isinstance(sequence, list):
            sequence = list(sequence)
        x = sequence[0]
        return (
            list(itertools.chain.from_iterable(sequence))
            if isinstance(x, list)
            else np.concatenate(sequence)
            if isinstance(x, np.ndarray)
            else torch.cat(sequence)
            if isinstance(x, torch.Tensor)
            else sequence
        )

    first = data[0]
    return (
        type(first)._make(concat_fn(x[i] for x in data) for i, _ in enumerate(first))
        if _is_namedtuple(first)
        else type(first)(concat_fn(x[i] for x in data) for i, _ in enumerate(first))
        if isinstance(first, tuple)
        else type(first)((key, concat_fn(x[key] for x in data)) for key in first)
        if isinstance(first, dict)
        else concat_fn(data)
    )


def collate(iterable: Iterable[T]) -> Any:
    """Almost an alias for :code:`torch.utils.data.dataloader.default_collate`.

    Namely, the input is allowed to be any kind of iterable, not only a list. Firstly,
    if it is not a list, it is transformed to a list. Then, the list is passed to the
    original function and the result is returned as is.
    """
    if not isinstance(iterable, list):
        iterable = list(iterable)
    # > Module has no attribute "default_collate"
    return torch.utils.data.dataloader.default_collate(iterable)  # type: ignore
