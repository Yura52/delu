r"""Missing batteries from `torch.utils.data`."""

__all__ = ['NamedTensorDataset', 'Enumerate', 'iloader', 'iter_batches']

from collections import namedtuple
from typing import Any, Dict, Iterator, NamedTuple, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ._util import is_namedtuple
from .types import TensorIndex


class NamedTensorDataset(Dataset):
    """Named version of `~torch.utils.data.TensorDataset`.

    Args:
        *tensors: tensors **of the same length**
        names: names for tensors

    Examples:
        .. testcode::

            X, y = torch.randn(10, 2), torch.randn(10)
            dataset = NamedTensorDataset(X, y, names=['X', 'y'])
            # or
            dataset = NamedTensorDataset.from_dict({'X': X, 'y': y})
            assert dataset.X is X
            assert dataset.y is y
            # dataset.tensors is a named tuple with the fields 'X' and 'y'

            def step_fn(X, y):
                ...

            for batch in torch.utils.data.DataLoader(dataset, 2):
                # batch is a named tuple
                step_fn(batch.X, batch.y)
    """

    def __init__(self, *tensors: torch.Tensor, names: Sequence[str]) -> None:
        assert tensors
        assert len(tensors) == len(names)
        assert tensors[0].dim()
        assert all(len(x) == len(tensors[0]) for x in tensors)
        self._names = tuple(names)
        # "NamedTuple type as an attribute is not supported"
        self._tuple_cls = namedtuple(f'Batch_{id(self)}', self.names)  # type: ignore
        self._tensors = self._tuple_cls(*tensors)
        for k, v in zip(self.names, self.tensors):
            setattr(self, k, v)

    @classmethod
    def from_dict(cls, data: Dict[str, torch.Tensor]) -> 'NamedTensorDataset':
        """Construct `NamedTensorDataset` from a dictionary.

        Args:
            data:
        Returns:
            Dataset.
        """
        # Key and value orderings are consistent:
        # https://docs.python.org/3/library/stdtypes.html#dictionary-view-objects
        return cls(*data.values(), names=tuple(data.keys()))

    @property
    def tensors(self) -> NamedTuple:
        """Access the underlying tensors.

        Returns:
            The tensors.
        """
        return self._tensors

    @property
    def names(self) -> Tuple[str, ...]:
        """Get the field names.

        Returns:
            The names.
        """
        return self._names

    def __len__(self) -> int:
        """Return the length of all tensors.

        Returns:
            The length.
        """
        return len(self.tensors[0])

    def __getitem__(self, idx: TensorIndex) -> NamedTuple:
        """Get item(s) by index/indices.

        Args:
            idx: the index
        Returns:
            The item(s).

        Note:
            Efficient indexing with slices, arrays and tensors is also supported.
        """
        return self._tuple_cls._make(x[idx] for x in self.tensors)

    def __setattr__(self, attr, value):
        # The method surves as a protection against wrong usage that can lead to
        # inconsistency betweem attributes of self and self.tensors
        assert not hasattr(self, attr) or attr not in self.names
        return super().__setattr__(attr, value)


class Enumerate(Dataset):
    """Make dataset return both indices and items.

    .. rubric:: Tutorial

    .. testcode::

        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(torch.randn(9, 2), torch.randn(9))  # X, y
        for batch_idx, batch in DataLoader(Enumerate(dataset), batch_size=3):
            print(batch_idx)

    .. testoutput::

        tensor([0, 1, 2])
        tensor([3, 4, 5])
        tensor([6, 7, 8])
    """

    def __init__(self, dataset: Dataset) -> None:
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


class _IndicesDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> int:
        return i


def iloader(size: int, *args, **kwargs) -> DataLoader:
    """Make `~torch.utils.data.DataLoader` over indices.

    This thing has many names (such as "iter_batch_indices") and allows to iterate over
    batch indices, not over batches. **The shuffling logic is fully delegated to native
    PyTorch DataLoader**, i.e. no custom logic is performed under the hood.

    Args:
        size: the size of dataset (for example, :code:`len(dataset)`)
        *args: positional arguments for `torch.utils.data.DataLoader`
        **kwargs: keyword arguments for `torch.utils.data.DataLoader`
    Raises:
        AssertionError: if size is not positive

    See Also:
        `iter_batches`

    Examples:
        Usage for training:

        .. code-block::

            train_loader = iloader(len(train_dataset), batch_size, shuffle=True)
            for epoch in epoches:
                for batch_idx in loader:
                    ...

        More specific examples:

        .. testcode::

            dataset_size = 10  # len(dataset)
            for batch_idx in iloader(dataset_size, batch_size=3):
                print(batch_idx)

        .. testoutput::

            tensor([0, 1, 2])
            tensor([3, 4, 5])
            tensor([6, 7, 8])
            tensor([9])

        .. testcode::

            dataset_size = 10  # len(dataset)
            for batch_idx in iloader(dataset_size, 3, drop_last=True):
                print(batch_idx)

        .. testoutput::

            tensor([0, 1, 2])
            tensor([3, 4, 5])
            tensor([6, 7, 8])
    """
    assert size > 0
    return DataLoader(_IndicesDataset(size), *args, **kwargs)


def iter_batches(
    data: Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Dict[Any, torch.Tensor],
        TensorDataset,
        NamedTensorDataset,
    ],
    *args,
    **kwargs,
) -> Iterator:
    """*Efficiently* iterate over data in a batchwise manner.

    The function is useful when you want to efficiently iterate **once** over
    tensor-based data. See examples below for typical use cases.

    The function is a more efficient alternative to `torch.utils.data.DataLoader` when
    it comes to in-memory data, because it uses batch-based indexing instead of
    item-based indexing (DataLoader's behavior). **The shuffling logic is fully
    delegated to native PyTorch DataLoader**, i.e. no custom logic is performed under
    the hood.

    Args:
        data:
        *args: positional arguments for `iloader`
        **kwargs: keyword arguments for `iloader`
    Returns:
        Iterator over batches.

    Warning:
        Numpy-arrays are not supported because of how they behave when indexed by a
        torch tensor of the size 1. For details, see
        `the issue <https://github.com/numpy/numpy/issues/16543>`_

    Note:
        If you want to infititely iterate over batches, wrap the function in
        :code:`while True:`.

    See Also:
        `iloader`

    Examples:
        The function can be used for applying some function to batches, especially when
        used together with `zero.concat_dmap`:

        .. code-block::

            result = concat(map(fn, iter_batches(dataset_or_tensors_or_whatever, ...)))

        The function can also used for training:

        .. code-block::

            for epoch in epoches:
                for batch in iter_batches(data, batch_size, shuffle=True)):
                    ...
    """
    # mypy understands very little about this function
    if is_namedtuple(data):
        assert data
        f = lambda idx: type(data)._make(x[idx] for x in data)  # type: ignore # noqa
        size = len(data[0])
    elif isinstance(data, tuple):
        assert data
        f = lambda idx: type(data)(x[idx] for x in data)  # type: ignore # noqa
        size = len(data[0])
    elif isinstance(data, dict):
        assert data
        f = lambda idx: type(data)({k: v[idx] for k, v in data.items()})  # type: ignore # noqa
        size = len(next(iter(data.values())))
    else:
        f = data.__getitem__
        size = len(data)
    return map(f, iloader(size, *args, **kwargs))
