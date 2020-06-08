from collections import namedtuple
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .types import TensorIndex


class NamedTensorDataset(Dataset):
    def __init__(self, *tensors: torch.Tensor, names: Tuple[str, ...]) -> None:
        assert tensors
        assert len(tensors) == len(names)
        assert tensors[0].dim()
        assert all(len(x) == len(tensors[0]) for x in tensors)
        self._names = names
        # "NamedTuple type as an attribute is not supported"
        self._tuple_cls = namedtuple(f'Batch_{id(self)}', self.names)  # type: ignore
        self._tensors = self._tuple_cls(*tensors)
        for k, v in zip(self.names, self.tensors):
            setattr(self, k, v)

    @classmethod
    def from_dict(cls, data: Dict[str, torch.Tensor]) -> 'NamedTensorDataset':
        # Keys and value orderings are consistent:
        # https://docs.python.org/3/library/stdtypes.html#dictionary-view-objects
        return cls(*data.values(), names=tuple(data.keys()))

    @property
    def tensors(self) -> Tuple[torch.Tensor]:
        return self._tensors

    @property
    def names(self) -> Tuple[str, ...]:
        return self._names

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, idx: TensorIndex) -> Tuple[torch.Tensor]:
        return self._tuple_cls._make(x[idx] for x in self.tensors)

    def __setattr__(self, attr, value):
        # The method surves as a protection against wrong usage that can lead to
        # inconsistency betweem attributes of self and self.tensors
        assert not hasattr(self, attr) or attr not in self.names
        return super().__setattr__(attr, value)


class Enumerate(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, i) -> Tuple[Any, Any]:
        return i, self._dataset[i]


class _IndicesDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> int:
        return i


def iloader(size: int, *args, **kwargs) -> DataLoader:
    assert size > 0
    return DataLoader(_IndicesDataset(size), *args, **kwargs)


def iter_batches(
    data: Union[
        np.ndarray,
        torch.Tensor,
        Tuple[np.ndarray],
        Tuple[torch.Tensor],
        TensorDataset,
        NamedTensorDataset,
    ],
    *args,
    **kwargs,
):
    return (
        (tuple(x[idx] for x in data) for idx in iloader(len(data[0]), *args, **kwargs))
        if isinstance(data, tuple)
        else (data[idx] for idx in iloader(len(data), *args, **kwargs))
    )
