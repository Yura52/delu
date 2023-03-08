import dataclasses
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, Tuple, TypeVar, Union

import torch

from ._utils import is_namedtuple
from .data import make_index_dataloader

T = TypeVar('T')
K = TypeVar('K')


def to(data: T, *args, **kwargs) -> T:
    """Like `torch.Tensor.to`, but for collections of tensors.

    The function takes a (nested) collection of tensors
    and creates its copy where all the tensors are transformed
    with `torch.Tensor.to`.

    Args:
        data: a tensor or a (nested) collection of tensors. Only "simple" collections
            are allowed such as (named)tuples, lists, dictionaries, etc.
        args: positional arguments for `torch.Tensor.to`
        kwargs: positional arguments for `torch.Tensor.to`
    Returns:
        transformed data.

    Examples:
        .. testcode::

            device = torch.device('cpu')  # in practice, this can be 'cuda'

            x = torch.tensor(0)
            x = to(x, dtype=torch.float, device=device)

            batch = {
                'x': torch.tensor([0.0, 1.0]),
                'y': torch.tensor([0, 1]),
            }
            batch = to(batch, device)

            x = [
                torch.tensor(0.0),
                {'a': torch.tensor(1.0), 'b': torch.tensor(2.0)},
                (torch.tensor(3.0), torch.tensor(4.0))
            ]
            x = to(x, torch.half)
    """

    def TO_(x):
        return to(x, *args, **kwargs)

    # mypy does not understand what is going on here, hence a lot of "type: ignore"
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)  # type: ignore
    elif isinstance(data, SimpleNamespace) or dataclasses.is_dataclass(data):
        return type(data)(**{k: TO_(v) for k, v in vars(data).items()})  # type: ignore
    elif is_namedtuple(data):
        return type(data)._make(TO_(x) for x in data)  # type: ignore
    elif isinstance(data, (tuple, list)):
        return type(data)(TO_(x) for x in data)  # type: ignore
    elif isinstance(data, dict):
        return type(data)({k: TO_(v) for k, v in data.items()})  # type: ignore
    else:
        raise ValueError(
            f'the input contains an object of the unsupported type {type(data)}.'
            ' See the documentation for details'
        )


def cat(iterable: Iterable[T], dim: int = 0) -> T:
    """Like `torch.cat`, but for collections of tensors.

    It is especially useful for concatenating outputs of a function or a model
    that returns not a single tensor, but a tuple/dictionary/dataclass of tensors.
    For example::

        class Model(nn.Module):
            ...
            def forward(...) -> tuple[Tensor, Tensor]:
                ...
                return y_pred, embeddings

        model = Model(...)
        dataset = Dataset(...)
        dataloader = DataLoader(...)

        # prediction
        model.eval()
        with torch.inference_mode():
            y_pred, embeddings = cat(model(batch) for batch in dataloader)
        assert isinstance(y_pred, torch.Tensor) and len(y_pred) == len(dataset)
        assert isinstance(embeddings, torch.Tensor) and len(embeddings) == len(dataset)

    The function can be best understood from examples.
    Roughly speaking, the function performs two steps:

        1. Transform the sequence of collections into a collection of sequencies.
        2. Apply `torch.cat(..., dim=dim)` to the fields of the obtained collection.

    See other examples below.

    Args:
        iterable: the iterable of tuples/dictionaries/dataclasses of tensors.
            All items of the iterable must be of the same type and have the same
            structure (tuples must be of the same length, dictionaries must have the
            same keys, dataclasses must have the same fields). Dataclasses can have
            only tensor-valued fields.
    Returns:
        Concatenated items of the iterable.
    Raises:
        ValueError: if the iterable is empty or contains unsupported collections.

    See also:
        `iter_batches`

    Examples:
        Below, only one-dimensional data and dim=0 are considered for simplicity.

        .. testcode::

            tensor = torch.tensor

            parts = [
                (tensor([0.0, 1.0]), tensor([0, 1])),
                (tensor([2.0, 3.0]), tensor([2, 3])),
            ]
            concatenated = cat(parts)
            assert torch.equal(concatenated[0], tensor([0.0, 1.0, 2.0, 3.0]))
            assert torch.equal(concatenated[1], tensor([0, 1, 2, 3]))

            parts = [
                {'a': tensor([0.0, 1.0]), 'b': tensor([0, 1])},
                {'a': tensor([2.0, 3.0]), 'b': tensor([2, 3])},
            ]
            concatenated = cat(parts)
            assert torch.equal(concatenated['a'], tensor([0.0, 1.0, 2.0, 3.0]))
            assert torch.equal(concatenated['b'], tensor([0, 1, 2, 3]))

            from dataclasses import dataclass
            @dataclass
            class Part:
                # all fields must be tensors
                a: torch.Tensor
                b: torch.Tensor

            parts = [
                Part(tensor([0.0, 1.0]), tensor([0, 1])),
                Part(tensor([2.0, 3.0]), tensor([2, 3])),
            ]
            concatenated = cat(parts)
            assert torch.equal(concatenated.a, tensor([0.0, 1.0, 2.0, 3.0]))
            assert torch.equal(concatenated.b, tensor([0, 1, 2, 3]))
    """
    data = iterable if isinstance(iterable, list) else list(iterable)
    if not data:
        raise ValueError('iterable must be non-empty.')

    def tcat(x):
        return torch.cat(x, dim=dim)

    first = data[0]
    if isinstance(first, torch.Tensor):
        raise ValueError(
            'Use torch.cat instead of delu.cat for concatenating a sequence of tensors.'
            ' Use delu.cat when concatenating a sequence of collections of tensors.'
        )
    elif isinstance(first, tuple):
        constructor = type(first)._make if is_namedtuple(first) else type(first)  # type: ignore  # noqa: E501
        return constructor(tcat([x[i] for x in data]) for i in range(len(first)))
    elif isinstance(first, dict):
        return type(first)((key, tcat([x[key] for x in data])) for key in first)  # type: ignore  # noqa: E501
    elif dataclasses.is_dataclass(first):
        fields = {}
        for field in dataclasses.fields(first):
            if field.type is not torch.Tensor:
                raise ValueError(
                    f'All dataclass fields must be PyTorch Tensors. Found {field.type}'
                )
            fields[field.name] = tcat([getattr(x, field.name) for x in data])
        return type(first)(**fields)
    else:
        raise ValueError(f'The collection type {type(first)} is not supported.')


def iter_batches(
    data: Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Dict[Any, torch.Tensor],
        torch.utils.data.TensorDataset,
    ],
    *args,
    **kwargs,
) -> Iterator:
    """Efficiently iterate over data (tensor, tuple of tensors, dict of tensors etc.)
    by batches.

    The function is useful when you want to *efficiently* iterate **once** over
    tensor-based data by batches. See examples below for typical use cases.

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

    Note:
        If you want to infinitely iterate over batches, wrap the function in
        ``while True:``.

    See also:
        - `delu.data.make_index_dataloader`
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
    if is_namedtuple(data):
        assert data
        f = lambda idx: type(data)._make(x[idx] for x in data)  # type: ignore # noqa
        size = len(data[0])  # type: ignore
    elif isinstance(data, tuple):
        assert data
        f = lambda idx: type(data)(x[idx] for x in data)  # type: ignore # noqa
        size = len(data[0])  # type: ignore
    elif isinstance(data, dict):
        assert data
        f = lambda idx: type(data)({k: v[idx] for k, v in data.items()})  # type: ignore # noqa
        size = len(next(iter(data.values())))  # type: ignore
    else:
        f = data.__getitem__
        size = len(data)  # type: ignore
    return map(f, make_index_dataloader(size, *args, **kwargs))
