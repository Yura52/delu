import dataclasses
import math
from types import SimpleNamespace
from typing import Iterable, Iterator, Optional, TypeVar

import torch

from ._utils import is_namedtuple

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

            # in practice, this can be 'cuda' or any other device
            device = torch.device('cpu')
            tensor = torch.tensor

            x = tensor(0)
            x = to(x, dtype=torch.float, device=device)

            batch = {
                'x': tensor([0.0, 1.0]),
                'y': tensor([0, 1]),
            }
            batch = to(batch, device)

            x = [
                tensor(0.0),
                {'a': tensor(1.0), 'b': tensor(2.0)},
                (tensor(3.0), tensor(4.0))
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
            result = cat(parts)
            assert isinstance(result, tuple) and len(result) == 2
            assert torch.equal(result[0], tensor([0.0, 1.0, 2.0, 3.0]))
            assert torch.equal(result[1], tensor([0, 1, 2, 3]))

            parts = [
                {'a': tensor([0.0, 1.0]), 'b': tensor([0, 1])},
                {'a': tensor([2.0, 3.0]), 'b': tensor([2, 3])},
            ]
            result = cat(parts)
            assert isinstance(result, dict) and set(result) == {'a', 'b'}
            assert torch.equal(result['a'], tensor([0.0, 1.0, 2.0, 3.0]))
            assert torch.equal(result['b'], tensor([0, 1, 2, 3]))

            from dataclasses import dataclass
            @dataclass
            class Data:
                # all fields must be tensors
                a: torch.Tensor
                b: torch.Tensor

            parts = [
                Data(tensor([0.0, 1.0]), tensor([0, 1])),
                Data(tensor([2.0, 3.0]), tensor([2, 3])),
            ]
            result = cat(parts)
            assert isinstance(result, Data)
            assert torch.equal(result.a, tensor([0.0, 1.0, 2.0, 3.0]))
            assert torch.equal(result.b, tensor([0, 1, 2, 3]))
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
    data: T,
    batch_size: int,
    shuffle: bool = False,
    *,
    generator: Optional[torch.Generator] = None,
    drop_last: bool = False,
) -> Iterator[T]:
    """Iterate over tensor or collection of tensors by (random) batches.

    The function makes batches over the first dimension of the tensors in ``data``
    and returns an iterator over collections of the same type as the input.

    Args:
        data: the tensor or the collection (tuple/dict/dataclass) of tensors.
            If data is a collection, then the tensors must have the same first
            dimension. If data is a dataclass, then all its fields must be tensors.
        batch_size: the batch size. If ``drop_last`` is False, then the last batch can
            be smaller than ``batch_size``.
        shuffle: if True, iterate over random batches (without replacement),
            not sequentially.
        generator: the argument for `torch.randperm` when ``shuffle`` is True.
        drop_last: same as the ``drop_last`` argument for `torch.utils.data.DataLoader`.
            When True and the last batch is smaller then ``batch_size``, then this last
            batch is not returned.
    Returns:
        Iterator over batches.
    Raises:
        ValueError: if the data is empty.

    Note:
        The function lazily indexes to the provided input with batches of indices.
        This works faster than iterating over the tensors in ``data`` with
        `torch.utils.data.DataLoader`.

    See also:
        - `cat`

    Examples:
        .. code-block::

            for epoch in epochs:
                for batch in iter_batches(data, batch_size, shuffle=True)):
                    ...

        .. testcode::

            a = torch.tensor([0, 1, 2, 3, 4])
            b = torch.tensor([0, 10, 20, 30, 40])
            batch_size = 2

            for batch in iter_batches(a, batch_size):
                assert isinstance(batch, torch.Tensor)
            for batch in iter_batches((a, b), batch_size):
                assert isinstance(batch, tuple)
            for batch in iter_batches({'a': a, 'b': b}, batch_size):
                assert isinstance(batch, dict) and set(batch) == {'a', 'b'}

            assert len(list(iter_batches((a, b), batch_size))) == 3
            assert len(list(iter_batches((a, b), batch_size, drop_last=True))) == 2

            from dataclasses import dataclass
            @dataclass
            class Data:
                a: torch.Tensor
                b: torch.Tensor

            for batch in iter_batches(Data(a, b), batch_size):
                assert isinstance(batch, Data)
    """
    if not shuffle and generator is not None:
        raise ValueError('When shuffle is False, generator must be None.')

    if isinstance(data, torch.Tensor):
        if not len(data):
            raise ValueError('data must be non-empty')
        item = data
        get_batch = data.__getitem__
    elif isinstance(data, tuple):
        if not data:
            raise ValueError('data must be non-empty')
        item = data[0]
        constructor = type(data)._make if is_namedtuple(data) else type(data)  # type: ignore  # noqa: E501
        get_batch = lambda idx: constructor(x[idx] for x in data)  # type: ignore  # noqa: E731,E501
    elif isinstance(data, dict):
        if not data:
            raise ValueError('data must be non-empty')
        item = next(iter(data.values()))
        get_batch = lambda idx: type(data)({k: v[idx] for k, v in data.items()})  # type: ignore # noqa: E731,E501
    elif dataclasses.is_dataclass(data):
        fields = list(dataclasses.fields(data))
        if not fields:
            raise ValueError('data must be non-empty')
        for field in fields:
            if field.type is not torch.Tensor:
                raise ValueError('All dataclass fields must be tensors')
        item = getattr(data, fields[0].name)
        get_batch = lambda idx: type(data)(  # type: ignore  # noqa: E731
            **{field.name: getattr(data, field.name)[idx] for field in fields}
        )
    else:
        raise ValueError(f'The collection {type(data)} is not supported.')

    size = len(item)
    device = item.device
    n_batches = math.ceil(size / batch_size)
    for i, idx in enumerate(
        (
            torch.randperm(size, generator=generator, device=device)
            if shuffle
            else torch.arange(size, device=device)
        ).split(batch_size)
    ):
        if i + 1 == n_batches and len(idx) < batch_size and drop_last:
            return
        yield get_batch(idx)  # type: ignore
