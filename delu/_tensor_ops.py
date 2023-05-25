import dataclasses
import math
from typing import Iterator, List, Optional, TypeVar

import torch

from ._utils import deprecated, is_namedtuple

T = TypeVar('T')
K = TypeVar('K')


def to(data: T, *args, **kwargs) -> T:
    """Like `torch.Tensor.to`, but for collections of tensors.

    The function allows changing devices and data types for (nested) collections of
    tensors similarly to how `torch.Tensor.to` does this for a single tensor.

    Note:
        Technically, the function simply traverses the input and applies
        `torch.Tensor.to` to tensors (non-tensor values are not allowed).

    Args:
        data: the tensor or the (nested) collection of tensors. Allowed collections
            include: (named)tuples, lists, dictionaries and dataclasses.
            For dataclasses, all their fields must be tensors.
        args: the positional arguments for `torch.Tensor.to`
        kwargs: the key-word arguments for `torch.Tensor.to`
    Returns:
        transformed data.

    Examples:
        .. testcode::

            # in practice, this can be 'cuda' or any other device
            device = torch.device('cpu')
            tensor = torch.tensor

            x = tensor(0)
            x = delu.to(x, dtype=torch.float, device=device)

            batch = {
                'x': tensor([0.0, 1.0]),
                'y': tensor([0, 1]),
            }
            batch = delu.to(batch, device)

            x = [
                tensor(0.0),
                {'a': tensor(1.0), 'b': tensor(2.0)},
                (tensor(3.0), tensor(4.0))
            ]
            x = delu.to(x, torch.half)
    """

    def TO_(x):
        return to(x, *args, **kwargs)

    # mypy does not understand what is going on here, hence a lot of "type: ignore"
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)  # type: ignore
    elif isinstance(data, (tuple, list)):
        constructor = type(data)._make if is_namedtuple(data) else type(data)  # type: ignore  # noqa: E501
        return constructor(TO_(x) for x in data)  # type: ignore
    elif isinstance(data, dict):
        return type(data)((k, TO_(v)) for k, v in data.items())  # type: ignore
    elif dataclasses.is_dataclass(data):
        return type(data)(**{k: TO_(v) for k, v in vars(data).items()})  # type: ignore
    else:
        raise ValueError(
            f'the input contains an object of the unsupported type {type(data)}.'
            ' See the documentation for details'
        )


def cat(data: List[T], dim: int = 0) -> T:
    """Like `torch.cat`, but for collections of tensors.

    A typical use case is concatenating a model/function's outputs for batches
    into a single output for the whole dataset::

        class Model(nn.Module):
            def forward(self, ...) -> tuple[Tensor, Tensor]:
                ...
                return y_pred, embeddings

        # Concatenate a sequence of tuples (batch_y_pred, batch_embeddings) into a single tuple.
        y_pred, embeddings = delu.cat([model(batch) for batch in dataloader])

    The function operates recursively, so nested structures are supported as well
    (e.g. ``tuple[Tensor, dict[str, tuple[Tensor, Tensor]]]``). See other examples below.

    Note:
        Technically, roughly speaking, the function "transposes" the list of
        collections to a collection of lists and applies `torch.cat` to those lists.

    Args:
        data: the list of (nested) (named)tuples/dictionaries/dataclasses of tensors.
            All items of the list must be of the same type and have the same
            structure (tuples must be of the same length, dictionaries must have the
            same keys, dataclasses must have the same fields, etc.). All the "leaf"
            values must be of the type `torch.Tensor`.
        dim: the dimension over which the tensors are concatenated.
    Returns:
        Concatenated items of the list.
    Raises:
        ValueError: if ``data`` is empty or contains unsupported collections.

    See also:
        `delu.iter_batches`

    Examples:
        Below, only one-dimensional data and dim=0 are considered for simplicity.

        .. testcode::

            tensor = torch.tensor

            batches = [
                # (batch_x, batch_y)
                (tensor([0.0, 1.0]), tensor([[0], [1]])),
                (tensor([2.0, 3.0]), tensor([[2], [3]])),
            ]
            # result = (x, y)
            result = delu.cat(batches)
            assert isinstance(result, tuple) and len(result) == 2
            assert torch.equal(result[0], tensor([0.0, 1.0, 2.0, 3.0]))
            assert torch.equal(result[1], tensor([[0], [1], [2], [3]]))

            batches = [
                # {'x': batch_x, 'y': batch_y}
                {'x': tensor([0.0, 1.0]), 'y': tensor([[0], [1]])},
                {'x': tensor([2.0, 3.0]), 'y': tensor([[2], [3]])},
            ]
            result = delu.cat(batches)
            assert isinstance(result, dict) and set(result) == {'x', 'y'}
            assert torch.equal(result['x'], tensor([0.0, 1.0, 2.0, 3.0]))
            assert torch.equal(result['y'], tensor([[0], [1], [2], [3]]))

            from dataclasses import dataclass
            @dataclass
            class Data:
                # all fields must be tensors
                x: torch.Tensor
                y: torch.Tensor

            batches = [
                Data(tensor([0.0, 1.0]), tensor([[0], [1]])),
                Data(tensor([2.0, 3.0]), tensor([[2], [3]])),
            ]
            result = delu.cat(batches)
            assert isinstance(result, Data)
            assert torch.equal(result.x, tensor([0.0, 1.0, 2.0, 3.0]))
            assert torch.equal(result.y, tensor([[0], [1], [2], [3]]))

            batches = [
                {
                    'x': tensor([0.0, 1.0]),
                    'y': (tensor([[0], [1]]), tensor([[10], [20]]))
                },
                {
                    'x': tensor([2.0, 3.0]),
                    'y': (tensor([[2], [3]]), tensor([[30], [40]]))
                },
            ]
            result = delu.cat(batches)
            assert isinstance(result, dict) and set(result) == {'x', 'y'}
            assert torch.equal(result['x'], tensor([0.0, 1.0, 2.0, 3.0]))
            assert torch.equal(result['y'][0], tensor([[0], [1], [2], [3]]))
            assert torch.equal(result['y'][1], tensor([[10], [20], [30], [40]]))

            x = tensor([0.0, 1.0, 2.0, 3.0, 4.0])
            y = tensor([[0], [10], [20], [30], [40]])
            batch_size = 2
            ab = delu.cat(list(delu.iter_batches((x, y), batch_size)))
            assert torch.equal(ab[0], x)
            assert torch.equal(ab[1], y)
    """  # noqa: E501
    if not data:
        raise ValueError('data must be non-empty.')

    first = data[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(data, dim=dim)  # type: ignore
    elif isinstance(first, tuple):
        constructor = type(first)._make if is_namedtuple(first) else type(first)  # type: ignore  # noqa: E501
        return constructor(
            cat([x[i] for x in data], dim=dim) for i in range(len(first))  # type: ignore  # noqa: E501
        )
    elif isinstance(first, dict):
        return type(first)((key, cat([x[key] for x in data], dim=dim)) for key in first)  # type: ignore  # noqa: E501
    elif dataclasses.is_dataclass(first):
        return type(first)(
            **{
                field.name: cat([getattr(x, field.name) for x in data], dim=dim)
                for field in dataclasses.fields(first)
            }
        )  # type: ignore
    else:
        raise ValueError(f'The collection type {type(first)} is not supported.')


@deprecated('Instead, use `delu.cat`')
def concat(*args, **kwargs):
    """"""
    return cat(*args, **kwargs)


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
    A simple example (see below for more examples):

        .. testcode::

            n_objects = 100
            n_features = 4
            X = torch.randn(n_objects, n_features)
            y = torch.randn(n_objects)
            for batch_x, batch_y in delu.iter_batches(
                (X, y), batch_size=12, shuffle=True
            ):
                ...  # train(batch_x, batch_y)

    Args:
        data: the tensor or the collection ((named)tuple/dict/dataclass) of tensors.
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
        - `delu.cat`

    Examples:
        .. code-block::

            for epoch in range(n_epochs):
                for batch in delu.iter_batches(data, batch_size, shuffle=True)):
                    ...

        .. testcode::

            a = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
            b = torch.tensor([[0], [10], [20], [30], [40]])
            batch_size = 2

            for batch in delu.iter_batches(a, batch_size):
                assert isinstance(batch, torch.Tensor)
            for batch in delu.iter_batches((a, b), batch_size):
                assert isinstance(batch, tuple) and len(batch) == 2
            for batch in delu.iter_batches({'a': a, 'b': b}, batch_size):
                assert isinstance(batch, dict) and set(batch) == {'a', 'b'}

            from dataclasses import dataclass
            @dataclass
            class Data:
                a: torch.Tensor
                b: torch.Tensor

            for batch in delu.iter_batches(Data(a, b), batch_size):
                assert isinstance(batch, Data)

            ab = delu.cat(list(delu.iter_batches((a, b), batch_size)))
            assert torch.equal(ab[0], a)
            assert torch.equal(ab[1], b)

            n_batches = len(list(delu.iter_batches((a, b), batch_size)))
            assert n_batches == 3
            n_batches = len(list(delu.iter_batches((a, b), batch_size, drop_last=True)))
            assert n_batches == 2
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
        if any(len(x) != len(item) for x in data):
            raise ValueError('All tensors must have the same first dimension.')
        constructor = type(data)._make if is_namedtuple(data) else type(data)  # type: ignore  # noqa: E501
        get_batch = lambda idx: constructor(x[idx] for x in data)  # type: ignore  # noqa: E731,E501
    elif isinstance(data, dict):
        if not data:
            raise ValueError('data must be non-empty')
        item = next(iter(data.values()))
        if any(len(x) != len(item) for x in data.values()):
            raise ValueError('All tensors must have the same first dimension.')
        get_batch = lambda idx: type(data)({k: v[idx] for k, v in data.items()})  # type: ignore # noqa: E731,E501
    elif dataclasses.is_dataclass(data):
        fields = list(dataclasses.fields(data))
        if not fields:
            raise ValueError('data must be non-empty')
        item = getattr(data, fields[0].name)
        for field in fields:
            if field.type is not torch.Tensor:
                raise ValueError('All dataclass fields must be tensors')
            if len(getattr(data, field.name)) != len(item):
                raise ValueError('All tensors must have the same first dimension.')
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
