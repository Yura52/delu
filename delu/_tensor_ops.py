import dataclasses
import math
from typing import Iterator, List, Optional, TypeVar

import torch

from ._utils import deprecated, is_namedtuple

T = TypeVar('T')
K = TypeVar('K')


def to(data: T, /, *args, **kwargs) -> T:
    """Like `torch.Tensor.to`, but for collections of tensors.

    While `torch.Tensor.to` changes the device and/or data type of a tensor,
    `delu.to` changes the device and/or data type of a *collection* of tensors
    (tuples, named tuples, lists, dictionaries, dataclasses
    and nested combinations thereof).

    **Usage**

    >>> # In practice, the device type can be 'cuda' or anything else.
    >>> # Also, `delu.to` can be used to change data type.
    >>> device = torch.device('cpu')
    >>> batch_size = 64
    >>> x = torch.randn(batch_size, 10)
    >>> y = torch.randn(batch_size)
    >>> z = torch.randn(batch_size, 20, 30)

    `delu.to` can be applied to tuples:

    >>> batch = (x, y, z)
    >>> batch = delu.to(batch, device)

    `delu.to` can be applied to lists:

    >>> batch = [x, y, z]
    >>> batch = delu.to(batch, device)

    `delu.to` can be applied to dictionaries:

    >>> batch = {'x': x, 'y': y, 'z': z}
    >>> batch = delu.to(batch, device)

    `delu.to` can be applied to named tuples:

    >>> from typing import NamedTuple
    >>> class Data(NamedTuple):
    ...     x: torch.Tensor
    ...     y: torch.Tensor
    ...     z: torch.Tensor
    >>> batch = Data(x, y, z)
    >>> batch = delu.to(batch, device)
    >>> isinstance(batch, Data)
    True

    `delu.to` can be applied to dataclasses:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Data:
    ...     x: torch.Tensor
    ...     y: torch.Tensor
    ...     z: torch.Tensor
    >>> batch = Data(x, y, z)
    >>> batch = delu.to(batch, device)
    >>> isinstance(batch, Data)
    True

    `delu.to` can be applied to nested collections of tensors:

    >>> batch = ([x], {'hello': {'world': y}}, ((z,),))
    >>> batch = delu.to(batch, device)

    .. note::
        Technically, the function simply traverses the input and applies
        `torch.Tensor.to` to tensors.

    Args:
        data: the collection of tensors. Nested non-tensor values are not allowed.
        args: the positional arguments for `torch.Tensor.to`
        kwargs: the keyword arguments for `torch.Tensor.to`
    Returns:
        the transformed data.
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


def cat(data: List[T], /, dim: int = 0) -> T:
    """Like `torch.cat`, but for collections of tensors.

    While `torch.cat` concatenates a sequence of tensors,
    `delu.to` concatenates a sequence of *collections* of tensors
    (tuples, named tuples, dictionaries, dataclasses and nested combinations thereof;
    **nested lists are not allowed**).

    **Usage**

    Common setup:

    >>> # First batch.
    >>> x1 = torch.randn(64, 10)
    >>> y1 = torch.randn(64)
    >>> # Second batch.
    >>> x2 = torch.randn(64, 10)
    >>> y2 = torch.randn(64)
    >>> # The last (incomplete) batch.
    >>> x3 = torch.randn(7, 10)
    >>> y3 = torch.randn(7)
    >>> # Total size.
    >>> len(x1) + len(x2) + len(x3)
    135

    `delu.cat` can be applied to tuples:

    >>> batches = [(x1, y1), (x2, y2), (x3, y3)]
    >>> X, Y = delu.cat(batches)
    >>> print(len(X), len(Y))
    135 135

    `delu.cat` can be applied to dictionaries:

    >>> batches = [
    ...     {'x': x1, 'y': y1},
    ...     {'x': x2, 'y': y2},
    ...     {'x': x3, 'y': y3},
    ... ]
    >>> result = delu.cat(batches)
    >>> print(isinstance(result, dict), len(result['x']), len(result['y']))
    True 135 135

    `delu.cat` can be applied to named tuples:

    >>> from typing import NamedTuple
    >>> class Data(NamedTuple):
    ...     x: torch.Tensor
    ...     y: torch.Tensor
    >>> batches = [Data(x1, y1), Data(x2, y2), Data(x3, y3)]
    >>> result = delu.cat(batches)
    >>> print(isinstance(result, Data), len(result.x), len(result.y))
    True 135 135

    `delu.cat` can be applied to dataclasses:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Data:
    ...     x: torch.Tensor
    ...     y: torch.Tensor
    >>> batches = [Data(x1, y1), Data(x2, y2), Data(x3, y3)]
    >>> result = delu.cat(batches)
    >>> print(isinstance(result, Data), len(result.x), len(result.y))
    True 135 135

    `delu.cat` can be applied to nested collections:

    >>> batches = [
    ...     (x1, {'a': {'b': y1}}),
    ...     (x2, {'a': {'b': y2}}),
    ...     (x3, {'a': {'b': y3}}),
    ... ]
    >>> X, Y_nested = delu.cat(batches)
    >>> print(len(X), len(Y_nested['a']['b']))
    135 135

    **`delu.cat` cannot be applied to lists:**

    >>> # This does not work. Instead, use tuples.
    >>> # batches = [[x1, y1], [x2, y2], [x3, y3]]
    >>> # delu.cat(batches)  # Error

    .. note::
        Technically, the function "transposes" the list of
        collections to a collection of lists and applies `torch.cat` to those lists.

    Args:
        data: the list of collections of tensors.
            All items of the list must be of the same type, structure and layout, only
            the ``dim`` dimension can vary (same as for `torch.cat`).
            All the "leaf" values must be of the type `torch.Tensor`.
        dim: the dimension over which the tensors are concatenated.
    Returns:
        The concatenated items of the list.
    """
    if not isinstance(data, list):
        raise ValueError('The input must be a list')
    if not data:
        raise ValueError('The input must be non-empty')

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


def iter_batches(
    data: T,
    /,
    batch_size: int,
    *,
    shuffle: bool = False,
    generator: Optional[torch.Generator] = None,
    drop_last: bool = False,
) -> Iterator[T]:
    """Iterate over a tensor or a collection of tensors by (random) batches.

    The function makes batches along the first dimension of the tensors in ``data``.

    .. note::
        `delu.iter_batches` is significantly faster for in-memory tensors
        than `torch.utils.data.DataLoader`, because, when building batches,
        it uses batched indexing instead of one-by-one indexing.

    **Usage**

    >>> X = torch.randn(12, 32)
    >>> Y = torch.randn(12)

    `delu.iter_batches` can be applied to tensors:

    >>> for x in delu.iter_batches(X, batch_size=5):
    ...     print(len(x))
    5
    5
    2

    `delu.iter_batches` can be applied to tuples:

    >>> # shuffle=True can be useful for training.
    >>> dataset = (X, Y)
    >>> for x, y in delu.iter_batches(dataset, batch_size=5, shuffle=True):
    ...     print(len(x), len(y))
    5 5
    5 5
    2 2
    >>> # Drop the last incomplete batch.
    >>> for x, y in delu.iter_batches(
    ...     dataset, batch_size=5, shuffle=True, drop_last=True
    ... ):
    ...     print(len(x), len(y))
    5 5
    5 5
    >>> # The last batch is complete, so drop_last=True does not have any effect.
    >>> batches = []
    >>> for x, y in delu.iter_batches(dataset, batch_size=6, drop_last=True):
    ...     print(len(x), len(y))
    ...     batches.append((x, y))
    6 6
    6 6

    By default, ``shuffle`` is set to `False`, i.e. the order of items is preserved:

    >>> X2, Y2 = delu.cat(list(delu.iter_batches((X, Y), batch_size=5)))
    >>> print((X == X2).all().item(), (Y == Y2).all().item())
    True True

    `delu.iter_batches` can be applied to dictionaries:

    >>> dataset = {'x': X, 'y': Y}
    >>> for batch in delu.iter_batches(dataset, batch_size=5, shuffle=True):
    ...     print(isinstance(batch, dict), len(batch['x']), len(batch['y']))
    True 5 5
    True 5 5
    True 2 2

    `delu.iter_batches` can be applied to named tuples:

    >>> from typing import NamedTuple
    >>> class Data(NamedTuple):
    ...     x: torch.Tensor
    ...     y: torch.Tensor
    >>> dataset = Data(X, Y)
    >>> for batch in delu.iter_batches(dataset, batch_size=5, shuffle=True):
    ...     # batch is an instance of
    ...     print(isinstance(batch, Data), len(batch.x), len(batch.y))
    True 5 5
    True 5 5
    True 2 2

    `delu.iter_batches` can be applied to dataclasses:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Data:
    ...     x: torch.Tensor
    ...     y: torch.Tensor
    >>> dataset = Data(X, Y)
    >>> for batch in delu.iter_batches(dataset, batch_size=5, shuffle=True):
    ...     print(isinstance(batch, Data), len(batch.x), len(batch.y))
    True 5 5
    True 5 5
    True 2 2

    Args:
        data: the tensor or the collection of tensors.
            If data is a collection, then the tensors must be of the same size
            along the first dimension.
        batch_size: the batch size. If ``drop_last`` is False,
            then the last batch can be smaller than ``batch_size``.
        shuffle: if True, iterate over random batches (without replacement),
            not sequentially.
        generator: when ``shuffle`` is True, passing ``generator`` makes the function
            reproducible.
        drop_last: when ``True`` and the last batch is smaller then ``batch_size``,
            then this last batch is not returned
            (in other words,
            same as the ``drop_last`` argument for `torch.utils.data.DataLoader`).
    Returns:
        the iterator over batches.
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


@deprecated('Instead, use `delu.cat`.')
def concat(*args, **kwargs):
    """
    ⚠️ **DEPRECATED** ⚠️ <DEPRECATION MESSAGE>
    """
    return cat(*args, **kwargs)
