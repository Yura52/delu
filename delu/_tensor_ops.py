import dataclasses
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

import torch
import torch.nn as nn

from ._utils import deprecated

T = TypeVar('T')
K = TypeVar('K')


def to(obj: T, /, *args, **kwargs) -> T:
    """Change devices and data types of tensors and modules in an arbitrary Python object (like `torch.Tensor.to` / `torch.nn.Module.to`, but for any Python object).

    The two primary use cases for this function are changing the device and data types
    of tensors and modules that are a part of:

    - a complex Python object (e.g. a training state, checkpoint, etc.)
    - an object of an unknown type (when implementing generic pipelines)

    **Usage**

    >>> from dataclasses import dataclass
    >>>
    >>> class UserClass:
    ...     def __init__(self):
    ...         self.a = torch.randn(5)
    ...         self.b = ('Hello, world!', torch.randn(10))
    ...         self.c = nn.Linear(4, 7)
    ...
    >>> @dataclass
    >>> class UserDataclass:
    ...     d: List[UserClass]
    ...
    >>> data = (
    ...     torch.rand(3),
    ...     [{(False, 1): torch.tensor(1.0)}, 2.0],
    ...     UserDataclass([UserClass(), UserClass()]),
    ... )
    >>> delu.to(data, device='cpu', dtype=torch.float16)

    .. note::
        Technically, the function traverses the input ``data`` as follows:

        - for tensors/modules, `torch.Tensor.to`/`torch.nn.Module.to` is applied
          with the provided ``*args`` and ``**kwargs``; in particular, it means
          that tensors will be replaced with new tensors (in terms of Python `id`),
          but modules will be modified inplace;
        - for tuples, named tuples, lists, other sequences (see `typing.Sequence`),
          dictionaries and other mappings (see `typing.Mapping`),
          a new collection of the same type is returned,
          where `delu.to` is recursively applied
          to all values of the original collection;
        - in all other cases, the original object in terms of Python `id` is returned.
          If the object has attributes (defined in ``__dict__`` or ``__slots__``),
          then `delu.to` is recursively applied to all the attributes.

    Args:
        obj: the input object.
        args: the positional arguments for `torch.Tensor.to`/`torch.nn.Module.to`.
        kwargs: the keyword arguments for `torch.Tensor.to`/`torch.nn.Module.to`.

    Returns:
        the transformed object.
    """  # noqa: E501
    # mypy does not understand what is going on here, hence a lot of "type: ignore"

    if isinstance(obj, (torch.Tensor, nn.Module)):
        return obj.to(*args, **kwargs)  # type: ignore

    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj  # type: ignore

    elif isinstance(obj, Sequence):
        constructor = type(obj)
        if issubclass(constructor, tuple):
            # Handle named tuples.
            constructor = getattr(constructor, '_make', constructor)
        return constructor(to(x, *args, **kwargs) for x in obj)  # type: ignore

    elif isinstance(obj, Mapping):
        # Tensors can be keys.
        return type(obj)(
            (to(k, *args, **kwargs), to(v, *args, **kwargs)) for k, v in obj.items()
        )  # type: ignore

    else:
        for attr in obj.__slots__ if hasattr(obj, '__slots__') else obj.__dict__:
            try:
                setattr(obj, attr, to(getattr(obj, attr), *args, **kwargs))
            except Exception as err:
                raise RuntimeError(
                    f'Failed to update the attribute {attr}'
                    f' of the (perhaps, nested) value of the type {type(obj)}'
                    ' with the `delu.to` function'
                ) from err
        return obj


def cat(data: List[T], /, dim: int = 0) -> T:
    """Concatenate a sequence of collections of tensors (like `torch.cat`, but for collections of tensors).

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

    Args:
        data: the list of collections of tensors.
            All items of the list must be of the same type, structure and layout, only
            the ``dim`` dimension can vary (same as for `torch.cat`).
            All the "leaf" values must be of the type `torch.Tensor`.
        dim: the dimension along which the tensors are concatenated.
    Returns:
        The concatenated items of the list.
    """  # noqa: E501
    if not isinstance(data, list):
        raise ValueError('The input must be a list')
    if not data:
        raise ValueError('The input must be non-empty')

    first = data[0]

    if isinstance(first, torch.Tensor):
        return torch.cat(data, dim=dim)  # type: ignore

    elif isinstance(first, tuple):
        constructor = type(first)
        constructor = getattr(constructor, '_make', constructor)  # Handle named tuples.
        return constructor(
            cat([x[i] for x in data], dim=dim) for i in range(len(first))  # type: ignore
        )

    elif isinstance(first, dict):
        return type(first)((key, cat([x[key] for x in data], dim=dim)) for key in first)  # type: ignore

    elif dataclasses.is_dataclass(first):
        return type(first)(
            **{
                field.name: cat([getattr(x, field.name) for x in data], dim=dim)
                for field in dataclasses.fields(first)
            }
        )  # type: ignore

    else:
        raise ValueError(f'The collection type {type(first)} is not supported.')


def _make_index_batches(
    x: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    generator: Optional[torch.Generator],
    drop_last: bool,
) -> Iterable[torch.Tensor]:
    size = len(x)
    if not size:
        raise ValueError('data must not contain empty tensors')
    batch_indices = (
        torch.randperm(size, generator=generator, device=x.device)
        if shuffle
        else torch.arange(size, device=x.device)
    ).split(batch_size)
    return (
        batch_indices[:-1]
        if batch_indices and drop_last and len(batch_indices[-1]) < batch_size
        else batch_indices
    )


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
        data: the tensor or the non-empty collection of tensors.
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

    constructor: Callable[[Any], T]
    args = (batch_size, shuffle, generator, drop_last)

    if isinstance(data, torch.Tensor):
        item = data
        for idx in _make_index_batches(item, *args):
            yield data[idx]  # type: ignore

    elif isinstance(data, tuple):
        if not data:
            raise ValueError('data must be non-empty')
        item = data[0]
        for x in data:
            if not isinstance(x, torch.Tensor) or len(x) != len(item):
                raise ValueError(
                    'If data is a tuple, it must contain only tensors,'
                    ' and they must have the same first dimension'
                )
        constructor = type(data)  # type: ignore
        constructor = getattr(constructor, '_make', constructor)  # Handle named tuples.
        for idx in _make_index_batches(item, *args):
            yield constructor(x[idx] for x in data)

    elif isinstance(data, dict):
        if not data:
            raise ValueError('data must be non-empty')
        item = next(iter(data.values()))
        for x in data.values():
            if not isinstance(x, torch.Tensor) or len(x) != len(item):
                raise ValueError(
                    'If data is a dict, it must contain only tensors,'
                    ' and they must have the same first dimension'
                )
        constructor = type(data)  # type: ignore
        for idx in _make_index_batches(item, *args):
            yield constructor((k, v[idx]) for k, v in data.items())

    elif dataclasses.is_dataclass(data):
        fields = list(dataclasses.fields(data))
        if not fields:
            raise ValueError('data must be non-empty')
        item = getattr(data, fields[0].name)
        for field in fields:
            if field.type is not torch.Tensor:
                raise ValueError('All dataclass fields must be tensors.')
            if len(getattr(data, field.name)) != len(item):
                raise ValueError(
                    'All dataclass tensors must have the same first dimension.'
                )
        constructor = type(data)  # type: ignore
        for idx in _make_index_batches(item, *args):
            yield constructor(
                **{field.name: getattr(data, field.name)[idx] for field in fields}  # type: ignore
            )

    else:
        raise ValueError(f'The collection {type(data)} is not supported.')


@deprecated('Instead, use `delu.cat`.')
def concat(*args, **kwargs):
    """
    <DEPRECATION MESSAGE>
    """
    return cat(*args, **kwargs)
