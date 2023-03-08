import dataclasses
import itertools
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, Tuple, TypeVar, Union

import numpy as np
import torch

from ._utils import is_namedtuple
from .data import make_index_dataloader

T = TypeVar('T')


def to(data: T, *args, **kwargs) -> T:
    """Call (recursively) `torch.Tensor.to` (set device, dtype, etc.).

    TODO
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


def concat(iterable: Iterable[T]) -> T:
    """Concatenate items (tensors, numpy-arrays, tuples, dicts etc.) along the first
    dimension.

    `concat` is a more general version ``torch.cat(..., dim=0)``. It works not
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
    and functions to batches (e.g. to ``DataLoader``)::

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
        - in all examples there is ``data`` - a list of batches; in fact, it can be
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
        if is_namedtuple(first)
        else type(first)(concat_fn(x[i] for x in data) for i, _ in enumerate(first))
        if isinstance(first, tuple)
        else type(first)((key, concat_fn(x[key] for x in data)) for key in first)
        if isinstance(first, dict)
        else concat_fn(data)
    )


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

    Note:
        If you want to infinitely iterate over batches, wrap the function in
        ``while True:``.

    Warning:
        Numpy-arrays are not supported because of how they behave when indexed by a
        torch tensor of the size 1. For details, see
        `the issue <https://github.com/numpy/numpy/issues/16543>`_

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
