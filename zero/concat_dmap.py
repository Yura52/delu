"""Batchwise application of models and functions.

TL;DR: if you have an iterable that contains/produces batches of some kind
(tensors, numpy-arrays, tuples/dictionaries thereof and other not-too-specific content),
then use `concat` to concatenate all the items. The most obvious case is application of
models and functions to batches (for example, application of model to batches produced
by :code:`DataLoader`):

.. code-block::

    whole_result = concat(map(model_or_fn, iterable))

Additionally, if every input needs to be moved to :code:`input_device` and/or
intermediate results need to be moved to :code:`output_device`, then use `dmap` instead
of `map`:

.. code-block::

    whole_result = concat(dmap(model_or_fn, iterable, input_device, output_device))

To be more specific:

.. code-block::

    def step(batch):
        X, y = batch
        return model(X), y

    loader = DataLoader(dataset, ...)  # produces tuples (X_batch, y_batch)
    model.to(device)  # prepare model
    ...
    # apply model to the whole dataset
    y_pred, y = concat(dmap(model, loader, device, 'cpu'))
    assert len(y_pred) == len(dataset) and len(y) == len(dataset)

    # or
    def step(batch):
        X, y = batch
        return {'y_pred': model(X), 'y': y}

    # no changes:
    result = concat(dmap(model, loader, device, 'cpu'))
    assert result['y_pred'] == len(dataset) and len(result['y']) == len(dataset)
"""

__all__ = ['concat', 'dmap']

import itertools
from typing import Any, Callable, Dict, Iterable, Iterator, Tuple, Union

import numpy as np
import torch

from ._util import is_namedtuple
from .tensor import to_device
from .types import Device, S, T


def concat(iterable: Iterable[T]) -> Union[S, Tuple[S, ...], Dict[Any, S]]:
    """Concatenate items of the iterable along the first dimension.

    Use intuition and examples (see below) to understand what the function does.
    Technical specification is unlikely to help here, so there is none :)

    Args:
        iterable: items **of the same structure**
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
        exceeded. Note that manual implementation would involve the same conversion,
        just keep this in mind when using the function.

    Examples:
        How to read the examples:

        - the mental model for understanding the following examples is "concatenating
            data for 3 batches of sizes (2, 2, 3)". Note that sizes of batches are
            allowed to vary, but the structure is always the same
        - in all examples there is :code:`data` - a list of batches; in fact, it can be
            any "iterable of batches", including iterators and generators; the list is
            chosen to simplify the demonstration

        1-D example:

        .. testcode ::

            result = concat([
                torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4, 5, 6])
            ])
            assert torch.equal(result, torch.tensor([0, 1, 2, 3, 4, 5, 6]))

        2-D example:

        .. testcode ::

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

        .. testcode ::

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


def dmap(
    fn: Callable[..., S],
    iterable: Iterable[T],
    in_device: Device = None,
    out_device: Device = None,
    non_blocking: bool = False,
    star: bool = False,
) -> Iterator[S]:
    """Devices-aware version of `map`.

    Apply the function to all items of the iterable in a lazy devices-aware manner.

    Args:
        fn:
        iterable:
        in_device: if not `None`, then every item will be moved to this device using
            `~zero.tensor.to_device` before applying :code:`fn`
        out_device: if not `None`, then every output of `fn` will be moved to this
            device using `~zero.tensor.to_device`. Usually, the value `cpu` is used
            when intermidiate results do not fit into GPU/XLA memory.
        non_blocking: argument for `torch.Tensor.to`
        star: if `True`, then :code:`fn` is applied like this: :code:`fn(*item)` instead
            of :code:`fn(item)`
    Returns:
        Iterator over outputs of :code:`fn`.

    Examples:
        .. code-block::

            result = concat(dmap(step_fn, dataloader, model_device, 'cpu'))
    """

    def wrapper(x):
        if in_device is not None:
            x = to_device(x, in_device, non_blocking)
        result = fn(*x) if star else fn(x)
        if out_device is not None:
            # mypy: NaN
            result = to_device(result, out_device, non_blocking)  # type: ignore
        return result

    return map(wrapper, iterable)
