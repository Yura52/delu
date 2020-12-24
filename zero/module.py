__all__ = ['call', 'ecall', 'evaluation']

import contextlib
import typing as ty

import torch
import torch.nn as nn

T = ty.TypeVar('T')


def _to_device(value: T, device: torch.device) -> T:
    # TODO: support dataclasses
    if isinstance(value, torch.Tensor):
        return value.to(device)  # type: ignore
    elif isinstance(value, (tuple, list)):
        cls = type(value)
        data = (_to_device(x, device) for x in value)
        is_namedtuple = all(
            hasattr(value, x) for x in ['_fields', '_replace', '_asdict']
        )
        return cls(*data) if is_namedtuple else cls(data)  # type: ignore
    elif isinstance(value, dict):
        return cls(value)((k, _to_device(v, device)) for k, v in value.items())  # type: ignore
    else:
        return value


def call(module: ty.Union[nn.Module, nn.DataParallel], *args, **kwargs):
    """Move arguments to module's device and call the module with these arguments.

    With this function you don't have to do the following anymore:
    - pass the model's device everywhere as an additional argument along with the model
    - infer the model's device by hand
    - move model's arguments to the correct device before the call

    Args:
        module: the module. If an instance of `torch.nn.DataParallel`, then
            arguments are passed to the module as is.
        *args:
        **kwargs:

    Returns:
        result: the output of :code:`module(*args, **kwargs)` after the arguments are
            moved to the module's device.

    Note:
        The module's device is inferred as the device of its one randomly selected
        parameter. So, the function works only for cases when all the parameters of the
        module are located on the same device.

    Note:
        The transfer happens only to tensors and simple containers containing tensors
        (tuples, lists, named tuples and dictionaries). Other values are not changed
        anyhow. For example, the following call is successfully handled by `zero.call`::

            # all tensor_X variables will be moved to the module's device
            zero.call(
                model,
                tensor_0,
                1,
                'hello',
                [tensor_1, True, (tensor_2, False)],
                {'world': [[[tensor_3]]]},
                abc=my_namedtuple_with_tensors
            )

        However, if the arguments include instances of custom classes and some of their
        fields are tensors that need to be moved than this is not the case for
        `zero.call`.

    See also:

        - `ecall`

    Examples:
        .. testcode::

            model = torch.nn.Linear(3, 5)
            ...  # model is moved to some device here
            x = torch.randn(4, 3)
            zero.call(model, x)  # no need to move `x` to the model's device

    """
    if isinstance(module, nn.DataParallel):
        return module(*args, **kwargs)
    device = next(module.parameters()).device
    return module(*_to_device(args, device), **_to_device(kwargs, device))


@contextlib.contextmanager
def evaluation(*modules: nn.Module):
    """Context-manager for models evaluation.

    Warning:
        The function must be used only as a context manager as shown below in the
        examples. The behaviour for call without the `with` keyword is unspecified.

    This code...::

        model.eval()
        with torch.no_grad():
            ...

    ...is equivalent to ::

        with evaluation(model):
            ...

    Args:
        modules

    See also:

        - `ecall`

    Examples:
        .. testcode::

            a = torch.nn.Linear(1, 1)
            b = torch.nn.Linear(2, 2)
            with evaluation(a):
                ...
            with evaluation(a, b):
                ...

        .. testcode::

            model = torch.nn.Linear(1, 1)
            for grad in False, True:
                for train in False, True:
                    torch.set_grad_enabled(grad)
                    model.train(train)
                    with evaluation(model):
                        assert not model.training
                        assert not torch.is_grad_enabled()
                        ...
                    assert torch.is_grad_enabled() == grad_before_context
                    # model.training is unspecified here
    """
    assert modules
    for x in modules:
        x.eval()
    no_grad_context = torch.no_grad()
    no_grad_context.__enter__()
    try:
        yield
    finally:
        no_grad_context.__exit__(None, None, None)


def ecall(module: ty.Union[nn.Module, nn.DataParallel], *args, **kwargs):
    """Call the module (torch.no_grad() + module.eval() + input.to(device)).

    The function:

    1. switches the module to the evaluation mode
    2. turns off gradients
    3. moves the arguments to the module's device
    4. calls the module and returns the result

    In fact, the function is just a shortcut for the combination of `evaluation` and
    `call` (hence, all constraints of the `call` function are inherited, see the its
    documentation for details), i.e.::

        result = ecall(model, x)

        # is equivalent to:

        with evaluation(model):
            result = call(model, x)

    Args:
        module:
        args:
        kwargs:

    Returns:
        result:

    See also:

        - `evaluation`
        - `call`
    """
    with evaluation(module):
        return call(module, *args, **kwargs)
