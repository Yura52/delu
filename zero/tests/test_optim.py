import inspect

import torch
from pytest import mark

import zero.optim

from .util import make_model


def _is_optimizer(name):
    cls = getattr(zero.optim, name)
    return (
        cls is not None
        and isinstance(cls, type)
        and issubclass(cls, torch.optim.Optimizer)
    )


OPTIMIZER_NAMES = list(filter(_is_optimizer, zero.optim.__all__))


@mark.parametrize('name', OPTIMIZER_NAMES)
def test_zero_optimizer_requirements(name):
    assert name in dir(torch.optim)
    assert not name.startswith(('_', 'Base'))
    assert not name.endswith('Base')
    assert 'Mixin' not in name

    cls = getattr(torch.optim, name)
    assert inspect.isclass(cls)
    assert cls is not torch.optim.Optimizer
    assert issubclass(cls, torch.optim.Optimizer)
    assert inspect.signature(cls.step).parameters['closure'].default is None


@mark.parametrize('name', OPTIMIZER_NAMES)
@mark.parametrize('error', [False, True])
def test_zero_optimizer(name, error):
    if 'sparse' in name.lower():
        return

    cls = getattr(torch.optim, name)
    zero_cls = getattr(zero.optim, name)
    data = torch.ones(4, 3, dtype=torch.float32)
    torch.manual_seed(0)
    model = make_model(data)
    torch.manual_seed(0)
    zero_model = model = make_model(data)
    lr = 0.1
    optimizer = cls(model.model.parameters(), lr)
    zero_optimizer = zero_cls(zero_model.model.parameters(), lr)

    def zip_parameters():
        return zip(model.model.parameters(), zero_model.model.parameters())

    assert all({x.grad, y.grad} == {None} for x, y in zip_parameters())
    model.loss_fn().backward()
    zero_model.loss_fn().backward()

    def body():
        optimizer.zero_grad()
        for x, y in zip_parameters():
            assert torch.equal(x.grad, y.grad)
        model.loss_fn().backward()
        zero_model.loss_fn().backward()

    if error:
        # check that .step is not performed when an exception is raised
        try:
            with zero_optimizer:
                body()
                raise ZeroDivisionError()
            assert torch.equal(model.model.weight.data, model.weight.data)
            assert torch.equal(model.model.bias.data, model.bias.data)
        except ZeroDivisionError:
            pass

    else:
        for _ in range(10):
            with zero_optimizer:
                body()
            optimizer.step()
            for x, y in zip_parameters():
                assert torch.equal(x, y)
