from collections import namedtuple

import torch
from pytest import mark, raises

from zero.training import Eval, Train, backward

from .util import ObjectCounter

Model = namedtuple('Model', ['model', 'weight', 'bias', 'loss', 'optimizer'])


def make_model(data):
    model = torch.nn.Linear(data.shape[1], 1)
    return Model(
        model,
        model.weight.clone(),
        model.bias.clone(),
        lambda: model(data).sum(),
        torch.optim.SGD(model.parameters(), 0.0001),
    )


@mark.parametrize('train', [False, True])
@mark.parametrize('grad', [False, True])
def test_train_context_train_grad(train, grad):
    data = torch.ones(4, 3, dtype=torch.float32)
    models = [make_model(data) for _ in range(3)]

    for x in models:
        x.model.train(train)
    torch.set_grad_enabled(grad)
    if grad:
        # for testing that .zero_grad is called in __enter__
        for x in models:
            x.loss().backward()

    def check_inside_context(x):
        assert x.model.training
        assert torch.is_grad_enabled()
        if grad:
            # in this case .backward is called before entering a context, so .zero_grad
            # should take effect (without .backward .grad is None)
            assert not x.model.weight.grad.bool().any()
            assert not x.model.bias.grad.bool().any()
        x.loss().backward()
        assert x.model.weight.grad is not None
        assert x.model.bias.grad is not None

    def check_exit(x):
        assert not torch.equal(x.model.weight.data, x.weight)
        assert not torch.equal(x.model.bias.data, x.bias)
        assert torch.is_grad_enabled() == grad
        x.model.training == train

    x = models[-1]
    with Train(x.model, x.optimizer):
        check_inside_context(x)
    check_exit(x)

    two_models = models[:-1]
    with Train([x.model for x in two_models], [x.optimizer for x in two_models]):
        for x in two_models:
            check_inside_context(x)
    for x in two_models:
        check_exit(x)


def test_train_context_exception():
    data = torch.randn(1, 1)
    x = make_model(data)
    with raises(RuntimeError):
        with Train(x.model, x.optimizer):
            x.loss().backward()
            raise RuntimeError
        # step must not be taken
        assert torch.equal(x.model.weight, x.weight)
        assert torch.equal(x.model.bias, x.bias)


@mark.parametrize('train', [False, True])
@mark.parametrize('grad', [False, True])
@mark.parametrize('n_models', range(3))
def test_eval_context(train, grad, n_models):
    torch.set_grad_enabled(grad)
    with Eval([], None):
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled() == grad

    if not n_models:
        return

    models = [torch.nn.Linear(1, 1) for _ in range(n_models)]
    for x in models:
        x.train(train)
    metric = ObjectCounter(1)

    x = models[-1]
    metric.update(([1], None))
    with Eval(x, metric):
        assert not x.training
        assert metric.empty()
    assert x.training == train

    metric.update(([1], None))
    with Eval(models[:-1], metric):
        assert all(not x.training for x in models[:-1])
        assert metric.empty()
    assert all(x.training == train for x in models)


def test_backward():
    data = torch.ones(4, 3, dtype=torch.float32)
    model = make_model(data)

    loss = model.loss()
    backward(loss, None, True)  # gradients, retain_graph
    backward(loss, None, retain_graph=True)  # gradients, retain_graph
    value = backward(loss)
    assert value == loss.item()
