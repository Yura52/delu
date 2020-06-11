from collections import namedtuple

import torch as tr
from pytest import mark, raises

from zero.training import EvalContext, TrainContext

from .util import ObjectCounter

Model = namedtuple('Model', ['model', 'weight', 'bias', 'loss', 'optimizer'])


def make_model(data):
    model = tr.nn.Linear(data.shape[1], 1)
    return Model(
        model,
        model.weight.clone(),
        model.bias.clone(),
        lambda: model(data).sum(),
        tr.optim.SGD(model.parameters(), 0.0001),
    )


def test_train_context_incorrect_usage():
    # negative n_backwards
    with raises(AssertionError):
        TrainContext([], [], -1)

    data = tr.ones(4, 3, dtype=tr.float32)

    # backward outside of a context
    tc = TrainContext([], [])
    with raises(AssertionError):
        tc.backward(make_model(data).loss())

    for n_backwards in range(1, 4):
        # not enough backwards
        for actual_n_backwards in range(n_backwards - 1):
            with raises(AssertionError):
                with TrainContext([], [], n_backwards) as tctx:
                    for _ in range(actual_n_backwards):
                        tctx.backward(make_model(data).loss())
        # too many backwards
        with TrainContext([], [], n_backwards) as tctx:
            for _ in range(n_backwards):
                tctx.backward(make_model(data).loss())
            with raises(AssertionError):
                tctx.backward(make_model(data).loss())


@mark.parametrize('train', [False, True])
@mark.parametrize('grad', [False, True])
def test_train_context_train_grad(train, grad):
    data = tr.ones(4, 3, dtype=tr.float32)
    models = [make_model(data) for _ in range(3)]

    for x in models:
        x.model.train(train)
    tr.set_grad_enabled(grad)
    if grad:
        # for testing that .zero_grad is called in __enter__
        for x in models:
            x.loss().backward()

    def check_inside_context(x):
        assert x.model.training
        assert tr.is_grad_enabled()
        if grad:
            # in this case .backward is called before entering a context, so .zero_grad
            # should take effect (without .backward .grad is None)
            assert not x.model.weight.grad.bool().any()
            assert not x.model.bias.grad.bool().any()
        loss = x.loss()
        loss_item = tctx.backward(loss)
        assert loss_item == loss.item()
        assert x.model.weight.grad is not None
        assert x.model.bias.grad is not None

    def check_exit(x):
        assert not tr.equal(x.model.weight.data, x.weight)
        assert not tr.equal(x.model.bias.data, x.bias)
        assert tr.is_grad_enabled() == grad
        x.model.training == train

    x = models[-1]
    with TrainContext(x.model, x.optimizer) as tctx:
        check_inside_context(x)
    check_exit(x)

    two_models = models[:-1]
    with TrainContext(
        [x.model for x in two_models], [x.optimizer for x in two_models], 2
    ) as tctx:
        for x in two_models:
            check_inside_context(x)
    for x in two_models:
        check_exit(x)

    # test *args, **kwargs
    x = models[-1]
    with TrainContext(x.model, x.optimizer, 3) as tctx:
        loss = x.loss()
        tctx.backward(loss, None, True)  # gradients, retain_graph
        tctx.backward(loss, None, retain_graph=True)  # gradients, retain_graph
        tctx.backward(loss)


def test_train_context_exception():
    data = tr.randn(1, 1)
    x = make_model(data)
    with raises(RuntimeError):
        with TrainContext(x.model, x.optimizer) as tc:
            tc.backward(x.loss())
            raise RuntimeError
        # step must not be taken
        assert tr.equal(x.model.weight, x.weight)
        assert tr.equal(x.model.bias, x.bias)


@mark.parametrize('train', [False, True])
@mark.parametrize('grad', [False, True])
@mark.parametrize('n_models', range(3))
def test_eval_context(train, grad, n_models):
    tr.set_grad_enabled(grad)
    with EvalContext([], None):
        assert not tr.is_grad_enabled()
    assert tr.is_grad_enabled() == grad

    if not n_models:
        return

    models = [tr.nn.Linear(1, 1) for _ in range(n_models)]
    for x in models:
        x.train(train)
    metric = ObjectCounter(1)

    x = models[-1]
    metric.update(([1], None))
    with EvalContext(x, metric):
        assert not x.training
        assert metric.empty()
    assert x.training == train

    metric.update(([1], None))
    with EvalContext(models[:-1], metric):
        assert all(not x.training for x in models[:-1])
        assert metric.empty()
    assert all(x.training == train for x in models)
