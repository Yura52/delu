from collections import namedtuple

import torch as tr
from pytest import mark, raises

from zero.training import EvalContext, TrainContext

from .util import ObjectCounter

Model = namedtuple('Model', ['model', 'weight', 'bias', 'loss', 'optimizer'])


@mark.parametrize('train', [False, True])
@mark.parametrize('grad', [False, True])
@mark.parametrize('n_models', range(3))
def test_train_context(train, grad, n_models):
    data = tr.ones(4, 3, dtype=tr.float32)

    def make_model():
        model = tr.nn.Linear(data.shape[1], 1)
        return Model(
            model,
            model.weight.clone(),
            model.bias.clone(),
            lambda: model(data).sum(),
            tr.optim.SGD(model.parameters(), 0.0001),
        )

    if not n_models:
        loss = make_model().loss
        tc = TrainContext([], [])
        with raises(AssertionError):
            tc.backward(loss())
        return

    models = [make_model() for _ in range(n_models)]
    for x in models:
        x.model.train(train)

    tr.set_grad_enabled(grad)
    if grad:
        for x in models:
            x.loss().backward()

    def check_before(m):
        assert m.model.training
        assert tr.is_grad_enabled()
        if grad:
            # in this case .backward was called before entering a context, so zero_grad
            # should take effect (without .backward .grad is None)
            assert not m.model.weight.grad.bool().any()
            assert not m.model.bias.grad.bool().any()

    def check_after_0(m, tc):
        assert m.model.weight.grad is not None
        assert m.model.bias.grad is not None
        with raises(AssertionError):
            tc.backward(m.loss())

    def check_after_1(m):
        assert not tr.equal(m.model.weight.data, m.weight)
        assert not tr.equal(m.model.bias.data, m.bias)
        assert tr.is_grad_enabled() == grad
        m.model.training == train

    x = models[-1]
    with TrainContext(x.model, x.optimizer) as tc:
        check_before(x)
        tc.backward(x.loss())
        check_after_0(x, tc)
    check_after_1(x)

    models = models[:-1]
    with TrainContext([x.model for x in models], [x.optimizer for x in models]) as tc:
        for x in models:
            check_before(x)
        tc.backward([x.loss() for x in models])
        for x in models:
            check_after_0(x, tc)
    for x in models:
        check_after_1(x)

    # test an unusual form of losses
    models = [make_model() for _ in range(n_models)]
    with TrainContext([x.model for x in models], [x.optimizer for x in models]) as tc:
        tc.backward(
            [
                ({i: [x.model(data).sum()]}, [x.model(data).sum()])
                for i, x in enumerate(models)
            ]
        )
    for x in models:
        check_after_1(x)


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
