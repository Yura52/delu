import torch
import torch.nn as nn
from pytest import mark, raises

from zero.learning import evaluation


@mark.parametrize('train', [False, True])
@mark.parametrize('grad', [False, True])
@mark.parametrize('n_models', range(3))
def test_evaluation(train, grad, n_models):
    if not n_models:
        with raises(AssertionError):
            with evaluation():
                pass
        return

    torch.set_grad_enabled(grad)
    models = [nn.Linear(1, 1) for _ in range(n_models)]
    for x in models:
        x.train(train)
    with evaluation(*models):
        assert all(not x.training for x in models[:-1])
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled() == grad
    for x in models:
        x.train(train)

    @evaluation(*models)
    def f():
        assert all(not x.training for x in models[:-1])
        assert not torch.is_grad_enabled()
        for x in models:
            x.train(train)

    for _ in range(3):
        f()
        assert torch.is_grad_enabled() == grad
