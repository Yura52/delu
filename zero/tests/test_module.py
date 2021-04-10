import torch
import torch.nn as nn
from pytest import mark, raises

from zero.module import evaluation


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
        assert not torch.is_grad_enabled()  # type: ignore[code]
    assert torch.is_grad_enabled() == grad  # type: ignore[code]
