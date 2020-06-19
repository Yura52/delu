import torch
from pytest import mark, raises

from zero.training import Eval, ibackward

from .util import make_model


@mark.parametrize('train', [False, True])
@mark.parametrize('grad', [False, True])
@mark.parametrize('n_models', range(3))
def test_eval(train, grad, n_models):
    if not n_models:
        with raises(AssertionError):
            Eval()
        return

    torch.set_grad_enabled(grad)
    models = [torch.nn.Linear(1, 1) for _ in range(n_models)]
    for x in models:
        x.train(train)
    with Eval(*models):
        assert all(not x.training for x in models[:-1])
        assert not torch.is_grad_enabled()
    assert all(x.training == train for x in models)
    assert torch.is_grad_enabled() == grad


def test_ibackward():
    data = torch.ones(4, 3, dtype=torch.float32)
    model = make_model(data)

    def check_reset():
        assert all(x.grad is not None for x in model.model.parameters())
        for x in model.model.parameters():
            x.grad = None

    loss = model.loss_fn()
    ibackward(loss, None, True)  # gradients, retain_graph
    check_reset()
    ibackward(loss, None, retain_graph=True)  # gradients, retain_graph
    check_reset()
    value = ibackward(loss)
    check_reset()
    assert value == loss.item()
