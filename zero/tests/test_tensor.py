import torch

from zero._util import flatten
from zero.tensor import ibackward, to_device

from .util import Point, make_model, requires_gpu


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


@requires_gpu
def test_to_device():
    t = lambda x: torch.tensor(0, device=x)  # noqa
    cpu = torch.device('cpu')
    cuda = torch.device('cuda', 0)

    for x in cpu, cuda:
        data = t(x)
        assert to_device(data, x) is data
    assert to_device(t(cpu), cuda).device == cuda

    for Container in tuple, Point, list:
        constructor = Container._make if Container is Point else Container
        for device in [cpu, cuda]:
            data = constructor([t(cpu), t(cpu)])
            out = to_device(data, device)
            assert isinstance(out, Container)
            assert all(x.device == device for x in out)
            if device == cpu:
                for x, y in zip(out, data):
                    assert x is y

    data = [t(cpu), t(cpu)]
    for x, y in zip(to_device(data, cpu), data):
        assert x is y
    assert all(x.device == cuda for x in to_device(data, cuda))

    data = {
        'a': [t(cpu), (t(cpu), t(cpu))],
        'b': {'c': {'d': [[[t(cpu)]]]}},
        'c': Point(t(cpu), {'d': t(cpu)}),
    }
    for x, y in zip(flatten(to_device(data, cpu)), flatten(data)):
        assert x is y
    for x, y in zip(flatten(to_device(data, cuda)), flatten(data)):
        assert x.device == cuda
        assert type(x) == type(y)
