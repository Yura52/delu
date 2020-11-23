import math

import torch
from pytest import mark, raises, warns

from zero.training import ProgressTracker, evaluate, learn


@mark.parametrize('train', [False, True])
@mark.parametrize('grad', [False, True])
@mark.parametrize('n_models', range(3))
def test_evaluate(train, grad, n_models):
    if not n_models:
        with raises(AssertionError):
            with evaluate():
                pass
        return

    torch.set_grad_enabled(grad)
    models = [torch.nn.Linear(1, 1) for _ in range(n_models)]
    for x in models:
        x.train(train)
    with evaluate(*models):
        assert all(not x.training for x in models[:-1])
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled() == grad


def test_progress_tracker():
    score = -999999999

    # test initial state
    tracker = ProgressTracker(0)
    assert not tracker.success
    assert not tracker.fail

    # test successful update
    tracker.update(score)
    assert tracker.best_score == score
    assert tracker.success

    # test failed update
    tracker.update(score)
    assert tracker.best_score == score
    assert tracker.fail

    # test forget_bad_updates, reset
    tracker.forget_bad_updates()
    assert tracker.best_score == score
    tracker.reset()
    assert tracker.best_score is None
    assert not tracker.success and not tracker.fail

    # test positive patience
    tracker = ProgressTracker(1)
    tracker.update(score - 1)
    assert tracker.success
    tracker.update(score)
    assert tracker.success
    tracker.update(score)
    assert not tracker.success and not tracker.fail
    tracker.update(score)
    assert tracker.fail

    # test positive min_delta
    tracker = ProgressTracker(0, 2)
    tracker.update(score - 2)
    assert tracker.success
    tracker.update(score)
    assert tracker.fail
    tracker.reset()
    tracker.update(score - 3)
    tracker.update(score)
    assert tracker.success

    # patience=None
    tracker = ProgressTracker(None)
    for i in range(100):
        tracker.update(-i)
        assert not tracker.fail


@mark.parametrize('train', [False, True])
@mark.parametrize('star', [False, True])
def test_learn(train, star):
    model = torch.nn.Linear(3, 1)
    model.train(train)
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    loss_fn = (
        torch.nn.functional.mse_loss
        if star
        else lambda x: torch.nn.functional.mse_loss(*x)
    )
    f = lambda batch: batch.sum(1, keepdim=True)  # noqa

    def step(batch):
        assert model.training
        for x in model.parameters():
            assert not x.grad.bool().any()
        return model(batch), f(batch)

    batch = torch.randn(10, 3)
    model(batch).sum().backward()
    result = learn(model, optimizer, loss_fn, step, batch, star)
    assert (
        isinstance(result[0], float)
        and isinstance(result[1], tuple)
        and len(result[1]) == 2
    )
    # check optimizer.step()
    assert not torch.equal(result[1][0], model(batch))
    assert torch.equal(result[1][1], f(batch))

    for _ in range(100):
        learn(model, optimizer, loss_fn, step, batch, star)
    assert torch.nn.functional.mse_loss(model(batch), f(batch)).item() < 0.01


@mark.parametrize('value', [math.nan, math.inf])
@mark.parametrize('sign', [1, -1])
def test_learn_inf_nan(value, sign):
    model = torch.nn.Linear(3, 1)
    batch = torch.randn(10, 3) * sign * value
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    with warns(RuntimeWarning):
        learn(model, optimizer, torch.sum, model, batch)
