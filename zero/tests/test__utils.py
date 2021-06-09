import pickle
import random
from time import perf_counter, sleep

import numpy as np
import torch
import torch.nn as nn
from pytest import approx, mark, raises

import zero


def test_progress_tracker():
    score = -999999999

    # test initial state
    tracker = zero.ProgressTracker(0)
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
    tracker = zero.ProgressTracker(1)
    tracker.update(score - 1)
    assert tracker.success
    tracker.update(score)
    assert tracker.success
    tracker.update(score)
    assert not tracker.success and not tracker.fail
    tracker.update(score)
    assert tracker.fail

    # test positive min_delta
    tracker = zero.ProgressTracker(0, 2)
    tracker.update(score - 2)
    assert tracker.success
    tracker.update(score)
    assert tracker.fail
    tracker.reset()
    tracker.update(score - 3)
    tracker.update(score)
    assert tracker.success

    # patience=None
    tracker = zero.ProgressTracker(None)
    for i in range(100):
        tracker.update(-i)
        assert not tracker.fail


def test_timer():
    with raises(AssertionError):
        zero.Timer().pause()

    # initial state, run
    timer = zero.Timer()
    sleep(0.001)
    assert not timer()
    timer.run()
    assert timer()

    # pause
    timer.pause()
    timer.pause()  # two pauses in a row
    x = timer()
    sleep(0.001)
    assert timer() == x

    # add, sub
    timer.pause()
    with raises(AssertionError):
        timer.add(-1.0)
    timer.add(1.0)
    assert timer() - x == approx(1)
    with raises(AssertionError):
        timer.sub(-1.0)
    timer.sub(1.0)
    assert timer() == x

    # run
    timer.pause()
    x = timer()
    timer.run()
    timer.run()  # two runs in a row
    assert timer() != x
    timer.pause()
    x = timer()
    sleep(0.001)
    assert timer() == x
    timer.run()

    # reset
    timer.reset()
    assert not timer()


def test_timer_measurements():
    x = perf_counter()
    sleep(0.1)
    correct = perf_counter() - x
    timer = zero.Timer()
    timer.run()
    sleep(0.1)
    actual = timer()
    # the allowed deviation was obtained from manual runs on my laptop so the test may
    # behave differently on other hardware
    assert actual == approx(correct, abs=0.01)


def test_timer_context():
    with zero.Timer() as timer:
        sleep(0.01)
    assert timer() > 0.01
    assert timer() == timer()

    timer = zero.Timer()
    timer.run()
    sleep(0.01)
    timer.pause()
    with timer:
        sleep(0.01)
    assert timer() > 0.02
    assert timer() == timer()


def test_timer_pickle():
    timer = zero.Timer()
    timer.run()
    sleep(0.01)
    timer.pause()
    value = timer()
    sleep(0.01)
    assert pickle.loads(pickle.dumps(timer))() == timer() == value


def test_timer_format():
    def make_timer(x):
        timer = zero.Timer()
        timer.add(x)
        return timer

    assert str(make_timer(1)) == '0:00:01'
    assert str(make_timer(1.1)) == '0:00:01'
    assert make_timer(7321).format('%Hh %Mm %Ss') == '02h 02m 01s'


@mark.parametrize('train', [False, True])
@mark.parametrize('grad', [False, True])
@mark.parametrize('n_models', range(3))
def test_evaluation(train, grad, n_models):
    if not n_models:
        with raises(AssertionError):
            with zero.evaluation():
                pass
        return

    torch.set_grad_enabled(grad)
    models = [nn.Linear(1, 1) for _ in range(n_models)]
    for x in models:
        x.train(train)
    with zero.evaluation(*models):
        assert all(not x.training for x in models[:-1])
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled() == grad
    for x in models:
        x.train(train)

    @zero.evaluation(*models)
    def f():
        assert all(not x.training for x in models[:-1])
        assert not torch.is_grad_enabled()
        for x in models:
            x.train(train)

    for _ in range(3):
        f()
        assert torch.is_grad_enabled() == grad


def test_improve_reproducibility():
    def f():
        upper_bound = 100
        return [
            random.randint(0, upper_bound),
            np.random.randint(upper_bound),
            torch.randint(upper_bound, (1,))[0].item(),
        ]

    for seed in [None, 0, 1, 2]:
        seed = zero.improve_reproducibility(seed)
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
        results = f()
        zero.random.seed(seed)
        assert results == f()
