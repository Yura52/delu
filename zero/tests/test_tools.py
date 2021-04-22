import pickle
from time import perf_counter, sleep

from pytest import approx, raises

from zero.tools import ProgressTracker, Timer


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


def test_timer():
    with raises(AssertionError):
        Timer().pause()

    # initial state, run
    timer = Timer()
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
    timer = Timer()
    timer.run()
    sleep(0.1)
    actual = timer()
    # the allowed deviation was obtained from manual runs on my laptop so the test may
    # behave differently on other hardware
    assert actual == approx(correct, abs=0.01)


def test_timer_context():
    with Timer() as timer:
        sleep(0.01)
    assert timer() > 0.01
    assert timer() == timer()

    timer = Timer()
    timer.run()
    sleep(0.01)
    timer.pause()
    with timer:
        sleep(0.01)
    assert timer() > 0.02
    assert timer() == timer()


def test_timer_pickle():
    timer = Timer()
    timer.run()
    sleep(0.01)
    timer.pause()
    value = timer()
    sleep(0.01)
    assert pickle.loads(pickle.dumps(timer))() == timer() == value


def test_timer_format():
    def format_seconds(x, *args, **kwargs):
        timer = Timer()
        timer.add(x)
        return timer.format(*args, **kwargs)

    assert format_seconds(1) == '0:00:01'
    assert format_seconds(1.1) == '0:00:01'
    assert format_seconds(1.1, round_=False) == '0:00:01.100000'
    assert format_seconds(7321, '%Hh %Mm %Ss') == '02h 02m 01s'
