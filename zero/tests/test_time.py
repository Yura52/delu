from time import perf_counter, sleep

from pytest import approx, raises

from zero.time import Timer, format_seconds


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
    timer = Timer().run()
    sleep(0.1)
    actual = timer()
    # the allowed deviation was obtained from manual runs on my laptop so the test may
    # behave differently on other hardware
    assert actual == approx(correct, abs=0.01)


def test_format_seconds():
    assert format_seconds(1) == '00h 00m 01s'
    assert format_seconds(1.1) == '00h 00m 01s'
    assert format_seconds(1, '%S%S%S') == '010101'
