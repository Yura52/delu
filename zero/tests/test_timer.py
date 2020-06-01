from time import perf_counter, sleep

from pytest import approx, raises

from zero.timer import Timer


def test_timer():
    # test initial state, start
    timer = Timer()
    sleep(0.001)
    assert not timer()
    timer.start()
    with raises(AssertionError):
        timer.start()
    assert timer()

    # test pause
    timer.pause()
    with raises(AssertionError):
        timer.pause()
    x = timer()
    sleep(0.001)
    assert timer() == x

    # test add, sub
    timer.add(1.0)
    assert timer() - x == approx(1)
    timer.sub(1.0)
    assert timer() == x

    # test resume
    timer.resume()
    with raises(AssertionError):
        timer.resume()
    sleep(0.001)
    assert timer() != x
    timer.pause()
    x = timer()
    sleep(0.001)
    assert timer() == x
    timer.resume()

    # test context manager
    with raises(AssertionError):
        with timer:
            pass
    with timer.pause():
        x = timer()
        sleep(0.001)
        assert timer() == x
    sleep(0.001)
    assert timer() != x

    # test reset
    timer.reset()
    assert not timer()


def test_measurements():
    x = perf_counter()
    sleep(0.1)
    correct = perf_counter() - x
    timer = Timer().start()
    sleep(0.1)
    actual = timer()
    # the allowed deviation was obtained from manual runs on my laptop so the test may
    # behave differently on other hardware
    assert actual == approx(correct, abs=0.01)
