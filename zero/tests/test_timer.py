from time import perf_counter, sleep

from pytest import approx

from zero.timer import Timer


def test_timer():
    # initial state, start
    timer = Timer()
    sleep(0.001)
    assert not timer()
    timer.start()
    assert timer()

    # stop
    timer.stop()
    timer.stop()  # two stops in a row
    x = timer()
    sleep(0.001)
    assert timer() == x

    # add, sub
    timer.stop()
    timer.add(1.0)
    assert timer() - x == approx(1)
    timer.sub(1.0)
    assert timer() == x

    # start
    timer.stop()
    x = timer()
    timer.start()
    timer.start()  # two starts in a row
    assert timer() != x
    timer.stop()
    x = timer()
    sleep(0.001)
    assert timer() == x
    timer.start()

    # reset
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
