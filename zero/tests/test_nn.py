from pytest import raises

import zero


def test_lambda():
    assert zero.nn.Lambda(lambda: 0)() == 0
    assert zero.nn.Lambda(lambda x: x)(1) == 1
    assert zero.nn.Lambda(lambda x, y, z: x + y + z)(1, 2, z=3) == 6
    with raises(TypeError):
        zero.nn.Lambda(lambda x: x)()
