from pytest import raises

import delu


def test_lambda():
    assert delu.nn.Lambda(lambda: 0)() == 0
    assert delu.nn.Lambda(lambda x: x)(1) == 1
    assert delu.nn.Lambda(lambda x, y, z: x + y + z)(1, 2, z=3) == 6
    with raises(TypeError):
        delu.nn.Lambda(lambda x: x)()
