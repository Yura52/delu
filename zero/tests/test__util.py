from zero._util import flatten, is_namedtuple, traverse

from .util import Point


def test_is_namedtuple():
    assert not is_namedtuple(())
    assert is_namedtuple(Point(1, 2))


def test_traverse():
    def fn(x):
        return x * 2

    assert traverse(fn, 3) == 6
    assert traverse(fn, 'abc') == 'abcabc'

    assert traverse(fn, ()) == ()
    assert traverse(fn, []) == []
    assert traverse(fn, {}) == {}

    assert traverse(fn, (3,)) == (6,)
    assert traverse(fn, [3]) == [6]
    assert traverse(fn, {'a': 3}) == {'a': 6}
    assert traverse(fn, (3, 4)) == (6, 8)
    x = traverse(fn, Point(3, 4))
    assert isinstance(x, Point) and x == (6, 8)
    assert traverse(fn, [3, 4]) == [6, 8]
    assert traverse(fn, {'a': 3, 'b': 4}) == {'a': 6, 'b': 8}

    assert traverse(fn, ([],)) == ([],)
    assert traverse(fn, [()]) == [()]
    assert traverse(fn, {'a': []}) == {'a': []}

    assert traverse(fn, ([1, 2], (3, 4))) == ([2, 4], (6, 8))
    assert traverse(fn, [(), [1, 2], {'a': 3, 'b': 4}]) == [
        (),
        [2, 4],
        {'a': 6, 'b': 8},
    ]

    input_ = {
        'a': [{'b': 1}, {'c': (2, 3)}],
        'b': {'c': {'d': {'e': 4}}},
        'c': [[5, 6, (7, 8), [[[9]]]]],
        'd': Point(Point(3, 4), Point(3, 4)),
    }
    correct = {
        'a': [{'b': 2}, {'c': (4, 6)}],
        'b': {'c': {'d': {'e': 8}}},
        'c': [[10, 12, (14, 16), [[[18]]]]],
        'd': Point(Point(6, 8), Point(6, 8)),
    }
    actual = traverse(fn, input_)
    assert (
        actual == correct
        and isinstance(actual['d'], Point)
        and isinstance(actual['d'].x, Point)
        and isinstance(actual['d'].y, Point)
    )


def test_flatten():
    data = {'a': 1, 'b': [2, 3], 'c': [4, (5, 6), [[[7], [], ['abc']]]]}
    assert list(flatten(data)) == [1, 2, 3, 4, 5, 6, 7, 'abc']
