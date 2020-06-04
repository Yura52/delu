from collections import UserDict

from zero._util import flatten, to_list, traverse


def test_to_list():
    x = []
    assert to_list(x) is x

    x = [0]
    assert to_list(x) is x

    assert to_list(0) == [0]

    assert to_list((0,)) == [(0,)]


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
    assert traverse(fn, [3, 4]) == [6, 8]
    assert traverse(fn, {'a': 3, 'b': 4}) == {'a': 6, 'b': 8}

    assert traverse(fn, ([],)) == ([],)
    assert traverse(fn, [()]) == [()]
    assert traverse(fn, {'a': []}) == {'a': []}

    assert traverse(fn, ([1, 2], (3, 4))) == ([2, 4], (6, 8))
    assert traverse(fn, [(), [1, 2], {'a': 3, 'b': 4}]) == [(), [2, 4], {'a': 6, 'b': 8}]

    input_ = {
        'a': [{'b': 1}, {'c': (2, 3)}],
        'b': {'c': {'d': {'e': 4}}},
        'c': [[5, 6, (7, 8), [[[9]]]]],
    }
    output = {
        'a': [{'b': 2}, {'c': (4, 6)}],
        'b': {'c': {'d': {'e': 8}}},
        'c': [[10, 12, (14, 16), [[[18]]]]],
    }
    assert traverse(fn, input_) == output

    class MyDict(UserDict):
        pass
    assert traverse(fn, MyDict({'a': 3, 'b': 4})) == MyDict({'a': 6, 'b': 8})


def test_flatten():
    data = {'a': 1, 'b': [2, 3], 'c': [4, (5, 6), [[[7], [], ['abc']]]]}
    assert list(flatten(data)) == [1, 2, 3, 4, 5, 6, 7, 'abc']
