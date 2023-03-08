import torch
from pytest import raises
from torch.utils.data import DataLoader, TensorDataset

import delu.data as dd


def test_enumerate():
    dataset = TensorDataset(torch.arange(10), torch.arange(10))
    x = dd.Enumerate(dataset)
    assert x.dataset is dataset
    assert len(x) == 10
    assert x[3] == (3, (torch.tensor(3), torch.tensor(3)))


def test_fndataset():
    dataset = dd.FnDataset(lambda x: x * 2, 3)
    assert len(dataset) == 3
    assert dataset[0] == 0
    assert dataset[1] == 2
    assert dataset[2] == 4

    dataset = dd.FnDataset(lambda x: x * 2, 3, lambda x: x * 3)
    assert len(dataset) == 3
    assert dataset[0] == 0
    assert dataset[1] == 6
    assert dataset[2] == 12

    dataset = dd.FnDataset(lambda x: x * 2, [1, 10, 100])
    assert len(dataset) == 3
    assert dataset[0] == 2
    assert dataset[1] == 20
    assert dataset[2] == 200

    dataset = dd.FnDataset(lambda x: x * 2, (x for x in range(0, 10, 4)))
    assert len(dataset) == 3
    assert dataset[0] == 0
    assert dataset[1] == 8
    assert dataset[2] == 16


def test_make_index_dataloader():
    with raises(ValueError):
        dd.make_index_dataloader(0)

    for x in range(1, 10):
        assert len(dd.make_index_dataloader(x)) == x

    data = torch.arange(10)
    for batch_size in range(1, len(data) + 1):
        torch.manual_seed(batch_size)
        correct = list(DataLoader(data, batch_size, shuffle=True, drop_last=True))
        torch.manual_seed(batch_size)
        actual = list(
            dd.make_index_dataloader(
                len(data), batch_size, shuffle=True, drop_last=True
            )
        )
        for x, y in zip(actual, correct):
            assert torch.equal(x, y)
