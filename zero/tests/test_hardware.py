import gc
from unittest.mock import patch

import torch
from pytest import raises

import zero.hardware as hardware
from zero._util import flatten


def test_free_memory():
    with (
        patch('gc.collect')
    ) as _, (
        patch('torch.cuda.empty_cache')
    ) as _:
        hardware.free_memory()
        gc.collect.call_count == 2
        torch.cuda.empty_cache.assert_not_called()

    with (
        patch('gc.collect')
    ) as _, (
        patch('torch.cuda.empty_cache')
    ) as _, (
        patch('torch.cuda.synchronize')
    ) as _, (
        patch('torch.cuda.is_available', lambda: True)
    ) as _:
        hardware.free_memory()
        gc.collect.call_count == 2
        torch.cuda.empty_cache.assert_called_once()
        torch.cuda.synchronize.assert_called_once()


class mocked_nvidia_smi:
    # taken from a real machine with 2xRTX 2080 (idle)
    raw_info = {'gpu': [
        {
            'fb_memory_usage': {
                'total': 11019.4375, 'used': 0.0625, 'free': 11019.375, 'unit': 'MiB'
            },
            'utilization': {'gpu_util': 0, 'unit': '%'}
        },
        {
            'fb_memory_usage': {
                'total': 11016.9375, 'used': 0.0625, 'free': 11016.875, 'unit': 'MiB'
            },
            'utilization': {'gpu_util': 0, 'unit': '%'}
        }
    ]}

    @staticmethod
    def getInstance():
        return mocked_nvidia_smi

    @staticmethod
    def DeviceQuery(query):
        assert query is hardware._GPU_INFO_QUERY
        return mocked_nvidia_smi.raw_info


@patch('zero.hardware.nvidia_smi', mocked_nvidia_smi)
def test_get_gpu_info():
    correct = [
        {
            'util%': 0,
            'total': 11019,
            'used': 0,
            'free': 11019,
            'used%': 0,
            'free%': 100
        },
        {
            'util%': 0,
            'total': 11016,
            'used': 0,
            'free': 11016,
            'used%': 0,
            'free%': 100
        }
    ]
    assert hardware.get_gpu_info() == correct

    correct = [
        {
            'util%': 0.0,
            'total': 11019.4375,
            'used': 0.0625,
            'free': 11019.375,
            'used%': 0.0005671795860723381,
            'free%': 99.99943282041393
        },
        {
            'util%': 0.0,
            'total': 11016.9375,
            'used': 0.0625,
            'free': 11016.875,
            'used%': 0.0005673082923453092,
            'free%': 99.99943269170765
        }
    ]
    actual = hardware.get_gpu_info(True)
    assert len(actual) == len(correct)
    for a, c in zip(actual, correct):
        for key in 'util%', 'total', 'used', 'free':
            assert a[key] == c[key]
        for key in 'used%', 'free%':
            assert round(a[key], 8) == round(c[key], 8)

    assert hardware.get_gpu_info(True) == correct


def test_get_gpu_info_cpu_fail():
    with raises(RuntimeError):
        hardware.get_gpu_info()


class Tensor:
    def __init__(self, device):
        self.device = device

    def to(self, device, non_blocking):
        del non_blocking
        return self if device == self.device else Tensor(device)


def test_to_device():
    data = Tensor('cpu')
    assert hardware.to_device(data, 'cpu') is data

    for Container in tuple, list:
        data = Container([Tensor('cpu'), Tensor('cpu')])
        out = hardware.to_device(data, 'cpu')
        assert isinstance(out, Container)
        for x, y in zip(hardware.to_device(data, 'cpu'), data):
            assert x is y

    data = [Tensor('cpu'), Tensor('cpu')]
    for x, y in zip(hardware.to_device(data, 'cpu'), data):
        assert x is y

    data = {
        'a': [Tensor('cpu'), (Tensor('cpu'), Tensor('cpu'))],
        'b': {'c': {'d': [[[Tensor('cpu')]]]}}
    }
    for x, y in zip(flatten(hardware.to_device(data, 'cpu')), flatten(data)):
        assert x is y
    for x in flatten(hardware.to_device(data, 'cuda')):
        assert x.device == 'cuda'
