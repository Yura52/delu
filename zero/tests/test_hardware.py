import gc
from unittest.mock import patch

import torch
from pynvml import NVMLError_LibraryNotFound
from pytest import mark, raises

# don't use "from zero.hardware import ...", because it breaks mocking
import zero.hardware as hardware
from zero._util import flatten

from .util import Point, requires_gpu


@mark.parametrize('gpu', [False, True])
def test_free_memory(gpu):
    with (patch('gc.collect')) as _, (patch('torch.cuda.empty_cache')) as _, (
        patch('torch.cuda.synchronize')
    ) as _, (patch('torch.cuda.is_available', lambda: gpu)) as _:
        hardware.free_memory()
        gc.collect.call_count == 2
        if gpu:
            torch.cuda.synchronize.assert_called_once()
            torch.cuda.empty_cache.assert_called_once()


class mocked_nvidia_smi:
    # taken from a real machine with 2xRTX 2080 (idle)
    raw_info = {
        'gpu': [
            {
                'fb_memory_usage': {
                    'total': 11019.4375,
                    'used': 0.0625,
                    'free': 11019.375,
                    'unit': 'MiB',
                },
                'utilization': {'gpu_util': 0, 'unit': '%'},
            },
            {
                'fb_memory_usage': {
                    'total': 11016.9375,
                    'used': 0.0625,
                    'free': 11016.875,
                    'unit': 'MiB',
                },
                'utilization': {'gpu_util': 0, 'unit': '%'},
            },
        ]
    }

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
            'free%': 100,
        },
        {
            'util%': 0,
            'total': 11016,
            'used': 0,
            'free': 11016,
            'used%': 0,
            'free%': 100,
        },
    ]
    assert hardware.get_gpu_info() == correct

    correct = [
        {
            'util%': 0.0,
            'total': 11019.4375,
            'used': 0.0625,
            'free': 11019.375,
            'used%': 0.0005671795860723381,
            'free%': 99.99943282041393,
        },
        {
            'util%': 0.0,
            'total': 11016.9375,
            'used': 0.0625,
            'free': 11016.875,
            'used%': 0.0005673082923453092,
            'free%': 99.99943269170765,
        },
    ]
    actual = hardware.get_gpu_info(True)
    assert len(actual) == len(correct)
    for a, c in zip(actual, correct):
        for key in 'util%', 'total', 'used', 'free':
            assert a[key] == c[key]
        for key in 'used%', 'free%':
            assert round(a[key], 8) == round(c[key], 8)

    assert hardware.get_gpu_info(True) == correct


def test_get_gpu_info_without_gpu():
    with patch(
        'zero.hardware.nvidia_smi.getInstance', side_effect=NVMLError_LibraryNotFound()
    ):
        with raises(RuntimeError):
            hardware.get_gpu_info()


@requires_gpu
def test_to_device():
    t = lambda x: torch.tensor(0, device=x)  # noqa
    cpu = torch.device('cpu')
    cuda = torch.device('cuda', 0)

    for x in cpu, cuda:
        data = t(x)
        assert hardware.to_device(data, x) is data
    assert hardware.to_device(t(cpu), cuda).device == cuda

    for Container in tuple, Point, list:
        constructor = Container._make if Container is Point else Container
        for device in [cpu, cuda]:
            data = constructor([t(cpu), t(cpu)])
            out = hardware.to_device(data, device)
            assert isinstance(out, Container)
            assert all(x.device == device for x in out)
            if device == cpu:
                for x, y in zip(out, data):
                    assert x is y

    data = [t(cpu), t(cpu)]
    for x, y in zip(hardware.to_device(data, cpu), data):
        assert x is y
    assert all(x.device == cuda for x in hardware.to_device(data, cuda))

    data = {
        'a': [t(cpu), (t(cpu), t(cpu))],
        'b': {'c': {'d': [[[t(cpu)]]]}},
        'c': Point(t(cpu), {'d': t(cpu)}),
    }
    for x, y in zip(flatten(hardware.to_device(data, cpu)), flatten(data)):
        assert x is y
    for x, y in zip(flatten(hardware.to_device(data, cuda)), flatten(data)):
        assert x.device == cuda
        assert type(x) == type(y)
