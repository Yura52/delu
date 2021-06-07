import gc
from collections import namedtuple
from unittest.mock import patch

import pynvml
import torch
from pynvml import NVMLError_LibraryNotFound
from pytest import mark, raises

import zero


@mark.parametrize('gpu', [False, True])
def test_free_memory(gpu):
    with (patch('gc.collect')) as _, (patch('torch.cuda.empty_cache')) as _, (
        patch('torch.cuda.synchronize')
    ) as _, (patch('torch.cuda.is_available', lambda: gpu)) as _:
        zero.hardware.free_memory()
        gc.collect.call_count == 2
        if gpu:
            torch.cuda.synchronize.assert_called_once()
            torch.cuda.empty_cache.assert_called_once()


def test_get_gpus_info():
    Memory = namedtuple('Memory', ['total', 'free', 'used'])
    Utilization = namedtuple('Utilization', ['gpu'])
    memory = Memory(3, 2, 1)
    utilization = Utilization(10)
    driver = '123.456'
    name = 'ABC123'
    n_devices = 2
    pynvml_fns = [
        f'pynvml.{x}'
        for x in [
            'nvmlInit',
            'nvmlDeviceGetCount',
            'nvmlDeviceGetHandleByIndex',
            'nvmlDeviceGetMemoryInfo',
            'nvmlDeviceGetName',
            'nvmlDeviceGetUtilizationRates',
            'nvmlSystemGetDriverVersion',
        ]
    ]

    with patch(pynvml_fns[0]), patch(pynvml_fns[1], return_value=n_devices), patch(
        pynvml_fns[2], return_value=0
    ), patch(pynvml_fns[3], return_value=memory), patch(
        pynvml_fns[4], return_value=name.encode('utf-8')
    ), patch(
        pynvml_fns[5], return_value=utilization
    ), patch(
        pynvml_fns[6], return_value=driver.encode('utf-8')
    ):
        assert zero.hardware.get_gpus_info() == {
            'driver': driver,
            'devices': [
                {
                    'name': name,
                    'memory_total': memory.total,
                    'memory_free': memory.free,
                    'memory_used': memory.used,
                    'utilization': utilization.gpu,
                }
            ]
            * n_devices,
        }
        for fn in pynvml_fns:
            getattr(pynvml, fn.split('.')[1]).assert_called()


def test_get_gpus_info_without_gpu():
    with patch('pynvml.nvmlInit', side_effect=NVMLError_LibraryNotFound()):
        with raises(RuntimeError):
            zero.hardware.get_gpus_info()
