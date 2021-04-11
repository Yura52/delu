import gc
from unittest.mock import patch

import torch
from pynvml import NVMLError_LibraryNotFound
from pytest import mark, raises

# don't use "from zero.hardware import ...", because it breaks mocking
import zero.hardware as hardware


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


def test_get_gpus_info_without_gpu():
    with patch('pynvml.nvmlInit', side_effect=NVMLError_LibraryNotFound()):
        with raises(RuntimeError):
            hardware.get_gpus_info()
