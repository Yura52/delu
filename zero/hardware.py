import gc
import math
from typing import Any, Dict, List

import torch
from pynvml import NVMLError_LibraryNotFound
from pynvml.smi import nvidia_smi

from ._util import traverse
from .types import Device, Recursive

_GPU_INFO_QUERY = 'memory.total, memory.used, memory.free, utilization.gpu'


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        # torch has wrong .pyi
        torch.cuda.synchronize()  # type: ignore
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gpu_info(precise: bool = False) -> List[Dict[str, Any]]:
    try:
        smi = nvidia_smi.getInstance()
    except NVMLError_LibraryNotFound as err:
        raise RuntimeError(
            'Failed to get information about GPU memory. '
            'Make sure that you actually have GPU and all relevant software installed.'
        ) from err
    raw_info = smi.DeviceQuery(_GPU_INFO_QUERY)
    process_float = (lambda x: float(x)) if precise else math.floor  # noqa

    def unpack_raw_gpu_info(raw_gpu_info):
        gpu_info = {'util%': raw_gpu_info['utilization']['gpu_util']}
        gpu_info.update(
            (x, process_float(raw_gpu_info['fb_memory_usage'][x]))
            for x in ['total', 'used', 'free']
        )
        for x in 'used', 'free':
            gpu_info[x + '%'] = process_float(gpu_info[x] / gpu_info['total'] * 100)
        return gpu_info

    return list(map(unpack_raw_gpu_info, raw_info['gpu']))


def to_device(
    data: Recursive[torch.Tensor], device: Device, non_blocking: bool = False
) -> Recursive[torch.Tensor]:
    # int is missing in .pyi
    return traverse(lambda x: x.to(device, non_blocking=non_blocking), data)  # type: ignore
