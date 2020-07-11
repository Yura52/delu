"""Tools related to devices, memory, etc."""

__all__ = ['free_memory', 'get_gpu_info', 'to_device']

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
    """Free GPU-memory occupied by `torch` and run the garbage collector.

    Warning:
        There is a small chunk of GPU-memory (occupied by drivers) that is impossible to
        free. It is a `torch` "limitation", so the function inherits this property.
    """
    gc.collect()
    if torch.cuda.is_available():
        # torch has wrong .pyi
        torch.cuda.synchronize()  # type: ignore
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gpu_info(precise: bool = False) -> List[Dict[str, Any]]:
    """Get statistics about GPU devices.

    Includes information about memory (total, free and used) and utilization. Some
    figures are represented in two ways: with raw units and with percentage.

    Args:
        precise: if False, all data is rounded (to Mb for memory, to % for percentages)
    Returns:
        Information about GPU devices.
    Raises:
        RuntimeError: if necessary cuda-related libraries are not found. Usually, it
            means that the function is run on a machine without GPU.

    Examples:
        .. code-block::

            print(get_gpu_info())

        Output example (formatted for convenience):

        .. code-block:: none

            [
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

    Note:
        The function directly collects information using the :code:`pynvml` library,
        hence, settings like :code:`CUDA_VISIBLE_DEVICES` don't affect the result.
    """
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
    """Move tensor(s) to device.

    Move data consisting of tensors to the given device using `torch.Tensor.to`.

    Args:
        data (`Recursive[torch.Tensor] <zero.types.Recursive>`)
        device (`Device <zero.types.Device>`)
        non_blocking: is forwarded to `torch.Tensor.to`
    Returns:
        `Recursive[torch.Tensor] <zero.types.Recursive>`:
            The same data, but moved to the given device.

    Examples:
        .. testcode::

            to_device(torch.tensor(0), 'cpu')
            to_device({'a': torch.tensor(0), 'b': [(torch.tensor(0),)]}, 'cpu')
    """
    # int is missing in .pyi
    return traverse(lambda x: x.to(device, non_blocking=non_blocking), data)  # type: ignore
