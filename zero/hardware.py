"""Tools related to devices, memory, etc."""

import gc
from typing import Any, Dict

import pynvml
import torch
from pynvml import NVMLError_LibraryNotFound


def free_memory() -> None:
    """Free GPU-memory occupied by `torch` and run the garbage collector.

    Warning:
        There is a small chunk of GPU-memory (occupied by drivers) that is impossible to
        free. It is a `torch` "limitation", so the function inherits this property.

    Inspired by: https://github.com/xtinkt/editable/blob/1c80efb80c196cdb925fc994fc9ed576a246c7a1/lib/utils/basic.py#L124
    """
    gc.collect()
    if torch.cuda.is_available():
        # torch has wrong .pyi
        torch.cuda.synchronize()  # type: ignore
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gpus_info() -> Dict[str, Any]:
    """Get information about GPU devices: driver version, memory, utilization etc.

    The example below shows what kind of information is returned as the result. All
    figures about memory are given in bytes.

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

            {
                'driver': '440.33.01',
                'devices': [
                    {
                        'name': 'GeForce RTX 2080 Ti',
                        'memory_total': 11554717696,
                        'memory_free': 11554652160,
                        'memory_used': 65536,
                        'utilization': 0,
                    },
                    {
                        'name': 'GeForce RTX 2080 Ti',
                        'memory_total': 11552096256,
                        'memory_free': 11552030720,
                        'memory_used': 65536,
                        'utilization': 0,
                    },
                ],
            }

    Warning:
        The 'devices' value contains information about *all* gpus regardless of the
        value of :code:`CUDA_VISIBLE_DEVICES`.
    """
    try:
        pynvml.nvmlInit()
    except NVMLError_LibraryNotFound as err:
        raise RuntimeError(
            'Failed to get information about GPU memory. '
            'Make sure that you actually have GPU and all relevant software installed.'
        ) from err
    n_devices = pynvml.nvmlDeviceGetCount()
    devices = []
    for device_id in range(n_devices):
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        devices.append(
            {
                'name': str(pynvml.nvmlDeviceGetName(handle), 'utf-8'),
                'memory_total': memory_info.total,
                'memory_free': memory_info.free,
                'memory_used': memory_info.used,
                'utilization': pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            }
        )
    return {
        'driver': str(pynvml.nvmlSystemGetDriverVersion(), 'utf-8'),
        'devices': devices,
    }
