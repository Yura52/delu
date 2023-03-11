"""An addition to `torch.cuda`."""
import gc

import torch


def free_memory() -> None:
    """Free GPU memory: `torch.cuda.synchronize` + `gc.collect` + `torch.cuda.empty_cache`.

    Warning:
        There is a small chunk of GPU-memory (occupied by drivers) that is impossible to
        free. It is a `torch` "limitation", so the function inherits this property.

    Example:
        .. testcode::

            delu.cuda.free_memory()
    """  # noqa: E501
    # Step 1: finish the ongoing computations.
    if torch.cuda.is_available():
        # torch has wrong .pyi
        torch.cuda.synchronize()  # type: ignore
    # Step 2: collect unused objects.
    gc.collect()
    # Step 3: free GPU-cache.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
