from __future__ import annotations

import torch


def tensor_to_numpy(t: torch.Tensor):
    """Safely convert a torch Tensor to a NumPy array.

    NumPy does not support some torch dtypes (notably bfloat16), so we upcast
    unsupported float dtypes to float32 before calling `.numpy()`.
    """
    t = t.detach()
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.to(dtype=torch.float32)
    return t.cpu().contiguous().numpy()
