"""Small distributed-eval helpers.

NIAH runs single-process by default (no torchrun needed). If you launch the eval
under ``torchrun``, rows are sharded across ranks and the per-rank tallies are
summed with ``all_reduce_sum``. With a single process these are all no-ops.
"""

from __future__ import annotations

from typing import List

import torch


def dist_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return (
            True,
            torch.distributed.get_rank(),
            torch.distributed.get_world_size(),
            torch.distributed.get_backend(),
        )
    return False, 0, 1, None


def _dist_device(device, backend):
    return device if backend == "nccl" else "cpu"


def all_reduce_sum(values: List[float], device, backend) -> List[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=_dist_device(device, backend))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.cpu().tolist()
