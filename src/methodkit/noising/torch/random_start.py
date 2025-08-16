from typing import Literal, no_type_check

import torch
import torch.linalg

Norm = Literal["l2", "linf"]


@no_type_check
def random_start(x: torch.Tensor, epsilon: float, norm: Norm = "l2") -> torch.Tensor:
    if norm == "linf":
        return x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    elif norm == "l2":
        batch, n = x.flatten(start_dim=1).shape
        noise = torch.randn(batch, n + 1, device=x.device)
        vector_norm = torch.linalg.vector_norm(noise, dim=-1, keepdim=True)
        normalized_noise = noise / vector_norm
        muller_noise = normalized_noise[:, :n].reshape(x.shape)
        return x + muller_noise * epsilon
    else:
        raise ValueError(f"Invalid norm: {norm}")
