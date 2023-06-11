import torch
from torch import nn

from typing import Tuple, Optional, List


class VAE(nn.Module):
    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        inference_hidden_dims: List[int] = [20, 20],
        q: str = "diagonal_gaussian",  # 'full_covariance_gaussian', 'inverse_autoregressive_flow',
        decoder: str = "bernoulli",
    ):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.inference_hidden_dims = inference_hidden_dims
        self.q = q
        self.decoder = decoder


class MLP(nn.Module):
    def __init__(
        self,
        dims: List[int] = [1, 20, 20, 1],
        nonlinearity: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.dims = dims
        self.nonlinearity = nonlinearity

        net = []
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                net.append(nonlinearity)

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
