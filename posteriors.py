import torch
import torch.nn as nn


class DiagonalGaussian(nn.Module):
    """Converts the output of the VAE encoder to a factorised Gaussian approx. posterior"""

    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        assert x.shape[-1] % 2 == 0
        loc, logvar = torch.split(x, x.shape[-1] // 2, dim=-1)
        return torch.distributions.MultivariateNormal(
            loc, torch.diag_embed(logvar.exp() + 1e-8)
        )


class FullCovGaussian(nn.Module):
    """Converts the output of the VAE encoder to a full-covariance Gaussian approx. posterior"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        pass
    