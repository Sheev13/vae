import torch
import torch.nn as nn

class DiagonalGaussian(nn.Module):
    """Converts the output of the VAE encoder to a factorised Gaussian approx. posterior"""

    def __init__(self):
        super().__init__()
        
        # latent_dim * (1 + 1) outputs for mean vector and variance vector
        self.multiplier = 2

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        assert x.shape[-1] % 2 == 0
        loc, logvar = torch.split(x, x.shape[-1] // 2, dim=-1)
        return torch.distributions.MultivariateNormal(
            loc, torch.diag_embed(logvar.exp() + 1e-8)
        )


class FullCovarianceGaussian(nn.Module):
    """Converts the output of the VAE encoder to a full-covariance Gaussian approx. posterior"""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        
        # latent_dim * (1 + latent_dim) outputs for mean vector and covariance matrix
        self.multiplier = 1 + latent_dim
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        assert x.shape[-1] % self.multiplier == 0
        loc = x[..., :self.latent_dim]
        L_prime = x[..., self.latent_dim:].view(x.shape[0], self.latent_dim, self.latent_dim)
        L = torch.triu(L_prime)  # choleksy decomp of covariance
        return torch.distributions.MultivariateNormal(loc, L@L.permute(0, 2, 1) + 1e-8)