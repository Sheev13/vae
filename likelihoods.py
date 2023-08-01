import torch
import torch.nn as nn

from typing import List



class GaussianLikelihood(
    nn.Module
):
    """Converts the output of the VAE decoder to a Gaussian likelihood function

    Args:
        noise: an option to model the noise as homoscedastic or heteroscedastic
        log_std: the initial log standard deviation used if the noise is homoscedastic
        train_noise: an option to train the homoscedastic noise variance
    """

    def __init__(
        self,
        noise: str = "heteroscedastic",
        log_std: float = 0.1,
        train_noise: bool = False,
    ):
        super().__init__()
        self.noise = noise
        self.log_std = nn.Parameter(torch.tensor(log_std), requires_grad=train_noise)

    @property
    def std(self):
        return self.log_std.exp()

    def forward(self, y: torch.Tensor) -> torch.distributions.Distribution:
        assert len(y.shape) == 3

        if self.noise == "heteroscedastic":
            assert y.shape[-1] % 2 == 0
            loc, logvar = torch.split(y, y.shape[-1] // 2, dim=-1)
            return torch.distributions.MultivariateNormal(
                loc, torch.diag_embed(logvar.exp() + 1e-8)
            )

        elif self.noise == "homoscedastic":
            cov = (
                (torch.eye(y.shape[-1]) * self.std**2 + 1e-8)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(y.shape[0], y.shape[1], 1)
            )
            return torch.distributions.MultivariateNormal(y, cov)

        else:
            raise NotImplementedError(
                'typo in noise description: options are "heteroscedastic" or "homoscedastic"'
            )

    def activate(self, y: torch.Tensor) -> torch.distributions.Distribution:
        return self(y)