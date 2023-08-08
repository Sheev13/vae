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
        noise: str = "pixelwise-heteroscedastic",
        std: float = 0.1,
        train_noise: bool = False,
        activation: nn.Module = nn.Sigmoid(),
    ):
        super().__init__()
                
        assert noise in (
            'pixelwise-heteroscedastic',
            'imagewise-heteroscedastic',
            'homoscedastic'
        ), 'typo in noise description: options are "pixelwise-heteroscedastic", "imagewise-heteroscedastic, or "homoscedastic"'
        self.noise = noise
        self.log_std = nn.Parameter(torch.tensor(std).log(), requires_grad=train_noise)
        if noise == "homoscedastic":
            self.multiplier = 1
        elif noise.endswith("heteroscedastic"):
            self.multiplier = 2
        
        self.activation = activation

    @property
    def std(self):
        return self.log_std.exp()

    def forward(self, y: torch.Tensor) -> torch.distributions.Distribution:

        if self.noise.endswith("heteroscedastic"):
            assert y.shape[-1] % 2 == 0
            loc, log_std = torch.split(y, y.shape[-1] // 2, dim=-1)
            if self.noise.startswith("imagewise"):
                log_std = log_std[:,:,0,0,:].unsqueeze(2).unsqueeze(2)  # pick only one value for the whole image
            loc = self.activation(loc)
            return torch.distributions.Normal(
                loc, log_std.exp()
            )

        elif self.noise == "homoscedastic":
            y = self.activation(y)
            scale = torch.ones_like(y) * self.std
            return torch.distributions.Normal(y, scale)