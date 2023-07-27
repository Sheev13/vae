import torch
from torch import nn

from typing import Tuple, Optional, List


class VAE(nn.Module):
    """Represents a Variational Autoencoder for use on images

    Args:
        image_dims: the image dimensions
        latent_dim: the dimension of the latent variable/code
        enc_hidden_dims: the architecture of the recognition model, excluding input and output layers
        dec_hidden_dims: the architecture of the decoder, excluding input and output layers
        q_form: the choice of variational posterior
        likelihood: the choice of likelihood function
        noise: the choice of hetero- or homo- scedastic noise if Gaussian likelihood
        noise_log_std: Gaussian likelihood noise initialisation if homoscedastic Gaussian likelihood used
        train_noise: whether to train the above parameter
        prior_std: the std of the isotropic Guassian prior. User-defined hyperparameter
        nonlinearity: the inter-layer activation function of encoding and decoding networks
    """

    def __init__(
        self,
        image_dims: List[int],
        latent_dim: int,
        enc_hidden_dims: List[int] = [20, 20],
        dec_hidden_dims: List[int] = [20, 20],
        q_form: str = "diagonal_gaussian",  # 'full_covariance_gaussian', 'inverse_autoregressive_flow'???,
        likelihood: str = "gaussian",  # 'bernoulli'
        noise: str = "heteroscedastic",  # 'homoscedastic'
        noise_log_std: torch.Tensor = 0.1,
        train_noise: bool = False,
        prior_std: torch.Tensor = torch.tensor(1.0),
        nonlinearity: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        assert len(image_dims) == 2
        self.image_dims = image_dims
        self.num_pixels = image_dims[0] * image_dims[1]
        self.latent_dim = latent_dim
        self.enc_hidden_dims = (
            [self.num_pixels] + enc_hidden_dims + [self.latent_dim * 2]
        )
        self.dec_hidden_dims = [self.latent_dim] + dec_hidden_dims + [self.num_pixels]
        self.q_form = q_form
        self.likelihood = likelihood
        self.prior_std = prior_std
        self.nonlinearity = nonlinearity

        if likelihood == "gaussian":
            self.likelihood = GaussianLikelihood(
                noise=noise, log_std=noise_log_std, train_noise=train_noise
            )
        else:
            raise NotImplementedError("Likelihood chosen not yet implemented")

        self.encoder = MLP(dims=self.enc_hidden_dims, nonlinearity=self.nonlinearity)
        self.decoder = MLP(dims=self.dec_hidden_dims, nonlinearity=self.nonlinearity)

        self.prior = torch.distributions.MultivariateNormal(
            loc=torch.zeros(latent_dim),
            covariance_matrix=prior_std**2 * torch.eye(latent_dim),
        )

    def _likelihood_activation(y: torch.Tensor) -> torch.distributions.Distribution:
        assert len(y.shape) == 3

    def forward(self, x: torch.Tensor, num_samples: int = 1):
        assert len(x.shape) == 3
        assert x.shape[-2:] == self.image_dims
        batch_size = x.shape[0]

        q = self.encoder(x.reshape(batch_size, self.num_pixels), activation=self.q_form)
        z = q.rsample(
            sample_shape=torch.Size([num_samples])
        )  # reparameterisation trick

        assert z.shape == [batch_size, self.latent_dim, num_samples]

        y = self.decoder(
            z.view(batch_size * num_samples, self.latent_dim),
        ).view(batch_size, num_samples, self.num_pixels)

        p = self.likelihood.activate(y)

        return q, z, p

    def elbo(self, x: torch.Tensor, num_samples: int = 1):
        pass


class GaussianLikelihood(nn.Module):
    def __init__(
        self,
        noise: str = "heteroscedastic",
        log_std: torch.Tensor = 0.1,
        train_noise: bool = False,
    ):
        self.noise = noise
        self.log_std = nn.Parameter(log_std, requires_grad=train_noise)

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
            )  # TODO: this won't work since diag_embed doesn't support multiple batch dims?

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

    def forward(self, x: torch.Tensor, activation: str = "") -> torch.Tensor:
        x = self.net(x)
        if activation == "diagonal_gaussian":
            return self._diag_gauss(x)
        if activation == "":
            return x
        else:
            raise NotImplementedError(
                "Variational posteriors other than diagonal Gaussian not yet implemented"
            )

    def _diag_gauss(x: torch.Tensor) -> torch.distributions.Distribution:
        assert x.shape[-1] % 2 == 0
        loc, logvar = torch.split(x, x.shape[-1] // 2, dim=-1)
        return torch.distributions.MultivariateNormal(
            loc, torch.diag(logvar.exp() + 1e-8)
        )
