import torch
from torch import nn

from typing import Tuple, Optional, List


class VAE(nn.Module):
    """Represents a Variational Autoencoder that acts on image data

    Args:
        image_dims: the image dimensions
        colour_channels: the number of colour channels in the image data
        latent_dim: the dimension of the latent variable/code
        enc_hidden_dims: the architecture of the recognition model, excluding input and output layers
        dec_hidden_dims: the architecture of the decoder, excluding input and output layers
        q_form: the choice of variational posterior
        likelihood: the choice of likelihood function
        noise: the choice of hetero- or homo- scedastic noise if Gaussian likelihood
        noise_log_std: the initial log standard deviation used if the noise is homoscedastic
        train_noise: an option to train the homoscedastic noise variance
        prior_std: the std of the isotropic Guassian prior. User-defined hyperparameter
        nonlinearity: the inter-layer activation function of encoding and decoding networks
    """

    def __init__(
        self,
        image_dims: List[int] = [28, 28],
        colour_channels: int = 1,
        latent_dim: int = 16,
        enc_hidden_dims: List[int] = [20, 20],
        dec_hidden_dims: List[int] = [20, 20],
        q_form: str = "diagonal_gaussian",  # 'full_covariance_gaussian', 'inverse_autoregressive_flow'???,
        likelihood: str = "gaussian",  # 'bernoulli'
        noise: str = "heteroscedastic",  # 'homoscedastic'
        noise_log_std: float = 0.1,
        train_noise: bool = False,
        prior_std: torch.Tensor = torch.tensor(1.0),
        nonlinearity: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        assert len(image_dims) == 2
        self.image_dims = image_dims
        self.colour_channels = colour_channels
        self.num_pixels = image_dims[0] * image_dims[1]
        self.latent_dim = latent_dim
        self.q_form = q_form
        self.likelihood = likelihood
        self.prior_std = prior_std
        self.nonlinearity = nonlinearity

        self.enc_dims = (
            [self.num_pixels * self.colour_channels]
            + enc_hidden_dims
            + [self.latent_dim * 2]
        )

        self.output_multiplier = 1  # TODO: move this into likelihood class as attribute
        if likelihood == "gaussian" and noise == "heteroscedastic":
            self.output_multiplier = 2

        self.dec_dims = (
            [self.latent_dim]
            + dec_hidden_dims
            + [self.num_pixels * self.colour_channels * self.output_multiplier]
        )

        if likelihood == "gaussian":
            self.likelihood = GaussianLikelihood(
                noise=noise, log_std=noise_log_std, train_noise=train_noise
            )
        else:
            raise NotImplementedError("Likelihood chosen not yet implemented")

        self.encoder = MLP(dims=self.enc_dims, nonlinearity=self.nonlinearity)
        self.decoder = MLP(dims=self.dec_dims, nonlinearity=self.nonlinearity)

        self.prior = torch.distributions.MultivariateNormal(
            loc=torch.zeros(latent_dim),
            covariance_matrix=prior_std**2 * torch.eye(latent_dim),
        )

    def forward(self, x: torch.Tensor, num_samples: int = 1):
        assert len(x.shape) == 4
        assert x.shape[1] == self.image_dims[0] and x.shape[2] == self.image_dims[1]
        assert x.shape[-1] == self.colour_channels
        batch_size = x.shape[0]

        q = self.encoder(
            x.view(batch_size, self.num_pixels * self.colour_channels),
            activation=self.q_form,
        )
        z = q.rsample(sample_shape=torch.Size([num_samples])).permute(
            1, 0, 2
        )  # reparameterisation trick

        assert len(z.shape) == 3
        assert z.shape[0] == batch_size
        assert z.shape[1] == num_samples
        assert z.shape[2] == self.latent_dim

        y = self.decoder(
            z.reshape(batch_size * num_samples, self.latent_dim),
        ).view(
            batch_size,
            num_samples,
            self.num_pixels * self.colour_channels * self.output_multiplier,
        )

        p = self.likelihood.activate(y)

        return q, z, p

    def elbo(self, x: torch.Tensor, num_samples: int = 1):  # TODO: add metrics
        metrics = {}
        batch_size = x.shape[0]
        q, _, p = self(x, num_samples=num_samples)
        kl = torch.distributions.kl.kl_divergence(q, self.prior).sum()
        exp_ll = (
            p.log_prob(
                x.view(batch_size, self.num_pixels * self.colour_channels)
                .unsqueeze(1)
                .repeat(1, num_samples, 1)
            )
            .mean(1)
            .sum(-1)
        )
        elbo = exp_ll - kl.unsqueeze(0).repeat(batch_size)
        metrics["elbo"] = elbo.mean()
        metrics["ll"] = exp_ll.mean()
        metrics["kl"] = kl
        return elbo, metrics


class GaussianLikelihood(
    nn.Module
):  # TODO: move this to seperate module and implement other likelihoods
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


class MLP(nn.Module):
    """Represents a multilayer perceptron (MLP)

    Args:
        dims: a sequence of numbers specifying the number of neurons in each layer
        nonlinearity: the choice of nonlinear activation functions acting between layers
    """

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

    def _diag_gauss(
        self, x: torch.Tensor
    ) -> (
        torch.distributions.Distribution
    ):  # TODO: implement this within an approx posterior class
        assert x.shape[-1] % 2 == 0
        loc, logvar = torch.split(x, x.shape[-1] // 2, dim=-1)
        return torch.distributions.MultivariateNormal(
            loc, torch.diag_embed(logvar.exp() + 1e-8)
        )


# TODO: implement CNN to improve performance on MNIST / CelebA (eventually)
