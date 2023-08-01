import torch
from torch import nn

from likelihoods import GaussianLikelihood
from posteriors import DiagonalGaussian
from networks import MLP, EncodingCNN, DecodingCNN

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
        image_dims: Tuple[int] = (28, 28),
        colour_channels: int = 1,
        latent_dim: int = 16,
        enc_architecture: str = "cnn",
        dec_architecture: str = "cnn",
        enc_mlp_hidden_dims: Optional[List[int]] = [50, 50],
        enc_cnn_chans: Optional[List[int]] = [32, 64],
        dec_mlp_hidden_dims: Optional[List[int]] = [50, 50],
        dec_cnn_chans: Optional[List[int]] = [32, 64],
        kernel_size: Optional[int] = 3,
        posterior_form: str = "diagonal_gaussian",  # 'full_covariance_gaussian',
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
        self.posterior_form = posterior_form
        self.likelihood = likelihood
        self.prior_std = prior_std
        self.nonlinearity = nonlinearity

        if posterior_form == "diagonal_gaussian":
            self.posterior = DiagonalGaussian()
        else:
            raise NotImplementedError("Selected posterior not yet implemented")

        self.output_multiplier = 1  # TODO: move this into likelihood class as attribute
        if likelihood == "gaussian" and noise == "heteroscedastic":
            self.output_multiplier = 2

        self.dec_dims = (
            [self.latent_dim]
            + dec_mlp_hidden_dims
            + [self.num_pixels * self.colour_channels * self.output_multiplier]
        )

        if likelihood == "gaussian":
            self.likelihood = GaussianLikelihood(
                noise=noise, log_std=noise_log_std, train_noise=train_noise
            )
        else:
            raise NotImplementedError("Likelihood chosen not yet implemented")

        self.encoder = Encoder(
            enc_architecture,
            image_dims,
            colour_channels,
            enc_mlp_hidden_dims,
            enc_cnn_chans,
            kernel_size,
            latent_dim,
            self.posterior,
            nonlinearity,
        )
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

        q = self.encoder(x)
        z = q.rsample(sample_shape=torch.Size([num_samples])).permute(
            1, 0, 2
        )  # reparameterisation trick

        assert len(z.shape) == 3
        assert z.shape[0] == batch_size
        assert z.shape[1] == num_samples
        assert z.shape[2] == self.latent_dim

        # finished implementation refactor up to here in the pipeline
        # TODO: refactor from here to end including decoder and likelihood

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


class Encoder(nn.module):
    def __init__(
        self,
        architecture: str,  # 'mlp' or 'cnn'
        image_dims: Tuple[int] = (28, 28),
        colour_channels: int = 1,
        mlp_hidden_dims: Optional[List[int]] = [50, 50],
        cnn_chans: Optional[List[int]] = [32, 64],
        kernel_size: int = 3,
        latent_dim: int = 16,
        posterior: nn.Module = DiagonalGaussian(),
        nonlinearity: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        if architecture not in ["mlp", "cnn"]:
            raise NotImplementedError(
                f"Only 'cnn' and 'mlp' are supported, not {architecture}."
            )
        if architecture == "mlp":
            assert mlp_hidden_dims is not None
        if architecture == "cnn":
            assert cnn_chans is not None

        self.image_dims = image_dims
        self.colour_channels = colour_channels
        self.latent_dim = latent_dim
        self.nonlinearity = nonlinearity
        self.num_pixels = image_dims[0] * image_dims[1]
        self.posterior = posterior
        self.architecture = architecture

        if architecture == "mlp":
            self.enc_dims = (
                [self.num_pixels * self.colour_channels]
                + mlp_hidden_dims
                + [
                    self.latent_dim * 2
                ]  # 2 here assumes diagonal Gaussian latent variable posterior
            )
            self.network = MLP(dims=self.enc_dims, nonlinearity=self.nonlinearity)

        elif architecture == "cnn":
            if cnn_chans[0] != colour_channels:
                cnn_chans = [colour_channels] + cnn_chans
            encoding_cnn_chans = cnn_chans + [latent_dim * 2]
            self.network = EncodingCNN(
                channels=encoding_cnn_chans,
                kernel_size=kernel_size,
                nonlinearity=self.nonlinearity,
            )

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        batch_size = x.shape[0]

        if self.architecture == "mlp":
            x = x.view(batch_size, self.num_pixels * self.colour_channels)
        elif self.architecture == "cnn":
            x = x.permute(
                0, 3, 1, 2
            )  # double check this, am just moving dim 3 to dim 1

        x = self.network(x)

        assert len(x.shape) == 2
        assert x.shape[0] == batch_size
        assert x.shape[1] == self.latent_dim

        q = self.posterior(x)

        return q
