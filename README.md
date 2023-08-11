# Variational Autoencoder Implementation for Image Data

This repository contains an implementation of a variational autoencoder (VAE) (Kingma and Welling, "Auto-Encoding Variational Bayes", 2013) in PyTorch that supports three-dimensional data, such as images with any number of colour channels. 

The implementation supports encoders and decoders that use either multilayer perceptrons (MLPs) or convolutional neural networks (CNNs). Mean-field or full covariance Gaussian variational posteriors for the latent variable are supported, as well as Gaussian or Bernoulli likelihoods for the data. The Gaussian likelihood supports homoscedastic, imagewise heteroscedastic, or pixelwise heteroscedastic noise models. The $\beta$-NLL approach (Seitzer et al., "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks", 2022) is implemented for use in the pixelwise heteroscedastic noise regime to improve training.

In completing my MEng project within the subfield of amortised variational inference for Bayesian neural networks (BNNs), VAEs lay firmly in my periphery throughout due to the many similarities, and so this project was completed in order to familiarise myself with VAEs in a thorough way.

