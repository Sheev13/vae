# Variational Autoencoder Implementation for Image Data

This repository contains an implementation of a variational autoencoder (VAE) (Kingma and Welling, "Auto-Encoding Variational Bayes", 2013) in PyTorch that supports image data of any 2D dimension with any number of colour channels. 

The implementation supports encoders and decoders that use either multilayer perceptrons (MLPs) or convolutional neural networks (CNNs). Mean-field Gaussian (diagonal/factorised Gaussian) and full covariance Gaussian variational approximate posteriors are supported for the latent variable. The implementation supports Gaussian likelihoods with homoscedastic, imagewise heteroscedastic, and pixelwise heteroscedastic noise, but there are also plans to support a Bernoulli likelihood for binary image data (e.g. MNIST image pixels can be viewed as binary). The $\beta$-NLL approach (Seitzer et al., "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks", 2022) is implemented for use in the pixelwise heteroscedastic noise regime.

In completing my MEng project within the subfield of amortised variational inference for Bayesian neural networks (BNNs), VAEs lay firmly in my periphery throughout due to the many similarities, and so this project was completed in order to familiarise myself with VAEs in a thorough way.
