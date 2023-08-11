import torch
import torch.nn as nn
import warnings

from typing import List, Tuple


class MLP(nn.Module):
    """Represents a multilayer perceptron (MLP)

    Args:
        dims: a sequence of numbers specifying the number of neurons in each layer
        nonlinearity: the choice of nonlinear activation function acting between layers
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncodingCNN(nn.Module):
    """Represents a convolutional neural net (CNN) for specific use in the encoder of the VAE

    Args:
        image_dims: the image dimensions
        channels: the channels of each CNN layer
        kernel_size: the size of kernel used for convolutions
        nonlinearity: the choice of nonlinear activation function acting between layers
    """

    def __init__(
        self,
        image_dims: Tuple[int] = (28, 28),
        channels: List[int] = [1, 32, 64, 16],
        kernel_size: int = 4,
        nonlinearity: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        
        
        ds = 2  # DownSampling factor
        
        
        # automatic downsamping factor determination (ideally 2 for smoothness):

        max_dim = max(image_dims)
        for layer in range(len(channels)-1):
            max_dim //= ds
        if max_dim > 1:
            ds = 3
            
            max_dim = max(image_dims)
            for layer in range(len(channels)-1):
                max_dim //= ds
            if max_dim > 1:
                raise ValueError("Invalid encoding CNN architecture; need more layers for downsampling purposes.")
            
            
        # figure out layer at which downsampling should stop:
        
        # limit_layer = len(channels) - 2
        # min_dim = min(image_dims)
        # for layer in range(len(channels)):
        #     if min_dim // ds < 1:
        #         limit_layer = layer #- 1
        #         break
        #     min_dim //= ds
        limit_layer = len(channels) - 2
        min_dim = min(image_dims)
        for layer in range(len(channels)):
            if ds ** (layer) > min_dim:
                limit_layer = layer - 1
                break
            
        
        # alter the kernel size and compute corresponding paddings
        # note that a second kernel size might be needed for unity stride layers

        if ds == 2 and kernel_size % 2 != 0:
            warnings.warn(f"Kernel size being made even ({kernel_size+1}) for downsampling factor of 2 in encoder")
            kernel_size += 1
        if ds == 3 and kernel_size % 2 == 0:
            warnings.warn(f"Kernel size being made odd ({kernel_size-1}) for downsampling factor of 3 in encoder")
            kernel_size -= 1
            
        pad = (kernel_size - ds) // 2
        
        if kernel_size % 2 == 0:
            second_kernel = kernel_size - 1
        else:
            second_kernel = kernel_size

        same_pad = (second_kernel - 1) // 2
        
        
        # construct network

        net = []
        for i in range(len(channels) - 1):
            if i <= limit_layer:
                net.append(
                    nn.Conv2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size,
                        stride=ds,
                        padding=pad,
                    )
                )
            if i == limit_layer:
                net.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
            if i > limit_layer:
                net.append(
                    nn.Conv2d(channels[i], channels[i + 1], second_kernel, padding=same_pad)
                )
            if i < len(channels) - 2:
                net.append(nonlinearity)

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        # for layer in self.net:
        #     x = layer(x)
        #     print(x.shape)
        # return x.squeeze(-1).squeeze(-1)
        return self.net(x).squeeze(-1).squeeze(-1)


class DecodingCNN(nn.Module):
    """Represents a convolutional neural net (CNN) for specific use in the decoder of the VAE

    Args:
        image_dims: the image dimensions
        channels: the channels of each CNN layer
        kernel_size: the size of kernel used for convolutions
        nonlinearity: the choice of nonlinear activation function acting between layers
    """

    def __init__(
        self,
        image_dims: Tuple[int] = (28, 28),
        channels: List[int] = [16, 64, 32, 1],
        kernel_size: int = 4,
        nonlinearity: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        
        us = 2  # UpSampling factor
        
        
        # automatic upsamping factor determination (ideally 2 for smoothness):
        
        if us**(len(channels)) < min(image_dims):  # allows the manual upsampling to have effective scale factor of at most 'us'
            us = 3
            if us**(len(channels)) < min(image_dims):  # allows the manual upsampling to have effective scale factor of at most 'us'
                raise ValueError("Invalid decoding CNN architecture; need more layers for upsampling purposes.")


        # figure out layer at which upsampling should stop:
        
        limit_layer = len(channels) - 2
        min_dim = min(image_dims)
        for layer in range(len(channels)):
            if us ** (layer) > min_dim:
                limit_layer = layer - 1
                break
            
        
        # alter the kernel size and compute corresponding paddings
        # note that a second kernel size might be needed for unity stride layers
        
        if us == 2 and kernel_size % 2 != 0:
            warnings.warn(f"Kernel size being made even ({kernel_size+1}) for upsampling factor of 2 in decoder")
            kernel_size += 1
        if us == 3 and kernel_size % 2 == 0:
            warnings.warn(f"Kernel size being made odd ({kernel_size-1}) for upsampling factor of 3 in decoder")
            kernel_size -= 1
            
        pad = (kernel_size - us) // 2
        
        if kernel_size % 2 == 0:
            second_kernel = kernel_size - 1
        else:
            second_kernel = kernel_size

        same_pad = (second_kernel - 1) // 2
        
        # construct network

        net = []
        for i in range(len(channels) - 1):
            if i < limit_layer:
                net.append(
                    nn.ConvTranspose2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size,
                        stride=us,
                        padding=pad,
                    )
                )
            if i == limit_layer:
                net.append(nn.Upsample(size=image_dims))
            if i >= limit_layer:
                net.append(
                    nn.ConvTranspose2d(
                        channels[i], channels[i + 1], second_kernel, padding=same_pad
                    )
                )
            if i < len(channels) - 2:
                net.append(nonlinearity)

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        # for layer in self.net:
        #     x = layer(x)
        #     print(x.shape)
        # print('\r\r')
        # return x
        return self.net(x)