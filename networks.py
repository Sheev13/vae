import torch
import torch.nn as nn
import warnings

from typing import List, Tuple


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

        
class EncodingCNN(nn.Module):
    """Represents a convolutional neural net (CNN) for specific use in the encoder of the VAE

    Args:
    
    """

    def __init__(
        self,
        channels: List[int] = [1, 32, 64, 16],
        kernel_size: int = 3,
        nonlinearity: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        
        if len(channels) <= 3:
            pooling_scale = 2
        else:
            pooling_scale = 3
        
        net = []
        for i in range(len(channels) - 1):
            net.append(nn.Conv2d(channels[i], channels[i+1], kernel_size, padding="same"))
            if i < len(channels) - 1:
                net.append(nn.AvgPool2d(pooling_scale))
            else:
                net.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
            net.append(nonlinearity)
        
        self.net = nn.Sequential(*net)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze()
            

class DecodingCNN(nn.Module):
    """Represents a convolutional neural net (CNN) for specific use in the decoder of the VAE

    Args:
    
    """

    def __init__(
        self,
        image_dims: Tuple[int] = (28, 28),
        channels: List[int] = [16, 64, 32, 1],
        kernel_size: int = 3,
        nonlinearity: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        
        if len(channels) <= 3:
            upsampling_scale = 2
        else:
            upsampling_scale = 3
        
        net = []
        for i in range(len(channels) - 1):
            net.append(nn.Conv2d(channels[i], channels[i+1], kernel_size, padding="same"))
            if i < len(channels) - 1:
                net.append(nn.UpsamplingBilinear2d(scale_factor=upsampling_scale))
            else:
                if 2**(len(channels) - 1) <= min(image_dims):
                    net.append(nn.UpsamplingBilinear2d(size=image_dims))
                else: 
                    warnings.warn("Warning: ")
                    net.append(nn.AdaptiveAvgPool2d(output_size=image_dims))
            net.append(nonlinearity)
            
        self.net = nn.Sequential(*net)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
        
        
        
            