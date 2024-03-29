import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolution(nn.Module):
    """ Implements the convolution layer as described in this paper:
        http://www.aclweb.org/anthology/Q16-1019
    """
    def __init__(self, input_size, output_size, width, in_channels):
        """ Initializes the convolution layer.
            Args:
                input_size: int
                    The dimension of the input features.
                output_size: int
                    The dimension of the output features.
                width: int
                    The width of the convolution filters.
                in_channels: int
                    The number of input channels to the convolution layer.
        """
        super().__init__()
        self.conv = \
            nn.Conv2d(
                in_channels,
                output_size,
                kernel_size=(width, input_size),
                stride=1,
                padding=(width - 1, 0)
            )

    def forward(self, x):
        """ Computes the forward pass over the convolution layer.
            Args:
                x: torch.Tensor of shape (batch_size, 1, seq_len, height)
                    The input to the convolution layer.
            Returns:
                out: torch.Tensor of shape (batch_size, 1, width, out_channels)
                    The output of the convolution layer.
        """
        out = F.tanh(self.conv(x))  # (batch_size, out_channels, width, 1)
        out = out.permute(0, 3, 2, 1)  # (batch_size, 1, width, out_channels)
        return out
