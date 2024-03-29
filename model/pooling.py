import torch
import torch.nn as nn


class AllAP(nn.Module):
    """
        All-ap average pooling layer
    """

    def __init__(self, width):
        super(AllAP, self).__init__()
        self.ap = nn.AvgPool2d((width, 1), stride=1)

    def forward(self, x):
        """
            Args:
                x: torch.Tensor
                    convolution output of size (batch_size, 1, length, width)
            Returns:
                output: torch.Tensor
                    representation vector of size (batch_size, width)
        """
        return self.ap(x).squeeze(1).squeeze(1)


class WidthAP(nn.Module):
    """
        With-ap average pooling layer
    """

    def __init__(self, length, width):
        super(WidthAP, self).__init__()
        self.length = length
        self.width = width

    def forward(self, x, attention_matrix):
        """
            Args:
                x: torch.Tensor of shape (batch_size, 1, max_length + width - 1, height)
                    The output of the convolution layer.
                attention_matrix: 2D attention matrix (col-/row-wise summed vector)
                    of shape (batch_size, max_length + width - 1)
            Returns:
                out: torch.Tensor of shape (batch_size, 1, max_length, height)
                    The output of the w-ap layer.
        """
        pools = []
        attention_matrix = attention_matrix.unsqueeze(1).unsqueeze(3)
        #print(x.shape, attention_matrix.shape)
        for i in range(self.length):
            pools.append(
                (x[:, :, i:i + self.width, :]
                 * attention_matrix[:, :, i:i + self.width, :]).sum(dim=2, keepdim=True))

        return torch.cat(pools, dim=2)
