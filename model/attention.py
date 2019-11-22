import torch
import torch.nn as nn

from model.pooling import AllAP, WidthAP

import psutil


class Attention(nn.Module):
    """
        Attention layer of abcnn2
    """

    def __init__(self, x1_length, x2_length, filter_width):
        super(Attention, self).__init__()
        self.wp_x1 = WidthAP(x1_length, filter_width)
        self.wp_x2 = WidthAP(x2_length, filter_width)
        self.match_score = euclidean

    def forward(self, x1, x2):
        """
            Args:
                x1, x2: torch.Tensors of shape (batch_size, 1, max_length + width - 1, output_size)
                    The outputs from the convolutional layer.
            Returns:
                w1, w2: torch.Tensors of shape (batch_size, 1, max_length, output_size)
                    The outputs from the attention layer. This layer takes
                    the place of the Average Pooling layer seen in the BCNN and ABCNN-1
                    models.
        """
        attention_m = compute_attention_matrix(x1, x2, self.match_score)
        x1_vec = attention_m.sum(dim=2)
        x2_vec = attention_m.sum(dim=1)
        w1 = self.wp_x1(x1, x1_vec)
        w2 = self.wp_x2(x2, x2_vec)
        return w1, w2


def compute_attention_matrix(x1, x2, match_score):
    """ Computes the attention feature map for the batch of inputs x1 and x2.
        The following description is taken directly from the ABCNN paper:
        Let F_{i, r} in R^{d x s} be the representation feature map of
        sentence i (i in {0, 1}). Then we define the attention matrix A in R^{s x s}
        as follows:
            A_{i, j} = match-score(F_{0, r}[:, i], F_{1, r}[:, j])
        Args:
            x1, x2: torch.Tensors of shape (batch_size, 1, max_length, input_size)
                A batch of input tensors.
            match_score: function
                The match-score function to use.
        Returns:
            A: torch.Tensor of shape (batch_size, x1_length, x2_length)
                A batch of attention feature maps.
    """
    batch_size = x1.shape[0]
    x1, x2 = x1.squeeze(1), x2.squeeze(1)
    x1_length, x2_length = x1.shape[1], x2.shape[1]
    A = torch.empty((batch_size, x1_length, x2_length), dtype=torch.float)
    for i in range(x1_length):
        for j in range(x2_length):
            A[:, i, j] = match_score(x1[:, i, :], x2[:, j, :])
    print(psutil.virtual_memory())
    return A


def euclidean(x1, x2):
    """
        Args:
            x1, x2: torch.Tensors of shape (batch_size, 1, input_size)
                The batches of vectors we are computing match-scores for.
        Returns
            scores: torch.Tensor of shape (batch_size, 1)
                The match-scores for the batches of vectors x1 and x2.
    """
    return 1.0 / (1.0 + torch.norm(x1 - x2, p=2, dim=1))
