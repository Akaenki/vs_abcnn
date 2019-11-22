import torch
import torch.nn as nn
from torch.nn import functional as F

from model.attention import Attention
from model.pooling import AllAP, WidthAP
from model.conv import Convolution


class Model(nn.Module):
    """
        Main model
    """
    def __init__(self, drug_length, protein_length, n_drug, n_protein, embd_dim=20, filter_width=5, layer_size=1):
        """
            Args:
                 drug_length: drug compound input size, i.e. number of atoms
                 protein_length: protein input size, i.e number of amino acid
                 n_drug: length of the drug embedding dictionary
                 n_protein: length of the protein embedding dictionary (if we only use amino acid as a "word" this is only 20)
                 embd_dim: embedding vector length
                 filter_width: convolutional filter width
                 layer_size: number of abcnn2 layers
        """
        super(Model, self).__init__()
        self.layer_size = layer_size

        # Embedding Layers
        self.embd_drug = nn.Embedding(n_drug, embd_dim)
        self.embd_protein = nn.Embedding(n_protein, embd_dim)

        # ABCNN layers
        self.conv_cpd = nn.ModuleList([Convolution(embd_dim, embd_dim, filter_width, 1) for _ in range(layer_size)])
        self.conv_prt = nn.ModuleList([Convolution(embd_dim, embd_dim, filter_width, 1) for _ in range(layer_size)])
        self.attn = nn.ModuleList([Attention(drug_length, protein_length, filter_width) for _ in range(layer_size)])

        # all-ap average pooling layers
        self.ap_cpd = AllAP(drug_length)
        self.ap_prt = AllAP(protein_length)

        # final classifier
        self.fc = nn.Linear(embd_dim * 2, 2)

    def forward(self, inputs):
        """
            Args:
                 inputs: TBD
            Returns:
                outputs: torch.FloatTensor of shape (batch_size, 2)
                    The scores for each class for each pair of sequences.
        """
        drug, protein = inputs

        # Extract embedding vectors of drug and protein
        # TODO: Not sure
        x1 = self.embd_drug(drug).unsqueeze(1)
        x2 = self.embd_protein(protein).unsqueeze(1)

        # ABCNN layers
        for i in range(self.layer_size):
            x1 = self.conv_cpd[i](x1)
            x2 = self.conv_prt[i](x2)
            x1, x2 = self.attn[i](x1, x2)

        # Apply all-ap to obtain two final vector representation of drug and protein
        x1 = self.ap_cpd(x1)
        x2 = self.ap_prt(x2)
        # Concatenate vector representations and apply classifier
        out = torch.cat((x1, x2), dim=1)
        logits = self.fc(out)
        return logits
