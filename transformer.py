import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from transformer_modules import TransformerBlock, DenseInterpolation

class Transformer(nn.Module):

    def __init__(self, emb: int, heads: int,
                 depth: int,
                 num_features: int,
                 num_out_channels_emb: int=3,
                 dropout: float=0.0,
                 mask: bool=True):
        """
        Transformer for time series.
        Inputs:
        =======
        emb (int): Embedding dimension
        heads (int): Number of attention heads
        depth (int): Number of transformer blocks
        seq_length (int): length of the sequence
        num_features (int): number of time series features
        mask (bool): if mask diagonal
        """

        super().__init__()

        self.num_features = num_features

        # 1D Conv for actual values of time series
        self.time_series_features_encoding = nn.Conv1d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=1,
                bias=False
            )

        # positional embedding for time series
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=emb)

        # transformer blocks
        tblocks = []
        for _ in range(depth):
            tblocks.append(
                TransformerBlock(
                    emb=emb,
                    heads=heads,
                    seq_length=emb,
                    mask=mask,
                    dropout=dropout
                )
            )
        
        # transformer blocks put together
        self.transformer_blocks = nn.Sequential(*tblocks)

        # conv1d for embeddings
        self.conv1d_for_embeddings = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_out_channels_emb,
            kernel_size=1,
            bias=False
        )
        
        # feed forward through maxpooled embeddings to reduce them to probability
        self.feed_forward = torch.nn.Linear(
            emb,
            1
            )

        self.dropout = nn.Dropout(dropout)

        return None
    
    @staticmethod
    def init_weights(layer):
        """Purpose: initialize weights in each
        LINEAR layer.
        Input: pytorch layer
        """

        if isinstance(layer, torch.nn.Linear):
            np.random.seed(42)
            size = layer.weight.size()
            fan_out = size[0] # number of rows
            fan_in = size[1] # number of columns
            variance = np.sqrt(2.0/(fan_in + fan_out))
            # initialize weights
            layer.weight.data.normal_(0.0, variance)
        
    def forward(self, x):
        """
        Forward pass.
        x (torch.tensor): 3D tensor of size (batch, num_features, 1)
        """

        # 1D Convolution to convert time series data to features (kind of embeddings)
        time_series_features = self.time_series_features_encoding(x)
        b, t, e = time_series_features.size()

        positions = self.pos_embedding(torch.arange(t))[None, :, :].expand(b, t, e)
        
        # sum encoded time serie features and positional encodings and pass on to transfmer block
        x = time_series_features + positions
        x = self.dropout(x)

        x = self.transformer_blocks(x)

        x = self.conv1d_for_embeddings(x)

        # maxpool
        x = x.max(dim=1)[0]

        x = self.feed_forward(x)

        return torch.sigmoid(x)