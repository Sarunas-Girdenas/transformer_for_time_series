import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from transformer_modules import TransformerBlock, DenseInterpolation

class Transformer(nn.Module):

    def __init__(self, emb: int, heads: int,
                 depth: int,
                 num_features: int,
                 interpolation_factor: int=3,
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
        num_classes (int): number of classes
        num_features (int): number of time series features
        max_pool (bool): if true, use global max pooling in the last layer,
            else use global average pooling
        mask (bool): if mask diagonal
        """

        super().__init__()

        self.num_features, self.interpolation_factor = num_features, interpolation_factor

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

        # dense interpolation layer
        self.dense_interpolation = DenseInterpolation(
            seq_lenght=emb,
            factor=interpolation_factor
            )
        
        # feed forward layer to reduce interpolated layers to probability
        self.feed_forward = torch.nn.Linear(
            int(interpolation_factor*num_features),
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

        x = self.dense_interpolation(x.transpose(dim0=1, dim1=-1))

        x = x.contiguous().view(-1, int(self.num_features*self.interpolation_factor))

        x = self.feed_forward(x)

        return torch.sigmoid(x)