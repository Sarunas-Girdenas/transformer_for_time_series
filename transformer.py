import torch
from torch import nn
import torch.nn.functional as F

from transformer_modules import TransformerBlock

class Transformer(nn.Module):

    def __init__(self, emb: int, heads: int,
                 depth: int, num_classes: int,
                 num_features: int, max_pool: bool=True,
                 dropout: float=0.0,
                 mask=True):
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

        self.max_pool = max_pool

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

        # reduce back to number of classes (classifier)
        self.reduce_to_probabilities = nn.Linear(emb, num_classes)

        self.dropout = nn.Dropout(dropout)

        return None
    
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

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)

        x = self.reduce_to_probabilities(x)

        return F.log_softmax(x, dim=1)