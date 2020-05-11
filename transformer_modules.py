import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):

    def __init__(self, emb, heads=8, mask=True):
        """
        This is SelfAttentionWide from
        https://github.com/pbloem/former/blob/6a3295c94a151291de6838b98826da13e5f57bf3/former/modules.py
        
        Inputs:
        ========
        emb (int): embedding size
        head (int): number of attention heads
        mask (bool): if mask some parts of attention
        """

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        # compute queries, keys and values for all
        # heads (as a single concatenated vector)

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

        return None
    
    def forward(self, x):
        """
        Forward pass of the Self Attention Mechanism
        x (torch.tensor): b, t, e -  size of the input, where
        t - number of vectors
        e - dimension of the vector (embedding)
        b - batch size
        """

        b, t, e = x.size()
        h = self.heads
        
        assert e  == self.emb, f"Input embedding dim ({e}) does not match embedding dim ({self.emb})"

        keys = self.tokeys(x).view(b, t, h, e) # reshape x to be of size b, t, h, e
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # compute scaled dot-product self-attention
        # folding heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)
        
        queries = queries / (e ** (1/4))
        keys = keys / (e * (1/4))

        # get dot product of queries and keys
        dot_product = torch.bmm(queries, keys.transpose(1, 2))

        assert dot_product.size() == (b * h, t, t) # contains Attention weights

        # mask out upper half of the dot_product matrix excluding the diagonal
        if self.mask:
            SelfAttention.mask_(dot_product, maskval=float('-inf'), mask_diagonal=False)
        
        # apply softmax row-wise
        dot_product = F.softmax(dot_product, dim=2)

        # apply Self Attention to the values to get the output of each attention head
        out = torch.bmm(dot_product, values).view(b, h, t, e)

        # transpose back (swap t with h), unify Attention heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)
    
    @staticmethod
    def mask_(matrices, maskval=0.0, mask_diagonal=True):
        """
        Masks out all values in the given batch of matrices where i <= j holds,
        i < j if mask_diagonal is false
        Taken from: https://github.com/pbloem/former/blob/master/former/util/util.py
        In place operation
        :param tns:
        :return:
        """

        _, h, w = matrices.size()

        indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
        matrices[:, indices[0], indices[1]] = maskval


class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask,
                seq_length, feed_forward_hidden_mult=4,
                dropout=0.0):
        """
        Transformer block
        """

        super().__init__()

        # instance of attention
        self.attention = SelfAttention(emb=emb, heads=heads, mask=mask)

        # mask instance
        self.mask = mask

        # layer normalisation
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        # feed forward layer
        self.feed_forward_layers = nn.Sequential(
            nn.Linear(emb, feed_forward_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden_mult * emb, emb)
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass of a transformer block
        """

        # take X and self-attend to it
        attended = self.attention(x)

        # Residual connection 1 & normalisation
        x = self.norm1(attended + x)

        # dropout
        x = self.dropout(x)

        # feedforward layer
        feed_forward = self.feed_forward_layers(x)

        # Residual connection 2 & normalisation
        x  = self.norm2(feed_forward + x)

        # another dropout
        x = self.dropout(x)

        return x