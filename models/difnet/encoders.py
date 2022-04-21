from torch.nn import functional as F
from .utils import PositionWiseFeedForward, save_freq
import torch
from torch import nn
from .attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.lnorm1 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, m=0):

        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        if m == 0:
            att = self.lnorm(queries + self.dropout(att))
        else:
            att = self.lnorm1(queries + self.dropout(att))
        ff = self.pwff(att, m)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, Lf, T, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.N = N
        self.Lf = Lf
        self.T = T
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])

        self.padding_idx = padding_idx

    def forward(self, input, pixel, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        pixel_attention_mask = (torch.sum(pixel, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        out = input
        out1 = pixel

        for i, l in enumerate(self.layers):
            if i < self.Lf:
                for t in range(self.T):
                    out = l(out, out, out, attention_mask, attention_weights, m=0)
                    out1 = l(out1, out1, out1, pixel_attention_mask, attention_weights, m=1)

            elif i == self.Lf:
                x1 = out
                x2 = out1
                out = out + out1
                out = l(out, out, out, attention_mask, attention_weights, m=0)
            else:
                out = l(out, out, out, attention_mask, attention_weights, m=0)
                
        out = out + x1 + x2
        return out, attention_mask


class DifnetEncoder(MultiLevelEncoder):
    def __init__(self, Lf, T, N, padding_idx, d_in=2048, **kwargs):
        super(DifnetEncoder, self).__init__(Lf, T, N, padding_idx, **kwargs)

    def forward(self, input, pixel, attention_weights=None):

        return super(DifnetEncoder, self).forward(input, pixel, attention_weights=attention_weights)
