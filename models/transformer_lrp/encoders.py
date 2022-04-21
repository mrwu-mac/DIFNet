from torch.nn import functional as F
from .utils import PositionWiseFeedForward
import torch
from torch import nn
from .attention import MultiHeadAttention
from ..layers_lrp import *


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = Dropout(dropout)
        self.lnorm = LayerNorm(d_model)
        self.add = Add()
        self.clone = Clone()
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        res, queries = self.clone(queries, 2)
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        att = self.lnorm(self.add([res, self.dropout(att)]))
        ff = self.pwff(att)
        return ff
    
    def relprop(self, R, **kwargs):
        R = self.pwff.relprop(R, **kwargs)
        R = self.lnorm.relprop(R, **kwargs)
        (R1, R_a) = self.add.relprop(R, **kwargs)
        R_a = self.dropout.relprop(R_a, **kwargs)
        R_q, R_k, R_v = self.mhatt.relprop(R_a, **kwargs)
        R_q = self.clone.relprop((R1, R_q), **kwargs)
        return R_q, R_k, R_v


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx
        self.clone = Clone()

    def forward(self, input, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        out = input
        for layer in self.layers:
            q, k, v = self.clone(out, 3)
            out = layer(q, k, v, attention_mask, attention_weights)
            # outs.append(out.unsqueeze(1))

        # outs = torch.cat(outs, 1)
        return out, attention_mask
    
    def relprop(self, R, **kwargs):
        for blk in reversed(self.layers):
            R_q, R_k, R_v = blk.relprop(R, **kwargs)
            R = self.clone.relprop((R_q, R_k, R_v), **kwargs)
        return R


# class TransformerEncoder(MultiLevelEncoder):
#     def __init__(self, N, padding_idx, d_in=2048, **kwargs):
#         super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
#         self.fc = nn.Linear(d_in, self.d_model)
#         self.dropout = nn.Dropout(p=self.dropout)
#         self.layer_norm = nn.LayerNorm(self.d_model)

#     def forward(self, input, attention_weights=None):

#         out = F.relu(self.fc(input))
#         out = self.dropout(out)
#         out = self.layer_norm(out)
#         return super(TransformerEncoder, self).forward(out, attention_weights=attention_weights)
class TransformerEncoder_LRP(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder_LRP, self).__init__(N, padding_idx, **kwargs)
        # self.fc = nn.Linear(d_in, self.d_model)
        # self.dropout = nn.Dropout(p=self.dropout)
        # self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None):

        # out = F.relu(self.fc(input))
        # out = self.dropout(out)
        # out = self.layer_norm(out)
        return super(TransformerEncoder_LRP, self).forward(input, attention_weights=attention_weights)
    
    def relprop(self, R, **kwargs):
        return super(TransformerEncoder_LRP, self).relprop(R, **kwargs)