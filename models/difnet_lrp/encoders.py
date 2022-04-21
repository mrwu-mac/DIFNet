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
        self.lnorm1 = LayerNorm(d_model)
        self.add = Add()
        self.clone = Clone()
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, m=0):
        res, queries = self.clone(queries, 2)
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        if m == 0:
            att = self.lnorm(self.add([res, self.dropout(att)]))
        else:
            att = self.lnorm1(self.add([res, self.dropout(att)]))
        ff = self.pwff(att, m)
        return ff
    
    def relprop(self, R, m=0, **kwargs):
        R = self.pwff.relprop(R, m, **kwargs)
        if m == 0:
            R = self.lnorm.relprop(R, **kwargs)
        else:
            R = self.lnorm1.relprop(R, **kwargs)
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
        # self.norm = nn.LayerNorm(d_model)
        # self.clone = Clone()
        # self.clone1 = Clone()
        # self.clone2 = Clone()
        # self.clone3 = Clone()
        # self.add = Add()
        # self.add1 = Add()
        # self.add2 = Add()

    def forward(self, input, pixel, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        pixel_attention_mask = (torch.sum(pixel, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        out = input
        out1 = pixel
        for i, l in enumerate(self.layers):
            if i == 0:
                out = l(out, out, out, attention_mask, attention_weights, m=0)
                out1 = l(out1, out1, out1, pixel_attention_mask, attention_weights, m=1)
                out = l(out, out, out, attention_mask, attention_weights, m=0)
                out1 = l(out1, out1, out1, pixel_attention_mask, attention_weights, m=1)
                x = out
                x1 = out1
                out = out + out1
                res = out
            else:
                out = l(out, out, out, attention_mask, attention_weights, m=0)

        out = out + x + x1
        # out = out + res
        # out = self.norm(out)
        # for i, l in enumerate(self.layers):
        #     if i == 0:
        #         for iter in range(2):
        #             qi, ki, vi = self.clone(out, 3)
        #             qp, kp, vp = self.clone1(out1, 3)
        #             out = l(qi, ki, vi, attention_mask, attention_weights, m=0)
        #             out1 = l(qp, kp, vp, pixel_attention_mask, attention_weights, m=1)
        #         x1, out = self.clone2(out, 2)
        #         x2, out1 = self.clone3(out1, 2)
        #         out = self.add([out, out1])
        #     else:
        #         qi, ki, vi = self.clone(out, 3)
        #         out = l(qi, ki, vi, attention_mask, attention_weights, m=0)

        # out = self.add2([x2, self.add1([x1, out])])
        return out, attention_mask
    
    def relprop(self, R, **kwargs):
        # (R2, R) = self.add2.relprop(R, **kwargs)
        # (R1, R) = self.add1.relprop(R, **kwargs)
        # for i, blk in enumerate(reversed(self.layers)):
            

        #     R_q, R_k, R_v = blk.relprop(R, **kwargs)
        #     R = self.clone.relprop((R_q, R_k, R_v), **kwargs)
        return R


class DifnetEncoder_LRP(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(DifnetEncoder_LRP, self).__init__(N, padding_idx, **kwargs)

    def forward(self, input, pixel, attention_weights=None):

        return super(DifnetEncoder_LRP, self).forward(input, pixel, attention_weights=attention_weights)
    
    def relprop(self, R, **kwargs):
        return super(DifnetEncoder_LRP, self).relprop(R, **kwargs)