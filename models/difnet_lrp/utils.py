import torch
from torch import nn
from torch.nn import functional as F
from ..layers_lrp import *


def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out


def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out


class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''

    def __init__(self, d_model=512, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.dropout = Dropout(p=dropout)
        self.dropout_2 = Dropout(p=dropout)
        self.layer_norm = LayerNorm(d_model)
        self.layer_norm1 = LayerNorm(d_model)
        self.add1 = Add()
        self.relu = ReLU()
        self.clone1 = Clone()


    def forward(self, input, m=0):
        if self.identity_map_reordering:
            if m == 0:
                out = self.layer_norm(input)
            else:
                out = self.layer_norm1(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            res, out = self.clone1(input, 2)
            out = self.fc2(self.dropout_2(self.relu(self.fc1(out))))
            out = self.dropout(out)
            if m == 0:
                out = self.layer_norm(self.add1([res, out]))
            else:
                out = self.layer_norm1(self.add1([res, out]))
        return out
    
    def relprop(self, R, m=0, **kwargs):
        if m == 0:
            R = self.layer_norm.relprop(R, **kwargs)
        else:
            R = self.layer_norm1.relprop(R, **kwargs)
        (R1, R2) = self.add1.relprop(R, **kwargs)
        R2 = self.dropout.relprop(R2, **kwargs)
        R2 = self.fc2.relprop(R2, **kwargs)
        R2 = self.dropout_2.relprop(R2, **kwargs)
        R2 = self.relu.relprop(R2, **kwargs)
        R2 = self.fc1.relprop(R2, **kwargs)
        R = self.clone1.relprop((R1, R2), **kwargs)
        return R
