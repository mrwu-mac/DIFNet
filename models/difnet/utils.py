import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
# from scipy.fftpack import fft,ifft
import numpy as np


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
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)

    def forward(self, input, m=0):
        if self.identity_map_reordering:
            if m == 0:
                out = self.layer_norm(input)
            else:
                out = self.layer_norm1(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            if m == 0:
                out = self.layer_norm(input + out)
            else:
                out = self.layer_norm1(input + out)
        return out


def save_freq(x1, x2, x):
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)
    ax1.set_ylabel('A')
    ax1.set_title('Grid&Pixel&Fusion Feature')
    ax3.set_xlabel('freq')
    for i in range(49):
        x1_ft = np.fft.fft(x1.cpu().numpy()[0][i])/512
        x2_ft = np.fft.fft(x2.cpu().numpy()[0][i])/512
        x_ft = np.fft.fft(x.cpu().numpy()[0][i])/512
        # x1_ft = torch.rfft(x1, 3, normalized=False, onesided=False)
        # x2_ft = torch.rfft(x2, 3, normalized=False, onesided=False)
        # x_ft = torch.rfft(x, 3, normalized=False, onesided=False)
        x1f = np.arange(256)
        y1f = np.abs(x1_ft)[range(256)]
        ax1.plot(x1f, y1f)

        x2f = np.arange(256)
        y2f = np.abs(x2_ft)[range(256)]
        ax2.plot(x2f, y2f)

        xf = np.arange(256)
        yf = np.abs(x_ft)[range(256)]
        ax3.plot(xf, yf)
        # break

    plt.savefig('/home/wumingrui/cognet/test.jpg')