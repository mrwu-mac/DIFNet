import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model1 import CaptioningModel1

# fuse_bn_1+2
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
 
    def forward(self, x):
        return x.view(x.size(0), 133, -1).permute(0, 2, 1)


class Difnet(CaptioningModel1):
    def __init__(self, bos_idx, encoder, decoder):
        super(Difnet, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        # image embed
        self.embed_image = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.LayerNorm(512))
        # self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.embed_pixel = nn.Sequential(
                # nn.AdaptiveAvgPool2d((7, 7)),
                Flatten(),
                nn.Linear(133, self.decoder.d_model),
                # nn.Sigmoid(),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.LayerNorm(self.decoder.d_model))
        # self.embed_pixel = ModePool2D()

        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq, pixel, *args):
        images = self.embed_image(images)
        pixel = self.embed_pixel(pixel)
        enc_output, mask_enc = self.encoder(images, pixel)
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, pixel, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                visual = self.embed_image(visual)
                pixel = self.embed_pixel(pixel)
                self.enc_output, self.mask_enc = self.encoder(visual, pixel)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc)


class TransformerEnsemble(CaptioningModel1):
    def __init__(self, model: Difnet, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)


