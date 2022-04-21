import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model import CaptioningModel
from ..layers_lrp import *


class Transformer_LRP(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder):
        super(Transformer_LRP, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        # image embed
        self.embed_image = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.LayerNorm(512))

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

    def forward(self, images, seq, depths, *args):
        images = self.embed_image(images)
        enc_output, mask_enc = self.encoder(images)
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, depth, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                visual = self.embed_image(visual)
                self.enc_output, self.mask_enc = self.encoder(visual)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc)
    
    def relprop_decode(self, R, **kwargs):
        """ propagates relevances from rdo to output embeddings and encoder state """
        # if self.normalize_out:
            # R = self.dec_out_norm.relprop(R)

        # R_enc = 0.0
        # R_enc_scale = 0.0
        R_crop = 0.0
        # for layer in range(self.decodernum_layers_dec)[::-1]:
            # R = self.dec_ffn[layer].relprop(R)

            # relevance_dict = self.dec_enc_attn[layer].relprop(R, main_key='query_inp')
            # R = relevance_dict['query_inp']
            # R_enc += relevance_dict['kv_inp']
            # R_enc_scale += tf.reduce_sum(abs(relevance_dict['kv_inp']))

            # R = self.dec_attn[layer].relprop(R)
        R, R_enc = self.decoder.relprop(R, **kwargs)
        # print(R_enc.sum())
        R_crop = R
        # shift left: compensate for right shift
        # R_crop = tf.pad(R, [[0, 0], [0, 1], [0, 0]])[:, 1:, :]

        return {'emb_out': R_crop, 'enc_out': R_enc,
                'emb_out_before_crop': R}

    def relprop_encode(self, R, **kwargs):
        """ propagates relevances from enc_out to emb_inp """
        # if self.normalize_out:
            # R = self.enc_out_norm.relprop(R)
        # for layer in range(self.num_layers_enc)[::-1]:
            # R = self.enc_ffn[layer].relprop(R)
            # R = self.enc_attn[layer].relprop(R)
        R = self.encoder.relprop(R, **kwargs)
        return R

    def relprop_encode_decode(self, R, **kwargs):
        """ propagates relevances from rdo to input and optput embeddings """
        relevances = self.relprop_decode(R, **kwargs)
        # relevances['emb_inp'] = self.relprop_encode(relevances['enc_out'])
        relevances['emb_inp'] = relevances['enc_out']
        return relevances


class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer_LRP, weight_files):
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
