import torch
import pickle
from data import ImageDetectionsField, TextField, RawField, PixelField
from thop import profile, clever_format
from torchstat import stat
from fvcore.nn.flop_count import flop_count

import argparse
from models.transformer import TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention, Transformer
from models.difnet import TransformerEncoder_ILN_skip, TransformerDecoderLayer_ILN_skip, Transformer_ILN_skip
from models.transformer_lrp import TransformerEncoder_lrp, TransformerDecoderLayer_lrp, ScaledDotProductAttention_lrp, Transformer_lrp
from models.difnet_lrp import TransformerEncoder_p3t_lrp, Transformer_p_lrp




device = torch.device('cpu')
text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True,
                       nopoints=False)
text_field.vocab = pickle.load(open('vocab_transformer/vocab.pkl', 'rb'))

parser = argparse.ArgumentParser(description='transformer Depth')
parser.add_argument('--mode', type=str, default='base',
                        choices=['base', 'base_lrp', 'difnet_lrp', 'difnet'])
args = parser.parse_args()

if args.mode == 'base':
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention)
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
if args.mode == 'base_lrp':
    encoder = TransformerEncoder_lrp(3, 0, attention_module=ScaledDotProductAttention_lrp)
    decoder = TransformerDecoderLayer_lrp(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer_lrp(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
if args.mode == 'difnet_lrp':
    encoder = TransformerEncoder_p3t_lrp(3, 0, attention_module=ScaledDotProductAttention_lrp)
    decoder = TransformerDecoderLayer_lrp(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer_p_lrp(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
if args.mode == 'difnet':
    encoder = TransformerEncoder42172(3, 0, attention_module=ScaledDotProductAttention)
    decoder = TransformerDecoderLayer42172(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer42172(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)


# net = Model()  # 定义好的网络模型
net = model
input_v = torch.randn(1, 7*7, 2048)
input_c = torch.randint(1000, (1, 10)).long()
input_d1 = torch.randn(1, 133, 7, 7)
# input_d2 = torch.randn(1, 49, 64)
# stat(net, (input_v, input_c, input_d1))
flops, params = profile(net, (input_v, input_c, input_d1))
gflop_dict, _ = flop_count(model, (input_v, input_c, input_d1))
gflops = sum(gflop_dict.values())
print(gflops)

# flops, params = clever_format([flops, params], '%.6f')
# print('flops: ', flops, 'params: ', params)
print("Total number of paramerters in networks is {} G  ".format(sum(x.numel() for x in net.parameters())/1024/1024))
