import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import random
from data import ImageDetectionsField, TextField, RawField, PixelField
from pycocotools.coco import COCO
import evaluation
from models.transformer import TransformerEncoder, TransformerDecoder, ScaledDotProductAttention, Transformer
from models.transformer_lrp import TransformerEncoder_LRP, TransformerDecoder_LRP, ScaledDotProductAttention_LRP, Transformer_LRP
from models.difnet import Difnet, DifnetEncoder, DifnetDecoder
from models.difnet_lrp import Difnet_LRP, DifnetEncoder_LRP, DifnetDecoder_LRP
import torch
from torch.autograd import Variable
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import cv2
import json
import h5py
import matplotlib.pyplot as plt
# from pylab import *
import pylab
# import scipy.ndimage

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

index = 42

def get_topk_logits_selector(logits, k=3):
    """ takes logits[batch, nout, voc_size] and returns a mask with ones at k largest logits """
    topk_logit_indices = torch.topk(logits, k=k).indices.cpu()
    # indices = torch.stack([
    #     (torch.range(start=0, end=logits.shape[0] * logits.shape[1] * k-1) // logits.shape[1] * k),
    #     ((torch.range(start=0, end=logits.shape[0] * logits.shape[1] * k-1) // k) % logits.shape[1]),
    #     torch.reshape(topk_logit_indices, [-1])
    # ], dim=1).unsqueeze(0).long()  # (batch * nout * k, 3)
    # ones = torch.ones((indices.shape[0],))
    return torch.zeros(logits.shape).scatter_(2, topk_logit_indices, 1)
    # return tf.scatter_nd(indices, ones, shape=logits.shape)

def preprocess(filename, detections_path, caption, pixel_path, text_field, max_detections=49):
    print(filename)
    print(caption)
    image_id = int(filename.split('_')[-1].split('.')[0])
    ### grid
    try:
        f = h5py.File(detections_path, 'r')
        grid = f['%d_features' % image_id][()]
    except KeyError:
        warnings.warn('Could not find detections for %d' % image_id)
        gird = np.random.rand(10,2048)

    delta = max_detections - grid.shape[0]
    if delta > 0:
        grid = np.concatenate([grid, np.zeros((delta, grid.shape[1]))], axis=0)
    elif delta < 0:
        grid = grid[:max_detections]
    ### caption
    caption = text_field.preprocess(caption)
    caption = ['<bos>'] + caption
    # print(caption)
    caption = [caption]
    # caption = torch.unsqueeze(caption, dim=0)
    caption = text_field.numericalize(caption, device=device)
    ### pixel    
    d_path = os.path.join(pixel_path, 'val2014/' + str(image_id) + '.npy')
    assert os.path.exists(d_path)
    try:
        pixel = np.load(d_path)
        # precomp_data = mask_decode(precomp_data)
        # precomp_data = np.asfarray(precomp_data).astype(np.float32)
    except KeyError:
        warnings.warn('Could not find Pixel for %d' % image_id)
        pixel = np.random.rand(133, 7, 7).astype(np.float32)
    return torch.tensor(grid.astype(np.float32)), caption, torch.tensor(pixel.astype(np.float32))


def predict_captions(model, img_id, filename, text_field, detections_path, base_caption, our_caption, pixel_path, coco_image, caption_gt):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    ori_cap = base_caption
    image, caption, pixel = preprocess(filename, detections_path, base_caption, pixel_path, text_field)
    image = torch.unsqueeze(image, dim=0)
    # caption = torch.unsqueeze(caption, dim=0)
    pixel = torch.unsqueeze(pixel, dim=0)
    images = image.to(device)
    captions = caption.to(device)
    pixels = pixel.to(device)

    images1 = torch.tensor(np.zeros((1, 49, 2048)), dtype=torch.float32).to(device)
    pixels1 = torch.tensor(np.zeros((1, 49, 133)), dtype=torch.float32).to(device)
    # with torch.no_grad():
    model.eval()
    # out, _, attss= model.beam_search(images, pixels, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
    # out, atts = model(images, captions, pixels)
    out = model(images, captions, pixels)
    out1 = model(images1, captions, pixels)
    out2 = model(images1, captions, pixels1)
    # print(out.shape)
    # vs, inds = torch.topk(out[0][3], 5)
    vs, inds = torch.topk(out[0][3]-out2[0][3], 5)
    print(vs)
    words = []
    for ind in inds:
        words.append(text_field.vocab.itos[int(ind)])
    print(words)
    ######
    result = []
    inp_lrp = []
    out_lrp = []
    model.zero_grad()

    logits = out
    top1 = get_topk_logits_selector(logits, k=1)
    top1_logit = top1.to(logits.device) * logits
    for target_position in range(captions.shape[1]):
        out_mask = torch.nn.functional.one_hot(torch.tensor(target_position), logits.shape[1])[None, :, None]
        top1_prob = top1_logit.sum(dim=-1)[0][target_position].requires_grad_(True)
        # print(top1_prob)
        R_ = top1 * out_mask

        # top1_prob.backward(retain_graph=True)
        # R = model.loss._rdo_to_logits.relprop(R_)
        R = model.relprop_encode_decode(R_.to(logits.device), alpha=1)
        R_out = (abs(R['emb_out'])).sum(dim=-1)
        R_inp = R['emb_inp'].sum(dim=-1)
        # print(R_inp.sum())
        # R_inp = (abs(model.relprop_encode(R['enc_out'], alpha=1))).sum(dim=-1)

        inp_lrp.append(R_inp[0].detach().cpu().numpy())
        out_lrp.append(R_out[0].detach().cpu().numpy())
    result.append({'src': image.cpu().numpy(), 'dst': caption.cpu().numpy(),
                'inp_lrp': np.array(inp_lrp), 'out_lrp': np.array(out_lrp) 
                })
    # print(result)
    show_source2target(result, "./img_{}.jpg".format(img_id))
    #####
    # caps_gen = text_field.decode(out, join_words=False)
    # print(len(atts))
    # att = torch.mean(atts[-1]['attn'], dim=1).view(1, atts[-1]['attn'].shape[2], 7, 7)

    # att = torch.nn.functional.interpolate(att, scale_factor=32, mode='nearest')
    # att = np.squeeze(att.data.cpu().numpy())
    # print(att.shape)
                
    # print('gen:',gen)
    img_path = os.path.join(coco_image, filename)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    cv2.imwrite("./img_{}.jpg".format(img_id), img)
    # for i in range(att.shape[0]):
    #     show_cam_on_image(img, att[i], i)
    # show_caption(img, img_id, ori_cap, our_caption, caption_gt)

    # return scores

def avg_lrp_by_pos(data, seg='inp'):
    count = 0
    res = np.zeros(data[0][seg + '_lrp'].shape[0])
    for i in range(len(data)):
        if not (np.isnan(data[i][seg + '_lrp'])).any():
            res += np.sum(data[i][seg + '_lrp'], axis=-1)
            count += 1
    res /= count
    return res

def show_source2target(data, pictdir):
    # spectral_map = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]

    # color = spectral_map[-1]
    color = 'b'
    fig = plt.figure(figsize=(7, 6), dpi=100)

    res = avg_lrp_by_pos(data, seg='inp')[1:]
    plt.plot(range(2, len(res)+2), res, lw=2., color=color)
    plt.scatter(range(2, len(res)+2), res, lw=3.0, color=color)

    plt.xlabel("target token position", size=25)
    plt.ylabel("source contribution", size=25)
    plt.yticks(size=20)
    plt.xticks([2, 5, 10, 15, 20], size=20)
    plt.title('source ⟶ target(k)', size=25)
    plt.grid()

    pylab.savefig(pictdir, bbox_inches='tight')

def show_cam_on_image(img, mask, i):
    # heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    heatmap = np.expand_dims(mask, -1)
    img = img / 255
    alpha = 0.95
    cam = alpha * heatmap + (1-alpha) * np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("./cam_{}.jpg".format(i+1), np.uint8(255 * cam))

def show_caption(img, img_id, base_caption, our_caption, gts):
    w, h = img.shape[:2]
    l_img = np.zeros((w, 800, 3))
    l_img[:, :, :] = 255
    d_i = 30
    cv2.putText(l_img, 'baseline: ' + base_caption, (30, d_i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 4)
    d_i += 30
    cv2.putText(l_img, 'our: ' + our_caption, (30, d_i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 4)
    d_i += 30
    cv2.putText(l_img, 'gts:', (30, d_i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 4)
    
    for gt in gts:
        cv2.putText(l_img, gt, (70, d_i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 4)
        d_i += 30
        
    new_img = np.hstack((img, l_img))
    cv2.imwrite("./caption_{}.jpg".format(img_id), new_img)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='transformer pixel')
    parser.add_argument('--iid', type=int, required=True)
    parser.add_argument('--exp_name', type=str, default='transformer_grid_original')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='./saved_transformer_models/base_lrp_best.pth')
    parser.add_argument('--caption_path', type=str, default='/data/caption_output')
    parser.add_argument('--coco_dataset', default='/DataSet/COCO/val2014', help='path to coco dataset')
    parser.add_argument('--features_path', type=str, default='/DATA/coco_grid_feats2.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='/DATA/m2_annotations')
    parser.add_argument('--pixel_path', type=str, default='/dataset/high_pixel/channel_pixel_pooled')

    # parser.add_argument('--vocab_path', type=str, default='vocab_m2_transformer.pkl')

    parser.add_argument('--embed_size', type=int, default=512,
                        help='dimension of word embedding vectors')
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--mode', type=str, default='base', choices=['base', 'base_lrp', 'difnet', 'difnet_lrp'])
    args = parser.parse_args()

    print('{} vis'.format(args.mode))

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    # Model and dataloaders
    # encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
    #                                  attention_module_kwargs={'m': 40})
    # decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    # model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'base':
        encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'base_lrp':
        encoder = TransformerEncoder_lrp(3, 0, attention_module=ScaledDotProductAttention_lrp)
        decoder = TransformerDecoderLayer_lrp(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer_lrp(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'rskip':
        encoder = TransformerEncoder2(3, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoderLayer2(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer2(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'aoa':
        encoder = AoATransformerEncoder(3, 0, attention_module=ScaledDotProductAttention)
        decoder = AoATransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = AoATransformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'm2base':
        encoder = MemoryAugmentedEncoderbase(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 40})
        decoder = MeshedDecoderbase(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = m2base(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'm2pixel':
        encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 40})
        decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = m2Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'MIA':
        encoder = TransformerEncoder52(2, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoderLayer52(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer52(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'channel_1a3':
        encoder = TransformerEncoder41(3, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoderLayer41(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer41(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'channel_1a1':
        encoder = TransformerEncoder42(3, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoderLayer42(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer42(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'late_fuse':
        encoder = TransformerEncoder4211(3, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoderLayer4211(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer4211(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'fuse3':
        encoder = TransformerEncoder4212(3, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoderLayer4212(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer4212(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'fuse_bn_3_0_add':
        encoder = TransformerEncoder4213(3, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoderLayer4213(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer4213(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'fuse_bn_2_1_add':
        encoder = TransformerEncoder4214(3, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoderLayer4214(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer4214(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'fuse_bn_2_1_iter':
        encoder = TransformerEncoder42141(3, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoderLayer42141(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer42141(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'fuse_bn_2_1_mlb':
        encoder = TransformerEncoder4215(3, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoderLayer4215(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer4215(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'fuse_bn_2_1_ltc':
        encoder = TransformerEncoder4216(3, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoderLayer4216(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer4216(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'fuse_bn_1_2_add':
        encoder = TransformerEncoder4217(3, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoderLayer4217(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer4217(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'fuse_bn_1_2_1iter':
        encoder = TransformerEncoder42171(3, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoderLayer42171(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer42171(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'fuse_bn_1_2_1iter_3t':
        encoder = TransformerEncoder42172(3, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoderLayer42172(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer42172(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'fuse_bn_1_2_1iter_ltc':
        encoder = TransformerEncoder42173(3, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoderLayer42173(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer42173(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    data = torch.load(args.model_path)
    model.load_state_dict(data['state_dict'])
    # model.load_state_dict(data['state_dict'], strict=False)
    base_captions_file = open(os.path.join(args.caption_path, 'base_lrp.json'), 'r')
    our_captions_file = open(os.path.join(args.caption_path, 'channel_1a1.json'), 'r')
    base_captions = json.load(base_captions_file)
    our_captions = json.load(our_captions_file)
    coco = COCO(os.path.join(args.annotation_folder, 'captions_val2014.json'))
    i = 0
    # for img_id in captions.keys():
    #     i += 1
    #     if not i == index:
    #         continue
    #     img_id = int(img_id)
    #     filename = coco.loadImgs(img_id)[0]['file_name']
    #     caption = captions[str(img_id)]['gen'][0]
    #     caption_gt = captions[str(img_id)]['gts']
    #     scores = predict_captions(model, filename, text_field, args.features_path, caption, args.pixel_path, args.pixel1_path, args.coco_dataset, caption_gt)
    #     print('gt:', caption_gt)
    img_id = args.iid
    filename = coco.loadImgs(img_id)[0]['file_name']
    base_caption = base_captions[str(img_id)]['gen'][0]
    our_caption = our_captions[str(img_id)]['gen'][0]
    caption_gt = base_captions[str(img_id)]['gts']
    scores = predict_captions(model, img_id, filename, text_field, args.features_path, base_caption, our_caption, args.pixel_path, args.coco_dataset, caption_gt)
    print('gt:', caption_gt)
    base_captions_file.close()
    our_captions_file.close()
