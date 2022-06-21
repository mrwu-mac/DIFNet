import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import random
from data import ImageDetectionsField, TextField, RawField, PixelField
from pycocotools.coco import COCO
import evaluation
from models.transformer import TransformerEncoder, TransformerDecoder, ScaledDotProductAttention, Transformer
from models.transformer_lrp import TransformerEncoder_LRP, TransformerDecoder_LRP, ScaledDotProductAttention_LRP, Transformer_LRP
from models.difnet import Difnet, DifnetEncoder, DifnetDecoder
from models.difnet_lrp import Difnet_LRP, DifnetEncoder_LRP
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


def predict_captions(model, filename, text_field, detections_path, base_caption, pixel_path, coco_image, caption_gt):
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

    # images1 = torch.tensor(np.zeros((1, 49, 2048)), dtype=torch.float32).to(device)
    # pixels1 = torch.tensor(np.zeros((1, 49, 133)), dtype=torch.float32).to(device)
    # with torch.no_grad():
    model.eval()
    # out, _, attss= model.beam_search(images, pixels, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
    # out, atts = model(images, captions, pixels)
    out = model(images, captions, pixels)
    # out1 = model(images1, captions, pixels)
    # out2 = model(images1, captions, pixels1)
    # print(out.shape)
    # vs, inds = torch.topk(out[0][3], 5)
    # vs, inds = torch.topk(out[0][3]-out2[0][3], 5)
    # print(vs)
    # words = []
    # for ind in inds:
    #     words.append(text_field.vocab.itos[int(ind)])
    # print(words)
    ######
    # result = []
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
        inp_lrp.append(R_inp[0].detach().cpu().numpy())
        out_lrp.append(R_out[0].detach().cpu().numpy())
    
    return {'src': image.cpu().numpy(), 'dst': caption.cpu().numpy(),
                'inp_lrp': np.array(inp_lrp), 'out_lrp': np.array(out_lrp)}

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
    cv2.imwrite("/cam_{}.jpg".format(i+1), np.uint8(255 * cam))

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
    cv2.imwrite("/caption_{}.jpg".format(img_id), new_img)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='transformer pixel')
    # parser.add_argument('--iid', type=int, required=True)
    parser.add_argument('--exp_name', type=str, default='transformer_grid_original')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='./output/saved_transformer_models')
    parser.add_argument('--out_path', type=str, default='./output/output_lrp')
    parser.add_argument('--coco_dataset', default='/DataSet/COCO/val2014', help='path to coco dataset')
    parser.add_argument('--features_path', type=str, default='./dataset/coco_grid_feats2.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='./dataset/m2_annotations')
    parser.add_argument('--pixel_path', type=str, default='./dataset/segmentations')

    # parser.add_argument('--vocab_path', type=str, default='vocab.pkl')

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
        decoder = TransformerDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'base_lrp':
        encoder = TransformerEncoder_LRP(3, 0, attention_module=ScaledDotProductAttention_LRP)
        decoder = TransformerDecoder_LRP(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer_LRP(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'difnet':
        encoder = DifnetEncoder(1, 2, 3, 0, attention_module=ScaledDotProductAttention)
        decoder = DifnetDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Difnet(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'difnet_lrp':
        encoder = DifnetEncoder_LRP(3, 0, attention_module=ScaledDotProductAttention_LRP)
        decoder = TransformerDecoder_LRP(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Difnet_LRP(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)


    data = torch.load(os.path.join(args.model_path, args.exp_name + '.pth'))
    model.load_state_dict(data['state_dict'])
    # model.load_state_dict(data['state_dict'], strict=False)
    base_captions_file = open(os.path.join(args.out_path, args.exp_name + '.json'), 'r')
    base_captions = json.load(base_captions_file)
    # our_captions = json.load(our_captions_file)
    coco = COCO(os.path.join(args.annotation_folder, 'captions_val2014.json'))
  
    result = []
    with tqdm(desc='LRP', unit='it', total=len(base_captions.keys())) as pbar:
        for it, img_id in enumerate(base_captions.keys()):
            img_id = int(img_id)
            filename = coco.loadImgs(img_id)[0]['file_name']
            base_caption = base_captions[str(img_id)]['gen'][0]
            caption_gt = base_captions[str(img_id)]['gts']
            result.append(predict_captions(model, filename, text_field, args.features_path, base_caption, args.pixel_path, args.coco_dataset, caption_gt))
            pbar.update()

    base_captions_file.close()
    # our_captions_file.close()
    pickle.dump(result, open(os.path.join(args.out_path, '{}_result.pkl'.format(args.exp_name)), 'wb'))
    # show_source2target(result, "/data/relevance_visual/total_lrp.jpg")
