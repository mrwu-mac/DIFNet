import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import random
from data import ImageDetectionsField1, TextField, RawField, PixelField
from data import COCO, DataLoader
import evaluation
from models.transformer import TransformerEncoder, TransformerDecoder, ScaledDotProductAttention, Transformer
from models.transformer_lrp import TransformerEncoder_LRP, TransformerDecoder_LRP, ScaledDotProductAttention_LRP, Transformer_LRP
from models.difnet import Difnet, DifnetEncoder, DifnetDecoder
from models.difnet_lrp import Difnet_LRP, DifnetEncoder_LRP
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import cv2
import json

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def predict_captions(model, dataloader, text_field, out_file):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    outs = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, ((img_id, images, pixels), caps_gt) in enumerate(iter(dataloader)):
            # print(img_id.data.numpy()[0])
            images = images.to(device)
            # images = torch.zeros((50, 49, 2048)).to(device)
            pixels = pixels.to(device)
            # pixels = torch.zeros((50, 49, 133)).to(device)
            # depths1 = depths1.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, pixels, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
                
                # print('gen:',gen['%d_%d' % (it, i)])
                # print('gts:',gts['%d_%d' % (it, i)])
                # out_file.write('gen:{}'.format(gen['%d_%d' % (it, i)]))
                # out_file.write('gts:{}'.format(gts['%d_%d' % (it, i)]))
                # out_file.write('img_id: {}'.format(str(img_id.data.numpy()[0])))
                # out_file.write('\n')
                # out_file.write('gen:{}'.format(gen['%d_%d' % (it, i)]))
                # out_file.write('\n')
                # out_file.write('gts:{}'.format(gts['%d_%d' % (it, i)]))
                # out_file.write('\n')
                outs[str(img_id.data.numpy()[0])] = {'gen':gen['%d_%d' % (it, i)], 'gts':gts['%d_%d' % (it, i)]}
                # out_file.write(gen['%d_%d' % (it, i)][0])
                # out_file.write('\n')
            pbar.update()
    #         if it > 5:
    #             break
    # print(outs)
    json.dump(outs, out_file)
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, score_list = evaluation.compute_scores(gts, gen)
    cc = {}
    out1_file = open(os.path.join(args.out_path, args.exp_name + '_cider.json'), 'w')
    cc['cider'] = list(score_list['CIDEr'])
    json.dump(cc, out1_file)
    out1_file.close()
    
    return scores


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='transformer Depth')
    parser.add_argument('--exp_name', type=str, default='transformer_grid_original')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='./output/saved_transformer_models')
    parser.add_argument('--out_path', type=str, default='./output/output_lrp')
    parser.add_argument('--features_path', type=str, default='./dataset/coco_grid_feats2.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='./dataset/m2_annotations')
    parser.add_argument('--pixel_path', type=str, default='./dataset/segmentations')

    parser.add_argument('--embed_size', type=int, default=512,
                        help='dimension of word embedding vectors')
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--mode', type=str, default='base', choices=['base', 'base_lrp', 'difnet', 'difnet_lrp'])

    args = parser.parse_args()

    print('{} Evaluation'.format(args.mode))

    # Pipeline for image regions
    image_field = ImageDetectionsField1(detections_path=args.features_path, max_detections=49, load_in_tmp=False)
    # Pipeline for depth
    pixel_field = PixelField(pixel_path=args.pixel_path, load_in_tmp=False)
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
    #     print("Building vocabulary")
    #     text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
    #     pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    # else:
    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    # Create the dataset
    dataset = COCO(image_field, text_field, pixel_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    _, _, test_dataset = dataset.splits

    # Model and dataloaders
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

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'pixel': pixel_field})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    out_file = open(os.path.join(args.out_path, args.exp_name + '.json'), 'w')
    # out_file = open(os.path.join(args.out_path, args.exp_name + '.txt'), 'w')
    # out_file = None
    scores = predict_captions(model, dict_dataloader_test, text_field, out_file)
    out_file.close()
    print(scores)
