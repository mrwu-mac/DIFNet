import os
import pickle
import numpy as np
import cv2
import json
import h5py
import matplotlib.pyplot as plt
# from pylab import *
import pylab
import argparse

from scipy.stats import entropy

def all_inp_entropy(data, pos=None):
    res = []
    for i in range(len(data)):
        if not (np.isnan(data[i]['inp_lrp'])).any():
            res_ = np.sum(abs(data[i]['inp_lrp']), axis=-1)
            try:
                if pos is None:
                    res += [entropy(abs(data[i]['inp_lrp'][p])/res_[p]) for p in range(data[i]['inp_lrp'].shape[0])]
                else:
                    res.append(entropy(abs(data[i]['inp_lrp'][pos]) / res_[pos]))
            except Exception:
                pass
    # print(res)
    return res

def avg_lrp_by_pos(data, seg='inp'):
    count = 0
    res = np.zeros(20)
    for i in range(len(data)):
        if not (np.isnan(data[i][seg + '_lrp'])).any():
            d = np.sum(data[i][seg + '_lrp'], axis=-1)
            d = np.pad(d,(0,20-d.shape[0]),'constant',constant_values=(0,0))
            res += d
            count += 1
    res /= count
    return res


def show_source2target(all_data, labels, save_img_contribution):
    # spectral_map = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]

    # color = spectral_map[-1]
    fig = plt.figure(figsize=(7, 6), dpi=100)
    for i, (data, label) in enumerate(zip(all_data, labels)):
        res = avg_lrp_by_pos(data, seg='inp')[1:]
        plt.plot(range(2, 9 + 2), res[:9], label=label)

    # res = avg_lrp_by_pos(data, seg='inp')[1:]
    # res1 = avg_lrp_by_pos(data1, seg='inp')[1:]
    # res2 = avg_lrp_by_pos(data2, seg='inp')[1:]
    # plt.plot(range(2, len(res) + 2), res, 'bv-', label='Baseline w/o Enc: extra skip connection')
    # plt.plot(range(2, len(res) + 2), res1, 'g^--', label='Baseline')
    # plt.plot(range(2, len(res) + 2), res2, 'ro-', label='Ours')
    # plt.plot(range(2, 9+2), res[:9], 'bv-', label='Baseline w/o Enc: extra skip connection')
    # plt.plot(range(2, 9+2), res1[:9], 'g^--', label='Baseline')
    # plt.plot(range(2, 9+2), res2[:9], 'ro-', label='Ours')
    # plt.plot(range(2, len(res)-3), res[:14], lw=2., color='b')
    # plt.plot(range(2, len(res) - 3), res1[:14], lw=2., color='g')
    # plt.plot(range(2, len(res) - 3), res2[:14], lw=2., color='r')
    # plt.scatter(range(2, len(res)-3), res[:14], lw=3.0, color=color)
    # plt.scatter(range(2, len(res) - 3), res1[:14], lw=3.0, color=color)
    # plt.scatter(range(2, len(res) - 3), res2[:14], lw=3.0, color=color)

    plt.xlabel("target token position", size=20)
    plt.ylabel("visual contribution", size=20)
    plt.yticks(size=20)
    plt.xticks([2, 5, 10, 11], size=17)
    plt.title('visual ⟶ target(k)', size=20)
    plt.grid()
    plt.legend(fontsize=13)
    pylab.savefig(save_img_contribution, bbox_inches='tight')


def show_entropy(data, data1, data2, pictdir):
    # entropy
    color = 'b'
    fig = plt.figure(figsize=(7, 6), dpi=100)
    res = [np.mean(all_inp_entropy(data, pos=pos)) for pos in range(20)]
    res1 = [np.mean(all_inp_entropy(data1, pos=pos)) for pos in range(20)]
    res2 = [np.mean(all_inp_entropy(data2, pos=pos)) for pos in range(20)]
    # res = [np.mean(all_inp_entropy(data))]
    # plt.plot(range(1, len(res) + 1), res, lw=2., color=color)
    # plt.scatter(range(1, len(res) + 1), res, lw=3.0, color=color)
    plt.plot(range(2, len(res) + 2), res, 'bv-', label='Baseline w/o Enc: extra skip connection')
    plt.plot(range(2, len(res) + 2), res1, 'g^--', label='Baseline')
    plt.plot(range(2, len(res) + 2), res2, 'ro-', label='Ours')

    plt.xlabel("target token position", size=25)
    plt.ylabel("entropy", size=25)
    plt.yticks(size=20)
    plt.xticks([1, 5, 10, 15, 17], size=17)
    plt.title('Entropy of visual contributions', size=25)
    plt.grid()
    plt.legend()
    # pylab.savefig(pictdir + 'YOUR FNAME', bbox_inches='tight')
    pylab.savefig(pictdir, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DIFNet')
    parser.add_argument('--exp_name', type=str, nargs='+', default='DIFNet', help='select one or more lrp_results for showing')
    parser.add_argument('--out_path', type=str, default='./output/output_lrp')
    # parser.add_argument('--mode', type=str, default='base', choices=['base', 'base_lrp', 'difnet', 'difnet_lrp'])
    args = parser.parse_args()

    save_img_contribution = os.path.join(args.out_path, '{}.jpg'.format('contribution'))
    save_img_entropy = os.path.join(args.out_path, '{}.jpg'.format('entropy'))

    all_data = []
    for exp in args.exp_name:
        fname = '{}_result.pkl'.format(exp)
        data = pickle.load(open(os.path.join(args.out_path, fname), 'rb'))
        all_data.append(data)
    show_source2target(all_data, args.exp_name, save_img_contribution)