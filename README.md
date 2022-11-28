# DIFNet: A PyTorch Implementation
<p align="center">
  <img src="difnet.png" alt="DIFNet" width="750"/>
</p>

This repository contains the official code for our paper [DIFNet: Boosting Visual Information Flow for Image Captioning](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_DIFNet_Boosting_Visual_Information_Flow_for_Image_Captioning_CVPR_2022_paper.pdf) (CVPR 2022).

If our work is helpful to you or gives some inspiration to you, please star this project and cite our paper. Thank you!
```
@inproceedings{wu2022difnet,
  title={DIFNet: Boosting Visual Information Flow for Image Captioning},
  author={Wu, Mingrui and Zhang, Xuying and Sun, Xiaoshuai and Zhou, Yiyi and Chen, Chao and Gu, Jiaxin and Sun, Xing and Ji, Rongrong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18020--18029},
  year={2022}
}
```

## Installation
Clone the repository and create the `difnet` conda environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate difnet
```

Then download spacy data by executing the following command:
```
python -m spacy download en
```

Add evaluation module from [evaluation](https://github.com/aimagelab/meshed-memory-transformer/tree/master/evaluation).

Note: Python 3.6+ and Pytorch 1.6+ are required to run our code. 


## Data preparation
To run the code, annotations and detection features for the COCO dataset are needed. Please download the annotations file [annotations.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing) and extract it.

Detection features are computed based on the project [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa). To reproduce our results, please extract the raw COCO grid features process them according to the project [RSTNet](https://github.com/zhangxuying1004/RSTNet). You can also alternatively download the processed image features [coco_grid_feats](https://pan.baidu.com/s/1myelTYJE8a1HDZHkoccfIA) with the extraction code ```cvpr``` for convenience.

Segmentation features are computed with the code provided by [UPSNet](https://github.com/uber-research/UPSNet). To reproduce our result, please download the segmentation features file [segmentations.zip](https://drive.google.com/file/d/1R7GL9FTZgc0cpCoJ6UGWNuhvAiDciab7/view?usp=sharing) (~83M) and extract it.


## Evaluation
To reproduce the results reported in our paper, download the pretrained model file [DIFNet_lrp.pth](https://drive.google.com/file/d/1aDuiiIJomAvQlS-N7VTsqD45rnOt5Oj2/view?usp=sharing) and place it in the saved_transformer_models folder.

Run `sh test.sh` using the following arguments:

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name|
| `--mode` | select a model mode, ['base', 'base_lrp', 'difnet', 'difnet_lrp']|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--features_path` | Path to detection features file |
| `--pixel_path` | Path to pixel file |
| `--annotation_folder` | Path to folder with COCO annotations |

#### Expected output
Under `output_logs/`, you may also find the expected output of the evaluation code.


## Training procedure
Run `python train.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name|
| `--mode` | select a model mode, ['base', 'base_lrp', 'difnet', 'difnet_lrp']|
| `--batch_size` | Batch size (default: 50) |
| `--workers` | Number of workers (default: 4) |
| `--head` | Number of heads (default: 8) |
| `--warmup` | Warmup value for learning rate scheduling (default: 10000) |
| `--resume_last` | If used, the training will be resumed from the last checkpoint. |
| `--resume_best` | If used, the training will be resumed from the best checkpoint. |
| `--features_path` | Path to detection features file |
| `--pixel_path` | Path to segmentation feature file |
| `--annotation_folder` | Path to folder with COCO annotations |
| `--logs_folder` | Path folder for tensorboard logs (default: "tensorboard_logs")|

mode
```
base: baseline model
base_lrp: baseline model with lrp
difnet: DIFNet
difnet_lrp: DIFNet with lrp

```
For example, to train our model with the parameters used in our experiments, use 
```
sh train.sh
```
For test,
```
sh test.sh
```

For LRP(first generate caption.json file with `generate_caption.py`, and then use `lrp_total.py` to generate lrp_result.pkl file, finally use `show_lrp.py` to show lrp_result.),
```
sh lrp.sh
```
When the cache can't release, use(for example, nvidia0 for release GPU0)
```
fuser -v /dev/nvidia0 |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh
```

## Acknowledge
This repo is based on [M^2 Transformer](https://github.com/aimagelab/meshed-memory-transformer), [the-story-of-heads](https://github.com/lena-voita/the-story-of-heads) and [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability).
