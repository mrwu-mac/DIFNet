# DIFNet: A PyTorch Implementation
This repository contains the reference code for the paper DIFNet.

## Installation
Clone the repository and create the `difnet` conda environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate dif
```

Then download spacy data by executing the following command:
```
python -m spacy download en
```

Add evaluation module from [evaluation](https://github.com/aimagelab/meshed-memory-transformer/tree/master/evaluation).

Note: Python 3.6+ is required to run our code. 


## Data preparation
To run the code, annotations and detection features for the COCO dataset are needed. Please download the annotations file [annotations.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing) and extract it.

Detection features are computed with the code provided by [1]. To reproduce our result, please download the COCO features file [coco_detections.hdf5](https://drive.google.com/open?id=1MV6dSnqViQfyvgyHrmAT_lLpFbkzp3mx) (~53.5 GB), in which detections of each image are stored under the `<image_id>_features` key. `<image_id>` is the id of each COCO image, without leading zeros (e.g. the `<image_id>` for `COCO_val2014_000000037209.jpg` is `37209`), and each value should be a `(N, 2048)` tensor, where `N` is the number of detections. 

Segmentation features are computed with the code provided by [2]. To reproduce our result, please download the segmentation features file [segmentations.zip](https://drive.google.com/file/d/1x6RVn01eZKRtoNfZZh4tUSF57N40YxAW/view?usp=sharing) (~83M) and extract it.


## Evaluation
To reproduce the results reported in our paper, download the pretrained model file [DIFNet.pth](https://drive.google.com/file/d/1aDuiiIJomAvQlS-N7VTsqD45rnOt5Oj2/view?usp=sharing) and place it in the saved_transformer_models folder.

Run `python test.py` using the following arguments:

| Argument | Possible values |
|------|------|
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
transformer: baseline

```
For example, to train our model with the parameters used in our experiments, use
```
CUDA_VISIBLE_DEVICES=0 python train.py --mode difnet_lrp --exp_name DIFNet
```
For test,
```
CUDA_VISIBLE_DEVICES=0 python test.py --mode difnet_lrp --exp_name DIFNet
```

For LRP, you must generate caption.json file with test1.py and then use following command to generate lrp_result.pkl file, and then use show_lrp.py to show lrp_result.
```
CUDA_VISIBLE_DEVICES=0 python lrp_total.py --mode difnet_lrp --exp_name DIFNet
```
When the cache can't release, use(for example, nvidia0 for release GPU0)
```
fuser -v /dev/nvidia0 |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh
```

#### References
[1] HuaizuJiang,IshanMisra,MarcusRohrbach,ErikLearned- Miller, and Xinlei Chen. In defense of grid features for visual question answering. In Proceedings of the IEEE/CVF Con- ference on Computer Vision and Pattern Recognition, 2020.

[2] Yuwen Xiong, Renjie Liao, Hengshuang Zhao, Rui Hu, Min Bai, Ersin Yumer, and Raquel Urtasun. Upsnet: A unified panoptic segmentation network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

[3] Marcella Cornia, Matteo Stefanini, Lorenzo Baraldi, and Rita Cucchiara. Meshed-memory transformer for image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.