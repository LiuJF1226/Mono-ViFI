<div id="top" align="center">
  
# Mono-ViFI: A Unified Learning Framework for Self-supervised Single- and Multi-frame Monocular Depth Estimation
<!-- **Mono-ViFI: A Unified Learning Framework for Self-supervised Single- and Multi-frame Monocular Depth Estimation** -->
  
  Jinfeng Liu, [Lingtong Kong](https://scholar.google.com/citations?hl=zh-CN&user=KKzKc_8AAAAJ), [Bo Li](https://libraboli.github.io/), Zerong Wang, Hong Gu and Jinwei Chen

  vivo Mobile Communication Co., Ltd

  ECCV 2024 
  <!-- [[paper link]](https://arxiv.org/abs/2309.05254) -->

<!-- <p align="center">
  <img src="assets/demo.gif" alt="example input output gif" width="450" />
</p>
BDEdepth (HRNet18 640x192 KITTI) -->
</div>

## Table of Contents
- [Description](#description)
- [Setup](#setup)
- [Preparing datasets](#datasets)
  - [KITTI](#kitti)
  - [Make3D](#nm)
  - [Cityscapes](#cityscapes)
- [VFI Pre-training](#VFI)
- [Mono-ViFI Training](#training)
- [Evaluation](#evaluation)
  - [Evaluate with single-frame model](#single)
  - [Evaluate with multi-frame model](#multi)
- [Prediction](#prediction)
  - [Prediction for a single image](#image)
  - [Prediction for a video](#video)
- [Mono-ViFI Weights](#weights)
- [Related Projects](#acknowledgement)


## Description
This is the official PyTorch implementation for Mono-ViFI, which is built on the codebase of [BDEdepth](https://github.com/LiuJF1226/BDEdepth). If you find our work useful in your research, please consider citing our paper:

```

```



## Setup
Install the dependencies with:
```shell
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```


## <span id="datasets">Preparing datasets</span>
### KITTI
For KITTI dataset, you can prepare them as done in [Monodepth2](https://github.com/nianticlabs/monodepth2). Note that we directly train with the raw png images and do not convert them to jpgs. You also need to generate the groundtruth depth maps before training since the code will evaluate after each epoch. For the raw KITTI groundtruth (`eigen` eval split), run the following command. This will generate `gt_depths.npz` file in the folder `splits/kitti/eigen/`.
```shell
python export_gt_depth.py --data_path /home/datasets/kitti_raw_data --split eigen
```
For the improved KITTI groundtruth (`eigen_benchmark` eval split), please directly download it in this [link](https://www.dropbox.com/scl/fi/dg7eskv5ztgdyp4ippqoa/gt_depths.npz?rlkey=qb39aajkbhmnod71rm32136ry&dl=0). And then move the downloaded file (`gt_depths.npz`) to the folder `splits/kitti/eigen_benchmark/`.

### <span id="nm">Make3D</span>

For Make3D dataset, you can download it from [here](http://make3d.cs.cornell.edu/data.html#make3d).

### Cityscapes
For Cityscapes dataset, we follow the instructions in [ManyDepth](https://github.com/nianticlabs/manydepth). First Download `leftImg8bit_sequence_trainvaltest.zip` and `camera_trainvaltest.zip` in its [website](https://www.cityscapes-dataset.com/), and unzip them into a folder `/path/to/cityscapes/`. Then preprocess CityScapes dataset using the followimg command:
```shell
python prepare_cityscapes.py \
--img_height 512 \
--img_width 1024 \
--dataset_dir /path/to/cityscapes \
--dump_root /path/to/cityscapes_preprocessed \
--seq_length 3 \
--num_threads 8
```
Remember to modify `--dataset_dir` and `--dump_root` to your own path. The ground truth depth files are provided by ManyDepth in this [link](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip), which were converted from pixel disparities using intrinsics and the known baseline. Download it and unzip into `splits/cityscapes/`


## <span id="VFI">VFI Pre-training</span>

Download the following 6 checkpoints related to VFI in this [link](https://www.dropbox.com/scl/fo/0zeefm9e4kv0fzumqp490/h?rlkey=ev09rshvarnoyj9kr1qymkppl&dl=0):
* small IFRNet pretrained on Vimeo90K dataset : `IFRNet_S_Vimeo90K.pth`
* large IFRNet pretrained on Vimeo90K dataset : `IFRNet_L_Vimeo90K.pth`
* small IFRNet pretrained on KITTI dataset : `IFRNet_S_KITTI.pth`
* large IFRNet pretrained on KITTI dataset : `IFRNet_L_KITTI.pth`
* small IFRNet pretrained on Cityscapes dataset : `IFRNet_S_CS.pth`
* large IFRNet pretrained on Cityscapes dataset : `IFRNet_L_CS.pth`


To save time, you can skip VFI pre-training and directly use our provided checkpoints. Just create a folder `Mono-ViFI/weights/` and move `IFRNet_S_KITTI.pth, IFRNet_L_KITTI.pth, IFRNet_S_CS.pth, IFRNet_L_CS.pth` to this folder.

If you want to train VFI models by yourself, move `IFRNet_L_Vimeo90K.pth, IFRNet_S_Vimeo90K.pth` to the folder `Mono-ViFI/weights/`. We load Vimeo90K checkpoints to train on KITTI/Cityscapes. All the VFI training configs are in the folder `configs/vfi/`. For example,
the command for training large IFRNet on KITTI is:
```shell
### Training large IFRNet on KITTI
# single-gpu
CUDA_VISIBLE_DEVICES=0 python train_vfi.py -c configs/vfi/IFRNet_L_KITTI.txt

# multi-gpu
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_vfi.py -c configs/vfi/IFRNet_L_KITTI.txt
```

## <span id="training">Mono-ViFI Training</span>
Before training, move another 2 checkpoints downloaded from this link to the folder `Mono-ViFI/weights/`:
* HRNet18 backbone pretrained on ImageNet : `HRNet_W18_C_cosinelr_cutmix_300epoch.pth.tar`
* LiteMono backbone pretrained on ImageNet : `lite-mono-pretrain.pth`

You can refer to config files for the training settings/parameters/paths. All training configs are in the folders:
* ResNet18 backbone : `configs/resnet18`
* LiteMono backbone : `configs/litemono`
* D-HRNet backbone : `configs/dhrnet`

Remember to modify related paths to your own. Take ResNet18 as an example, the training commands are as follows.

Note: you can adjust `batch_size` according to your maximum GPU memory.

```shell
### Training with ResNet18 backbone (KITTI, 640x192)
# single-gpu
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/resnet18/ResNet18_KITTI_MR.txt

# multi-gpu
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py -c configs/resnet18/ResNet18_KITTI_MR.txt


### Training with ResNet18 backbone (KITTI, 1024x320)
# For 1024x320 resolution, we load 640x192 model and train for 5 epoches with 1e-5 learning rate.
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/resnet18/ResNet18_KITTI_HR.txt


### Training with ResNet18 backbone (Cityscapes, 512x192)
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/resnet18/ResNet18_CS.txt
```


## Evaluation
### <span id="single">Evaluate with single-frame model</span>
```shell
### KITTI 640x192 model, ResNet18

CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py \
--pretrained_path our_models/ResNet18_KITTI_MR.pth \
--backbone ResNet18 \
--batch_size 16 \
--width 640 \
--height 192 \
--kitti_path /data/juicefs_sharing_data/public_data/Datasets/KITTI/kitti_raw_data \
--make3d_path /data/juicefs_sharing_data/public_data/Datasets/make3d \
--cityscapes_path /data/juicefs_sharing_data/public_data/Datasets/cityscapes \
# --post_process
```
This script will evaluate on KITTI (both raw and improved GT), Make3D and Cityscapes together. If you don't want to evaluate on some of these datasets, for example KITTI, just do not specify the corresponding `--kitti_path` flag. It will only evaluate on the datasets which you have specified a path flag.

If you want to evalute with post-processing, add the `--post_process` flag (disabled by default).

### <span id="multi">Evaluate with multi-frame model</span>
```shell
### KITTI 640x192 model, ResNet18

CUDA_VISIBLE_DEVICES=0 python evaluate_depth_mf.py \
--pretrained_path our_models/ResNet18_KITTI_MR.pth \
--backbone ResNet18 \
--vfi_scale small \
--training_data kitti \
--batch_size 16 \
--width 640 \
--height 192 \
--kitti_path /data/juicefs_sharing_data/public_data/Datasets/KITTI/kitti_raw_data \
--cityscapes_path /data/juicefs_sharing_data/public_data/Datasets/cityscapes \
```

## Prediction

### <span id="image">Prediction for a single image (only single-frame model)</span>
You can predict the disparity (inverse depth) for a single image with:

```shell
python test_simple.py --image_path folder/test_image.png --pretrained_path our_models/DHRNet_KITTI_MR.pth --backbone DHRNet --height 192 --width 640 --save_npy
```

The `--image_path` flag can also be a directory containing several images. In this setting, the script will predict all the images (use `--ext` to specify png or jpg) in the directory:

```shell
python test_simple.py --image_path folder --pretrained_path our_models/DHRNet_KITTI_MR.pth --backbone DHRNet --height 192 --width 640 --ext png --save_npy
```

### <span id="video">Prediction for a video (both single- and multi-frame model)</span>

```shell
python test_video.py --image_path folder --pretrained_path our_models/DHRNet_KITTI_MR.pth --backbone DHRNet --vfi_scale small --training_data kitti --height 192 --width 640 --ext png --save_npy
```
Here the `--image_path` flag should be a directory containing several video frames. Note that these video frame files should be named in an ascending numerical order. For example, the first frame is named as `0000.png`, the second frame is named as `0001.png`, and etc. This command will also output a GIF file.

## <span id="weights">Mono-ViFI Weights</span>


## <span id="acknowledgement">Related Projects</span>
* [Monodepth2](https://github.com/nianticlabs/monodepth2) (ICCV 2019)
* [ManyDepth](https://github.com/nianticlabs/manydepth) (CVPR 2021)
* [Lite-Mono](https://github.com/noahzn/Lite-Mono) (CVPR 2023)
* [PlaneDepth](https://github.com/svip-lab/PlaneDepth) (CVPR 2023)
* [RA-Depth](https://github.com/hmhemu/RA-Depth) (ECCV 2022)
* [BDEdepth](https://github.com/LiuJF1226/BDEdepth) (IEEE RA-L 2023)
* [IFRNet](https://github.com/ltkong218/IFRNet) (CVPR 2022, our employed VFI model)
