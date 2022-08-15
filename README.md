# Low-resolution Human Pose Estimation

## Introduction
This is an official pytorch implementation of [*Low-resolution Human Pose Estimation*](https://arxiv.org/pdf/2109.09090.pdf). 

This work bridges the learning gap between heatmap and offset field especially for low-resolution human pose estimation.

![Illustrating the principle of the proposed CAL](/figures/CAL.png)

## Main Results

### Results on COCO val2017 with $64\times48$ input resolution setting
| Method     | Backbone  |    AP | Ap.5 | AP.75 | AP(M) | AP(L) |   AR |
|------------|-----------|-------|------|-------|-------|-------|------|
| HRNet      | HRNet-W32 |  29.7 | 75.7 | 13.1 | 29.3 | 30.7 | 37.3
| HRNet      | HRNet-W48 |  32.4 | 78.3 | 16.2 | 31.5 | 34.0 | 39.3
| UDP        | HRNet-W32 |  47.4 | 80.5 | 50.6 | 47.7 | 47.7 | 53.8
| UDP        | HRNet-W48 |  51.0 | 82.6 | 55.2 | 51.4 | 51.0 | 57.3
| HRNet+SPSR | HRNet-W32 |  50.0 | 81.6 | 53.6 | 53.6 | 46.0 | 55.3
| HRNet+SPSR | HRNet-W48 |  51.2 | 82.8 | 55.5 | 54.6 | 47.7 | 56.4
| UDP+SPSR   | HRNet-W32 |  52.5 | 80.8 | 57.7 | 56.2 | 48.3 | 57.4
| UDP+SPSR   | HRNet-W48 |  54.1 | 82.4 | 59.0 | 56.9 | 50.6 | 59.4 |
| CAL        | HRNet-W32 | **58.4** | **86.6** | **65.1** | **57.3** | **60.5** | **64.8** |
| CAL        | HRNet-W48 |  **61.5** | **88.1** | **68.7** | **60.7** | **63.5** | **66.3** |

### Note:
- Flip test is used.
- \+SPSR means using  [SPSR](https://arxiv.org/abs/2003.13081) model to recover super-resolution images for pose estimation.
<!-- - - Person detector has person AP of 60.9 on COCO test-dev2017 dataset.
- GFLOPs is for convolution and linear layers only.
\* means using additional data from [AI challenger](https://challenger.ai/dataset/keypoint) for training.
- \- means the detector ensemble with [HTC](https://github.com/open-mmlab/mmdetection) and [SNIPER](https://github.com/mahyarnajibi/SNIPER).


### Results on MPII val
| PCKh | Baseline             | Head | Shoulder | Elbow | Wrist |  Hip | Knee | Ankle |
|------|----------------------|------|----------|-------|-------|------|------|-------|
| 0.5  | HRNet_w32            | 97.1 | 95.9 | 90.3 | 86.5 | 89.1 | 87.1 | 83.3 | 90.3 |
| 0.5  | **HRNet_w32 + DARK** | 97.2 | 95.9 | 91.2 | 86.7 | 89.7 | 86.7 | 84.0 | 90.6 |
| 0.1  | HRNet_w32            | 51.1 | 42.7 | 42.0 | 41.6 | 17.9 | 29.9 | 31.0 | 37.7 |
| 0.1  | **HRNet_w32 + DARK** | 55.2 | 47.8 | 47.4 | 45.2 | 20.1 | 33.4 | 35.4 | 42.0 |

### Note:
- Flip test is used.
- Input size is 256x256
- GFLOPs is for convolution and linear layers only.
-  -->

## Development environment

The code is developed using python 3.5 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA 2080TI GPU cards. Other platforms or GPU cards are not fully tested.  

## Quick start

### 1. Preparation

#### 1.1 Prepare the dataset
For the MPII dataset, your directory tree should look like this:   
```
$HOME/datasets/MPII
├── annot
├── images
└── mpii_human_pose_v1_u12_1.mat
```
For the COCO dataset, your directory tree should look like this:   
```
$HOME/datasets/MSCOCO
├── annotations
├── images
│   ├── test2017
│   ├── train2017
│   └── val2017
└── person_detection_results
    ├── COCO_val2017_detections_AP_H_56_person.json
    └── COCO_test-dev2017_detections_AP_H_609_person.json
````

### 1.2 Prepare the pretrained models
Your directory tree should look like this:  
```
$HOME/datasets/models
└── pytorch
    ├── imagenet
    │   ├── hrnet_w32-36af842e.pth
    │   ├── hrnet_w48-8ef0771d.pth
    │   ├── resnet50-19c8e357.pth
    │   ├── resnet101-5d3b4d8f.pth
    │   └── resnet152-b121ed2d.pth
    ├── pose_coco
    │   ├── hg4_128×96.pth
    │   ├── hg8_128×96.pth
    │   ├── r50_128×96.pth
    │   ├── r101_128×96.pth
    │   ├── r152_128×96.pth
    │   ├── w32_128×96.pth
    │   ├── w48_128×96.pth
    │   ├── w32_256×192.pth
    │   ├── w32_384×288.pth
    │   └── w48_384×288.pth
    └── pose_mpii
        └── w32_256×256.pth
```

### 1.3 Prepare the environment
Setting the parameters in the file `prepare_env.sh` as follows:

```bash
# DATASET_ROOT=$HOME/datasets
# COCO_ROOT=${DATASET_ROOT}/MSCOCO
# MPII_ROOT=${DATASET_ROOT}/MPII
# MODELS_ROOT=${DATASET_ROOT}/models
```

Then execute:

```bash
bash prepare_env.sh
```

If you like, you can [**prepare the environment step by step**](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

### 2. Training and Testing

Testing using model zoo's models [[GoogleDrive]]() [[BaiduDrive]]()
```bash
# testing
cd scripts
./tools/run_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE}

# Training
cd scripts
./tools/run_train.sh ${CONFIG_FILE}
```

Examples:

Assume that you have already downloaded the pretrained models and place them like the section 1.2.

1. Testing on COCO dataset using HRNet_W32_64×48 model.
```bash
cd scripts
bash run_test.sh experiments/coco/hrnet/w32_64x48_adam_lr1e-3.yaml \
    models/pytorch/pose_coco/w32_64x48.pth
```
2. Training on COCO dataset.
```bash
cd scripts
bash run_train.sh experiments/coco/hrnet/w32_64x48_adam_lr1e-3.yaml
```

### Citation

If you use our code or models in your research, please cite with:

```
@article{wang2022low,
  title={Low-resolution human pose estimation},
  author={Wang, Chen and Zhang, Feng and Zhu, Xiatian and Ge, Shuzhi Sam},
  journal={Pattern Recognition},
  volume={126},
  pages={108579},
  year={2022}
}
```

### Discussion forum
[ILovePose](http://www.ilovepose.cn)

## Acknowledgement
Thanks for the open-source DARK, UDP and HRNet
* [Distribution Aware Coordinate Representation for Human Pose Estimation](https://github.com/ilovepose/DarkPose)
* [The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation](https://github.com/HuangJunJie2017/UDP-Pose)
* [Deep High-Resolution Representation Learning for Human Pose Estimation, Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/)