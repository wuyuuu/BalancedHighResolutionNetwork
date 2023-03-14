# Real-time Vision-based Multi-person Pose Estimation via Multi-branch Neural Networks

## Introduction

This is an official implementation for our work Balanced HRNet. 

![an](overview.png)


## Quick Start

### Installation

Please follow the installation process of [mmpose](https://github.com/open-mmlab/mmpose).

### Data Preparation


**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation.
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

## Training

### Training and testing on COCO dataset

```shell
# train with a signle GPU
python tools/train.py ${CONFIG_FILE} [optional arguments]

# train with multiple GPUs
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```
For more details, please refer to [mmpose](https://github.com/open-mmlab/mmpose).

## Android Implementation

