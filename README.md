# MetaPose

This repo is the official implementation for **MetaPose: Multimodal Enhancement and Transformation Alignment for Accurate 3D Human Pose Estimation**.

## Demo Video

<div align="center">
<img src="https://github.com/Visual-Pose-Lab/MetaPose/blob/main/figure/MetaPose_demo.gif" alt='Demo Video'>
</div>

## Introduction

**Abstract.** We propose MetaPose, a novel framework that leverages **M**ultimodal **E**nhancement and **T**ransformation **A**lignment for accurate single-frame 3D human pose estimation. MetaPose integrates spatial, visual, and semantic information by leveraging two innovative modules: Spatial-Driven Modality Enhancement and Semantic-Driven Modality Enhancement. The Spatial-Driven module utilizes 2D keypoints to guide deformable attention sampling on image features, effectively capturing positional and depth cues. The Semantic-Driven module encodes joint description using a Text Encoder, applying iterative deformable attention to capture structural relationships between body joints through language guidance. To unify these enhanced features, we propose a Transformation Alignment module that aligns different modalities into a unified distribution via learned normal distributions and cross-attention mechanisms. This unified pose representation is then lifted to 3D pose. Extensive experiments demonstrate that MetaPose achieves state-of-the-art performance on multiple benchmarks, validating the effectiveness of our approach.

<p align="center">
<img src='https://github.com/Visual-Pose-Lab/MetaPose/blob/main/figure/fig01.png' width='500'>
</p>

Previous lifting models typically included only 2D (a) joint coordinates, with some approaches integrating (b) image features. Our method combines (a) joint coordinates, (b) image features, and joint (c) text features to strengthen multimodal integration and improve pose estimation performance.

## Environment

The code is developed and tested under the following environment.

- Python 3.8.19
- PyTorch 1.12.1
- CUDA 11.3

```
cd metapose
conda create --name metapose --file conda_requirements.txt
conda activate metapose
pip install -r requirements.txt
```

## Human3.6M

### Preparation

1. Please refer to [H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox) and [CA-PF](https://github.com/QitaoZhao/ContextAware-PoseFormer) to prepare the dataset images, as we follow the same processing procedure. All images should be placed in the `images/` directory, while `h36m_train.pkl` and `h36m_validation.pkl` should be stored in the `data/` directory.

   **Note**: In our implementation, we directly saved the cropped video frames. If you wish to use the full video frames, certain modifications to the code will be required.

2. Download (COCO) pretrained weights for [HRNet-32/48](https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA) and place them under `data/pretrained/coco/`

3. After the above steps, your folder should look like this:

```
root/
├── data/
│   ├── h36m_train.pkl
│   ├── h36m_validation.pkl
│   └── pretrained/
│       └── coco/
│           ├── pose_hrnet_w32_256x192.pth
│           ├── pose_hrnet_w48_256x192.pth
│           └── README.MD
└── images/
        ├── s_01_act_02_subact_01_ca_01/
        │   ├── s_01_act_02_subact_01_ca_01_000001.jpg
        │   ├── ...
        │   └── s_01_act_02_subact_01_ca_01_001384.jpg
        ├── s_01_act_02_subact_01_ca_02/
        ├── ...
        └── s_11_act_16_subact_02_ca_04/
├── experiments/
├── datasets/
├── models/
│   ├── ...
│   ├── metapose.py
├── utils/
├── logs/
├── figure/
├── ...
└── train.py

```

### Train

You can train **MetaPose** with the following commands:

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=2345 train.py --config experiments/human36m/human36m.yaml --image_encoder hrnet_32 --text_encoder ViT-B/32 --logdir ./logs

# Multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=2345 train.py --config experiments/human36m/human36m.yaml --image_encoder hrnet_32 --text_encoder ViT-B/32  --logdir ./logs
```

### Test

Place the pre-trained model weights under `checkpoint/`, and run:

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=2345 train.py --config experiments/human36m/human36m.yaml --image_encoder hrnet_32 --text_encoder ViT-B/32 --logdir ./logs --eval

# Multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=2345 train.py --config experiments/human36m/human36m.yaml --image_encoder hrnet_32 --text_encoder ViT-B/32 --logdir ./logs --eval
```

## MPI-INF-3DHP

Coming Soon...

## Acknowledgment

Our code refers to the following repositories.

- [ContextPose](https://github.com/ShirleyMaxx/ContextPose-PyTorch-release)
- [CA-PF](https://github.com/QitaoZhao/ContextAware-PoseFormer)
- [H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox)

We thank the authors for releasing their codes.
