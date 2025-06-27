1. Please refer to [CA-PF](https://github.com/QitaoZhao/ContextAware-PoseFormer) and [H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox) to set up RGB images from the Human3.6M dataset. All RGB images should be put here. 

    **Note**: Only RGB images take around 200GB of disk space, while intermediate files take even more space. Carefully remove these intermediate files after each step if you do not have sufficient disk space.

2. Download `data_2d_h36m_cpn_ft_h36m_dbb.npz` from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md) and put it under `H36M-Toolbox/data/`. This file contains pre-processed CPN-detected keypoints.

3. Run `generate_labels_h36m.py` to generate labels (`h36m_train.pkl` and `h36m_validation.pkl`) for training and testing. It may take a while.

4. **Pre-processed data**: please check [here](https://drive.google.com/drive/folders/1OYKWnu_5GPLRfceD3Psf4-JZkloBodKx) for the pre-processed labels if you don't want to generate them yourself, but note that you still need to download the images from the dataset following the first step.

For ease of implementation, we borrowed this description from [CA-PF](https://github.com/QitaoZhao/ContextAware-PoseFormer).
