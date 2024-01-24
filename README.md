# UV Volumes for Real-time Rendering of Editable Free-view Human Performance
**[Project Page](https://fanegg.github.io/UV-Volumes) | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_UV_Volumes_for_Real-Time_Rendering_of_Editable_Free-View_Human_Performance_CVPR_2023_paper.pdf) | [Latest arXiv](https://arxiv.org/pdf/2203.14402.pdf) | [Supplementary](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Chen_UV_Volumes_for_CVPR_2023_supplemental.pdf)**

> UV Volumes for Real-time Rendering of Editable Free-view Human Performance  
> [Yue Chen*](https://fanegg.github.io/), [Xuan Wang*](https://xuanwangvc.github.io/), [Xingyu Chen](http://rover-xingyu.github.io/), [Qi Zhang](https://qzhang-cv.github.io/), [Xiaoyu Li](https://xiaoyu258.github.io/), [Yu Guo†](https://yuguo-xjtu.github.io/), [Jue Wang](https://juewang725.github.io/), [Fei Wang](https://scholar.google.com/citations?hl=zh-CN&user=uU2JTpUAAAAJ&view_op=list_works)  
> (* equal contribution，† corresponding author)  
> CVPR 2023

[![UV Volumes for Real-time Rendering of Editable Free-view Human Performance](https://res.cloudinary.com/marcomontalbano/image/upload/v1678176939/video_to_markdown/images/youtube--JftQnXLMmPc-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/JftQnXLMmPc "UV Volumes for Real-time Rendering of Editable Free-view Human Performance")

This repository is an official implementation of [UV-Volumes](https://fanegg.github.io/UV-Volumes) using [pytorch](https://pytorch.org/).

## Installation

Please see [INSTALL.md](INSTALL.md) for manual installation.

## Run the code on ZJU-MoCap

Please see [INSTALL.md](INSTALL.md) to download the dataset.

<!-- We provide the pretrained models at [here](https://). -->

### Training on ZJU-MoCap

Take the training on `sequence 313` as an example.

```
python3 train_net.py --cfg_file configs/zju_mocap_exp/313.yaml exp_name zju313 resume False output_depth True
```
You can monitor the training process by Tensorboard.
```
tensorboard --logdir data/record/UVvolume_ZJU
```

### Test on ZJU-MoCap

Take the test on `sequence 313` as an example.

<!-- 1. Download the corresponding pretrained model and put it to `$ROOT/data/trained_model/UVvolume_ZJU/zju313/latest.pth`. -->

```
python3 run.py --type evaluate --cfg_file configs/zju_mocap_exp/313.yaml exp_name zju313 use_lpips True test.frame_sampler_interval 1 use_nb_mask_at_box True save_img True T_threshold 0.75 
```


## Run the code on CMU Panoptic

Please see [INSTALL.md](INSTALL.md) to download and process the dataset.

<!-- We provide the pretrained models at [here](https://). -->

### Training on CMU Panoptic

Take the training on `171204_pose4_sample6` as an example.

```
python3 train_net.py --cfg_file configs/cmu_exp/p4s6.yaml exp_name p4s6 resume False output_depth True
```
You can monitor the training process by Tensorboard.
```
tensorboard --logdir data/record/UVvolume_CMU
```

### Test on CMU Panoptic

Take the test on `171204_pose4_sample6` as an example.

<!-- 1. Download the corresponding pretrained model and put it to `$ROOT/data/trained_model/UVvolume_CMU/p4s6/latest.pth`. -->

```
python3 run.py --type evaluate --cfg_file configs/cmu_exp/p4s6.yaml exp_name p4s6 use_lpips True test.frame_sampler_interval 1 use_nb_mask_at_box True save_img True
```


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{chen2023uv,
  title={UV Volumes for real-time rendering of editable free-view human performance},
  author={Chen, Yue and Wang, Xuan and Chen, Xingyu and Zhang, Qi and Li, Xiaoyu and Guo, Yu and Wang, Jue and Wang, Fei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16621--16631},
  year={2023}
}
```


## Acknowledge
Our code is based on the awesome pytorch implementation of [NeuralBody](https://github.com/zju3dv/neuralbody). We appreciate all the contributors.
