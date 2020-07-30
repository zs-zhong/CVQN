# CVQN
***********************************************************************************************************
This repository is for "Channel-Level Variable Quantization Network for Deep Image Compression"
(to appear in IJCAI, 2020)
By [Zhisheng Zhong](https://zzs1994.github.io), Hiroaki Akutsu and [Kiyoharu Aizawa](https://www.hal.t.u-tokyo.ac.jp/~aizawa/).


***********************************************************************************************************
## Table of contents
- [Overview](#overview)
- [Data Download](#data-download)
- [Folder Structure](#folder-structure)
- [Training and Evaluation](#training-and-evaluation)
***********************************************************************************************************

# Overview
<center>Framework of the channel-level variable quantization network.</center>
<div align=center><img src="https://github.com/zzs1994/CVQN/blob/master/page_image/overview_CVQN.jpg" width="90%" height="90%"></div align=center>

# Dependencies
- Python (3.7.5)
- PyTorch (1.2.0)
- torchvision (0.4.0)
- PyYaml (5.2)
- tensorboard (2.0.1)


# Data Download
These training datasets can be downloaded from the above links.

- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K)
- [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)
- [CLIC2019](https://www.compression.cc/challenge)

# Folder Structure
Your CVQN folder may be similar to this:

```
--logs (log folder)
--ckps (checkpoint folder)
--tbs (tensorboard log folder)
--yaml (yaml folder)
--pytorch_msssim
--config
--*.py
```

# Training and Evaluation
Please modify the training & evaluation dataset path in `yaml/XXX.yaml`. 

You can also modify other parameters to change the model and training strategy in the same file. 

An example to train a model:


```bash
python main_train_eval.py --config yaml/XXX.yaml
```

### Citation
If you find this code useful, please cite our paper:

```
@inproceedings{Zhong2020CVQN,
  title     = {Channel-Level Variable Quantization Network for Deep Image Compression},
  author    = {Zhong, Zhisheng and Akutsu, Hiroaki and Aizawa, Kiyoharu},
  booktitle = {Proceedings of the International Joint Conference on Artificial Intelligence},
  pages     = {467--473},
  year      = {2020}
}
```