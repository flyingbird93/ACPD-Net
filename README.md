# Aesthetic and Composition Collaboration Driven Image Cropping（A3C-Net）

This repository contains a pytorch implementation of the paper "Aesthetic and Composition Collaboration Driven Image Cropping"(Subject to ICASSP2023)

In this paper, we present an **A**esthetic **C**omposition **C**ollaboration **C**ropping **Net**work~(A3C-Net) to explicitly investigate the collaboration of image aesthetics and image composition for the cropping task in an end-to-end manner.

## Pipeline
![pipeline](https://user-images.githubusercontent.com/16755407/194859123-75929459-027d-4284-a261-daf91140f40d.png)


## Requirements
- System: **Linux**(e.g. Ubuntu/CentOS/Arch), macOS, or **Windows** Subsystem of Linux (WSL)
- Python version >=3.6
- Pytorch == 1.10.2 Cuda == 11.2 
- TensorboardX
- Opencv == 4.5.5
- mmdet == 2.24.1

## Install
- Clone repo
```python
git clone https://github.com/dafeigediaozhatian/A3C-Net
cd A3C-Net
```

- Install dependencies(pytorch, scikit-learn, opencv-python, pandas. Recommend to use Anaconda.)
```python
# Create a new conda environment
conda create -n A3C-Net python==3.8
conda activate A3C-Net

# Install other packages
conda env create -f requirements.yml
```


## pretrain model
- AVA dataset and CADB dataset model
  - Download the aesthetic pretrain model and put it in ./dataset/aesthetic_model.([download_link](https://pan.baidu.com/s/1F6Imkj7bFkIiKot4WgSxUw?pwd=u9un), 提取码：u9un) 
  - Download the composition pretrain model and put it in ./dataset/composition_model. ([download_link](https://pan.baidu.com/s/16Idk-C1ItPSJzueuAFPYZw?pwd=p5uq), 提取码：p5uq)
  - Download the detection pretrain model and put it in ./checkpoints/ ([download_link](https://pan.baidu.com/s/18V-IQzRV579kDRmJJefj5A?pwd=o3bs) ,提取码：o3bs)
  The directory structure should be like:
```
|--checkpoints
   |--faster_rcnn_r50_caffe_c4_mstrain_1x_coco_20220316_150527-db276fed.pth
|--dataset
   |--aesthetic_model
      |--aesthetic-resnet50-model-epoch-10.pkl
   |--composition_model
      |--composition-resnet50-model-epoch-10.pkl
```

## Training and test
Traning scripts for two datasets can be found in ##train_cropping.py##. The dataroot argument should be modified to path_to_<dataset_name>. Run the follwing command for training:
```python
# Training on FCDB and FLMS
python train_cropping.py

# Test the result
python test_cropping.py
```



## Citation
```
@inproceedings{shi2023aesthetic,
  title={Aesthetic and Composition Collaboration Driven Image Cropping},
  author={Tengfei Shi, Chenglizhao Chen, Yuanbo He, Wenfeng Song, Aiming Hao},
  conference={ICASSP},
  year={2023}
}
```
