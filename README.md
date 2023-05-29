# Joint Probability Distribution Regression for Image Cropping（ACPD-Net）

This repository contains a pytorch implementation of the paper "Joint Probability Distribution Regression for Image Cropping"(Subject to ICIP2023)

In this paper, we present an **A**esthetic and **C**omposition Joing **P**robability **D**istribution **Net**work~(ACPD-Net) to explicitly investigate the collaboration of image aesthetics and image composition for the cropping task in an end-to-end manner.

## Pipeline
![image](https://github.com/flyingbird93/ACPD-Net/assets/16755407/6776cd77-1f34-4a25-9299-84327424c543)


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
git clone https://github.com/dafeigediaozhatian/ACPD-Net
cd ACPD-Net
```

- Install dependencies(pytorch, scikit-learn, opencv-python, pandas. Recommend to use Anaconda.)
```python
# Create a new conda environment
conda create -n ACPD-Net python==3.8
conda activate ACPD-Net

# Install other packages
conda env create -f requirements.yml
```


## pretrain model
- AVA dataset and CADB dataset model
  - Download the aesthetic pretrain model and put it in ./dataset/aesthetic_model.([download_link](https://pan.baidu.com/s/1F6Imkj7bFkIiKot4WgSxUw?pwd=u9un), passward：u9un) 
  - Download the composition pretrain model and put it in ./dataset/composition_model. ([download_link](https://pan.baidu.com/s/16Idk-C1ItPSJzueuAFPYZw?pwd=p5uq), passward：p5uq)
  - Download the detection pretrain model and put it in ./checkpoints/ ([download_link](https://pan.baidu.com/s/18V-IQzRV579kDRmJJefj5A?pwd=o3bs) ,passward：o3bs)
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
Traning scripts for two datasets can be found in ```train_cropping.py```. The dataroot argument should be modified to path_to_<dataset_name>. Run the follwing command for training:
```python
# Training on FCDB and FLMS
python train_cropping.py

# Test the result
python test_cropping.py
```

## The Ablation study on the aesthetics threshold for probability θ
![image](https://github.com/flyingbird93/ACPD-Net/assets/16755407/7db91a94-53eb-4549-91b9-085a4f31175a)

## The model size and inference speed of SOTA methods
![image](https://github.com/flyingbird93/ACPD-Net/assets/16755407/de1dae79-e016-4979-b1b4-8c32390e3172)

## Citation
```
@inproceedings{shi2023aesthetic,
  title={Joint Probability Distribution Regression for Image Cropping},
  author={Tengfei Shi, Chenglizhao Chen, Yuanbo He, Wenfeng Song, Aiming Hao},
  conference={ICIP2023},
  year={2023}
}
```
