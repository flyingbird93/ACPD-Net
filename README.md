# Joint Probability Distribution Regression for Image Cropping（ACPD-Net）

This repository contains a pytorch implementation of the paper "Joint Probability Distribution Regression for Image Cropping"(Subject to ICIP2023)

In this paper, we present an **A**esthetic and **C**omposition Joing **P**robability **D**istribution **Net**work~(ACPD-Net) to explicitly investigate the collaboration of image aesthetics and image composition for the cropping task in an end-to-end manner.

## Motivation
![image](https://github.com/flyingbird93/ACPD-Net/assets/16755407/0d963026-7626-4998-89e7-ad5832fbbbaa)


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


## Pretrain model
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


## The Ablation study on the aesthetics threshold θ
The ablation study on the parameter θ in Eq. 2, and we select θ=0.05 as the optical choice, where the average intersection-over-union (IoU) and the average boundary
displacement error (BDE) as metrics.

![image](https://github.com/flyingbird93/ACPD-Net/assets/16755407/ddc13a0b-070c-4cb4-8044-69c4497a6166)




## The model size and inference speed of SOTA methods
The model parameters of our method are not the least, and its inference speed is not the fastest, but our model can meet real-time requirements (FPS > 60). All tests were conducted on Nvidia GTX 3090."

![image](https://github.com/flyingbird93/ACPD-Net/assets/16755407/de1dae79-e016-4979-b1b4-8c32390e3172)



## Visual results
Qualitative comparison and user study of different methods. Compared with other methods, our method can obtain better visually cropping results close to GT. Last row with the user study results show that most users favor our method.

![image](https://github.com/flyingbird93/ACPD-Net/assets/16755407/9da3a939-5114-419b-bd49-afa63d5b6d6a)


## Citation
```
@inproceedings{shi2023aesthetic,
  title={Joint Probability Distribution Regression for Image Cropping},
  author={Tengfei Shi, Chenglizhao Chen, Yuanbo He, Wenfeng Song, Aiming Hao},
  conference={ICIP2023},
  year={2023}
}
```
