# Aesthetic and Composition Collaboration Driven Image Cropping（A3C-Net）

This repository contains a pytorch implementation of the paper "Aesthetic and Composition Collaboration Driven Image Cropping"(Subject to ICASSP2023)

In this paper, we present an \textbf{A}esthetic \textbf{C}omposition \textbf{C}ollaboration \textbf{C}ropping \textbf{Net}work~(A3C-Net) to explicitly investigate the collaboration of ``image aesthetics'' and ``image composition'' for the cropping task in an end-to-end manner.

## Pipeline
![pipeline](https://img-blog.csdnimg.cn/0005050d73b4459284644d4d7c232379.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaGVsbG93b3JsZF9GbHk=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


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
  - Download the aesthetic pretrain model and put it in ./dataset/aesthetic_model.(download_link) 
  - Download the composition pretrain model and put it in ./dataset/composition_model. (download_link)
  The directory structure should be like:
	```
  checkpoints
    faster_rcnn_r50_caffe_c4_mstrain_1x_coco_20220316_150527-db276fed.pth (download_link)
	dataset
		aesthetic_model
      aesthetic-resnet50-model-epoch-10.pkl
    composition_model
      composition-resnet50-model-epoch-10.pkl
	```

## Training and test
Traning scripts for two datasets can be found in #train_cropping.py#. The dataroot argument should be modified to path_to_<dataset_name>. Run the follwing command for training:
```python
# Training on FCDB and FLMS
python train_cropping.py

# Test the result
python test_cropping.py
```



## Citation
```
@inproceedings{shi2023aesthetic,
  title={Multiple image joint learning for image aesthetic assessment},
  author={Tengfei Shi, Chenglizhao Chen, Yuanbo He, Wenfeng Song, Aiming Hao},
  booktitle={ICASSP},
  year={2023}
}
```
