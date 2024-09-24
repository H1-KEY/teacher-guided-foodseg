# Teacher-Guided Fine-Grained Food Segmentation (Yolo + Segment Anything)
Food Dataset을 활용한 Sementaic Segmentation 모델 제안
<br>[Dataset Link](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022/dataset_files)<br>
제안하는 구조: **Teacher-Guided Fine-Grained Semantic Segmentation**
***
## Structure
Train, Validation, Test dataset을 다운로드 후, datasets/train/images, datasets/val/images, datasets/test/images에 각각 압축해제
```
repo
  |——datasets
        |train
                |——images
                        |——006316.jpg
                        |——006331.jpg
                        |——....
                |——annotations.json
        |val
                |——images
                        |——008082.jpg
                        |——008135.jpg
                        |——....
                |——annotations.json
        |test
                |——images
                        |——007845.jpg
                        |——008615.jpg
                        |——....
  |——food-seg-modified
        |yolov8x-seg-modified
                |——weights
                        |——best.pt
                |——....
  |——yolov8
        |——fix_seed.py
        |——translation.py
        |——yolov8_10fold_train_1.py
        |——yolov8_10fold_train_2.py
        |——yolov8_10fold_aug_train_1.py
        |——yolov8_10fold_aug_train_2.py
        |——yolov8_single_inference.py
  category_info.txt
  color_list.npy
  ensemble.py
  food-seg.yaml
  pipeline.py
  sam_vit_h_4b8939.pth
  segment.py
  train_yolov8_modified.py
  example.ipynb
  yolov8_modified.yaml
```
***

## Development Environment
* Ubuntu 20.04
* AMD Ryzen Threadripper PRO 5995WX (64 cores, 128 threads)
* RTX 4090 2EA
* CUDA 11.8
* cuDNN 8.8.0

### Anaconda environment
[![Python 3.9.18](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-385/)
#### 1. Create anaconda environment & activate
```shell
conda create -n env_name python=3.9.18
```
```shell
conda activate env_name
```
#### 2. Install PyTorch following your hardwares. [ref](https://pytorch.org/get-started/locally/) In this repo, we have installed PyTorch 2.0 with CUDA 11.8
```shell
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --find-links https://download.pytorch.org/whl/cu118
```
#### 3. Install customized yolov8
```shell
cd yolov8
```
```shell
pip install -e .
```
#### 4. Install Segment Anything Model (SAM)
```shell
pip install git+https://github.com/facebookresearch/segment-anything.git
```

#### 5. Download SAM weight (ViT-H SAM)
```shell
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
#### 6. Install Jupyter
```shell
pip install jupyter notebook
```
***
## Methods
### 1. Format Transformation
Converts COCO format to YOLO format
```shell
python coco2yolo.py
```
### 2. Data Augmentation
Data Augmentation is pre-defined on the yolov8_modified.yaml and inside the modified yolov8

### 3. Fine-tune yolov8
We fine-tuned the largest yolov8 segmentation model (yolov8x-seg) with two GPUs.
```shell
NCCL_P2P_DISABLE=1 python train_yolov8_modified.py
```
### 4. Inference for whole test dataset
```shell
python pipeline.py
```
## Demo
The demo is included in example.ipynb

## References
* Kirillov, Alexander, et al. "Segment anything." arXiv preprint arXiv:2304.02643 (2023). [paper](https://arxiv.org/pdf/2304.02643.pdf) | [github](https://github.com/facebookresearch/segment-anything)
* Jocher, G., Chaurasia, A., & Qiu, J. (2023). YOLO by Ultralytics (Version 8.0.0) [Computer software]. [github](https://github.com/ultralytics/ultralytics)