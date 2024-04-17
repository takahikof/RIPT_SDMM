# RIPT + SDMM
## Introduction
This repository provides the official codes (PyTorch implementation) for the paper _**["Self-supervised Learning of Rotation-invariant 3D Point Set Features using Transformer and its Self-distillation"](https://arxiv.org/abs/2308.04725)**. The paper has been accepted as a paper of the Computer Vision and Image Understanding (CVIU) journal. 
```
Takahiko Furuya, Zhoujie Chen, Ryutarou Ohbuchi, Zhenzhong Kuang, Self-supervised Learning of Rotation-invariant 3D Point Set Features using Transformer and its Self-distillation, Computer Vision and Image Understanding (CVIU), to appear.
```
## Pre-requisites
Our code has been tested on Ubuntu 22.04. We highly recommend using the Docker container "nvcr.io/nvidia/pytorch:21.07-py3", which is provided by [Nvidia NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags). 
After launching the Docker container, run the following shell script to install the prerequisite libraries.
```
./prepare.sh
```
## Datasets
The 3D point cloud data used in our study can be downloaded from Google Drive.<br>
Save the downloaded h5 files in the "dataset" directory.
| Dataset | URL |
| ---- | ---- |
| ModelNet10 | [training_set](https://drive.google.com/file/d/1K0poxAMOX7SvRJFaGU_KXcuc3H39yon_/view?usp=sharing)&emsp;[testing_set](https://drive.google.com/file/d/1SIku6h9ta6jIcBON2nVnEXk2TDp9DFG7/view?usp=sharing) |
| ModelNet40 | [training_set](https://drive.google.com/file/d/1KnLZIklZ0MhqHo84NOAlMOERIcen3lXx/view?usp=sharing)&emsp;[testing_set](https://drive.google.com/file/d/1q_MQkCkLZkvm1f86AjvCCI6ChJ85yPoD/view?usp=sharing) |
| ShareNetCore55 | [training_set](https://drive.google.com/file/d/1Ssuk32p1Dl5XvoJ3023ZQc6pYTS64xMc/view?usp=sharing)&emsp;[testing_set](https://drive.google.com/file/d/1Vq1rJOU23GY_A5df7SDIffIFBAGgTnpx/view?usp=sharing) |
| ScanObjectNN OBJ_ONLY | [training_set](https://drive.google.com/file/d/1xH6hMb8YNGS0lEoWKFWe8jUOVRgRrEqZ/view?usp=sharing)&emsp;[testing_set](https://drive.google.com/file/d/1M7mBEt3AA92W_Ok-H0tFXsBDt76uTjzo/view?usp=sharing) |
| ScanObjectNN OBJ_BG | [training_set](https://drive.google.com/file/d/1DTfgJZ-AgyLp5-deUTjc1QEez0ZzRVtQ/view?usp=sharing)&emsp;[testing_set](https://drive.google.com/file/d/1B8WGxb5gjNYe6KoVs7AEHkkVp4W6pcLy/view?usp=sharing) |
| ScanObjectNN PB_T50_RS | [training_set](https://drive.google.com/file/d/1tb6g1nvVs_dud6KhJ3wdFU4qgjZsc1Kb/view?usp=sharing)&emsp;[testing_set](https://drive.google.com/file/d/12_16AVajG5IBpxyDtCyDf-vMKx1jnSYR/view?usp=sharing) |
## Training and evaluation
In the shell script file "Run.sh", you can specify the conditions of the experiment such as dataset, rotation of the 3D point clouds, hyperparameters etc. By executing "Run.sh", the proposed DNN (i.e., RIPT) is trained by using the proposed self-supervised self-distillation algorithm (i.e., SDMM). Every five epochs, the retrieval accuracy on the testing dataset is evaluated. The retrieval accuracy is measured in macro-averaged Mean Average Precision (MAP).
```
./Run.sh
```
