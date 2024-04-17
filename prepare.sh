#!/bin/bash
# Assumes using the docker container for Pytorch "nvcr.io/nvidia/pytorch:21.07-py3"
# which can be installed by executing the following command:
# docker pull nvcr.io/nvidia/pytorch:21.07-py3

pip install einops h5py timm

# For Pytorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2"
