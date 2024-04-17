#!/bin/bash

gpuid=0 # Specifies the ID of a GPU used for training/evaluation

# Specify dataset
dataset="./dataset/mn10" # ModelNet10
# dataset="./dataset/mn40" # ModelNet40
# dataset="./dataset/sn55" # ShapeNetCore55
# dataset="./dataset/sonn_withoutbg" # ScanObjectNN OBJ_ONLY
# dataset="./dataset/sonn_withbg" # ScanObjectNN OBJ_BG
# dataset="./dataset/sonn_pbt50rs" # ScanObjectNN PB_T50_RS (hardest)

# Specify rotation settings
rot_train="nr" # rotation setting for training shapes
rot_test="so3" # rotation setting for testing shapes

num_points=1024 # Number of 3D points per shape
batch_size_train=32
num_epochs_train=200 # Training epochs
num_dim_embedding=256 # Number of dimensions for latent feature vector
optimizer="adam" # Optimizer
init_lr=5e-4 # Initial learning rate
warmup_epochs=20 # Number of epochs for learning rate warmup

# Hyperparameters for data augmentation
da_scale_max=1.5
da_scale_min=`python -c "print(1.0/$da_scale_max)"`
da_jitter_std=0.01
da_jitter_clip=0.05

# Hyperparameters for multi-crop
mc_num_lcrop=2
mc_gscale_max=1.0
mc_gscale_min=0.6
mc_lscale_max=0.6
mc_lscale_min=0.4

# Hyperparameters for point mixup
mixup_alpha=1.0
mixup_coeff=0.5
mixup_mode="K"

# Hyperparameters for DINO training
dino_norm_student_last_layer=False
dino_use_bn_in_head=False
dino_momentum_teacher=0.996
dino_head_layers=3
dino_hidden_dim=1024
dino_bottleneck_dim=128
dino_out_dim=1024
dino_teacher_temp=0.1
dino_student_temp=0.4

# Hyperparameters for RIPT (tokenizer part)
num_tokens=256
token_scale=1.0
token_point_sampling_interval=4
num_dim_token=512
tokenizer_arch_encoder="pod"
tokenizer_lrf="em"
pod_num_bins=6

# Hyperparameters for RIPT (transformer part)
trsfm_num_blocks=2
trsfm_knn=4 # 4 -> 8 -> 16
trsfm_attention_mode="vector_sa"
trsfm_subsample_mode="fps"

# Train DNN
py_code="Train.py"
CUDA_VISIBLE_DEVICES=$gpuid python -u $py_code --dataset=$dataset --num_points=$num_points --batch_size=$batch_size_train --epochs=$num_epochs_train \
                                               --optimizer=$optimizer --init_lr $init_lr --warmup_epochs $warmup_epochs\
                                               --rot_train $rot_train --rot_test $rot_test \
                                               --num_dim_embedding $num_dim_embedding \
                                               --da_scale_max $da_scale_max --da_scale_min $da_scale_min --da_jitter_std $da_jitter_std --da_jitter_clip $da_jitter_clip \
                                               --mc_num_lcrop $mc_num_lcrop --mc_gscale_max $mc_gscale_max --mc_gscale_min $mc_gscale_min --mc_lscale_max $mc_lscale_max --mc_lscale_min $mc_lscale_min \
                                               --dino_out_dim $dino_out_dim --dino_norm_student_last_layer $dino_norm_student_last_layer --dino_use_bn_in_head $dino_use_bn_in_head \
                                               --dino_momentum_teacher $dino_momentum_teacher --dino_teacher_temp $dino_teacher_temp --dino_student_temp $dino_student_temp \
                                               --dino_head_layers $dino_head_layers --dino_hidden_dim $dino_hidden_dim --dino_bottleneck_dim $dino_bottleneck_dim \
                                               --mixup_alpha $mixup_alpha --mixup_coeff $mixup_coeff --mixup_mode $mixup_mode \
                                               --num_tokens $num_tokens --token_scale $token_scale --token_point_sampling_interval $token_point_sampling_interval --num_dim_token $num_dim_token \
                                               --tokenizer_arch_encoder $tokenizer_arch_encoder --tokenizer_lrf $tokenizer_lrf \
                                               --pod_num_bins $pod_num_bins \
                                               --trsfm_num_blocks $trsfm_num_blocks --trsfm_knn $trsfm_knn \
                                               --trsfm_attention_mode $trsfm_attention_mode --trsfm_subsample_mode $trsfm_subsample_mode

exit
