# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from distutils.util import strtobool

sys.path.append( "util" );
sys.path.append( "model/loss" );
sys.path.append( "model/evaluation" );
sys.path.append( "model/ript" )
from dataset import PointCloudDataset
from ript import model_RIPT
from multicrop import multicrop_collate_fn
from MultiCropWrapper import MultiCropWrapper
from pointmixup import point_mixup
from DINO import DinoHead, DinoLoss
import dino_utils as du
from timm.scheduler import CosineLRScheduler
import retrieval

torch.set_printoptions( edgeitems=1000 )

def count_parameters( model ):
    return sum( param.numel() for param in model.parameters() if param.requires_grad )

def train_one_epoch( args, data_loader, student, teacher, loss_func, opt, momentum_schedule, epoch ) :

    avg_loss = 0.0
    count = 0

    for pointsets, _ in data_loader : # for each minibatch

        # transfer the minibatch data to GPU
        pointsets = [ pc.to( args.device ) for pc in pointsets ] # a minibatch of multi-cropped point sets (contains global views and local views)

        # generate mixed views
        # (when args.mixup_coeff is 0.0, mixed views are fed into the student DNN, but their features are not used in loss computation)
        mixup_pointsets, mixup_lambdas = point_mixup( pointsets[0], args.mixup_alpha, args.mixup_mode ) # the minibatch of the first global view is used for point mixup
        pointsets.append( mixup_pointsets )

        opt.zero_grad() # clear gradients

        # extract features
        preds_stu, feats_stu = student( pointsets ) # all the views (global views, local views, and mixed views) pass through the student
        preds_tea, feats_tea = teacher( pointsets[:2] ) # only the two global views pass through the teacher

        # compute loss
        ls = loss_func( preds_stu, preds_tea, mixup_lambdas )
        if not math.isfinite( ls.item() ):
            print("Loss is infinity, stopping training".format(ls.item()), force=True)
            quit()
        avg_loss += ls
        count += 1

        # update parameters
        # update student parameters
        ls.backward()
        du.cancel_gradients_last_layer( epoch, student, args.dino_epochs_freeze_last_layer )
        opt.step()

        # update teacher parameters by EMA (exponential moving average)
        with torch.no_grad():
            m = momentum_schedule[ epoch ]  # momentum parameter
            for param_q, param_k in zip( student.parameters(), teacher.parameters() ):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    avg_loss = avg_loss.to('cpu').detach().numpy().copy() / count
    return avg_loss

def test_one_epoch( args, test_data_loader, student, teacher, epoch ) :

    def extract_features_from_one_batch( args, pointsets, model ) :
        _, feats = model( pointsets )
        return feats

    # extract features from testing dataset
    feat_test = []
    label_test = []
    count = 0
    for pointsets, labels in test_data_loader : # for each minibatch
        pointsets, labels = pointsets.to( args.device ), labels.to( args.device ).squeeze()
        feats = extract_features_from_one_batch( args, pointsets, teacher )
        feat_test.append( feats )
        label_test.append( labels )
        count = count + 1

    feat_test = torch.cat( feat_test, dim=0 )
    label_test = torch.cat( label_test, dim=0 )
    label_test = label_test.cpu().numpy()
    feat_test = feat_test.detach().cpu().numpy()

    mnn, micro_map, macro_map, rpcurve = retrieval.retrieval( feat_test, label_test )

    return macro_map


def train_and_test( args ) :

    # Prepare four things essential for DNN training
    # i.e., dataset, DNN architecture, loss function, and optimizer

    # Create dataset loaders
    # training dataset with multi-crop data augmentation (used for training)
    train_dataset = PointCloudDataset( args=args, path_dataset=args.dataset, partition='train',
                                       num_points=args.num_points, data_aug=True, rotation=args.rot_train )
    args.num_train_data = train_dataset.__len__() # Number of training 3D shapes
    train_loader = DataLoader( train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=multicrop_collate_fn )

    # testing dataset without data augmentation (used for evaluation)
    test_loader = DataLoader( PointCloudDataset( args=args, path_dataset=args.dataset, partition='test',
                                                 num_points=args.num_points, data_aug=False, rotation=args.rot_test ),
                              num_workers=4, batch_size=args.batch_size, shuffle=False, drop_last=False )

    # Create DNNs
    encoder_stu = model_RIPT( args )
    encoder_tea = model_RIPT( args )

    print( "# of learnable parameters of student: " + str( count_parameters( encoder_stu ) ) )
    print( "# of learnable parameters of teacher: " + str( count_parameters( encoder_tea ) ) )
    student = MultiCropWrapper( args, encoder_stu, DinoHead( args, "student", args.num_dim_embedding ) )
    teacher = MultiCropWrapper( args, encoder_tea, DinoHead( args, "teacher", args.num_dim_embedding ) )
    student = nn.DataParallel( student.to( args.device ) )
    teacher = nn.DataParallel( teacher.to( args.device ) )

    # teacher and student start with the same weights
    teacher.load_state_dict( student.state_dict() )
    # disable gradient computation of the teacher
    for p in teacher.parameters():
        p.requires_grad = False

    # Create loss
    loss_func = DinoLoss( args ).to( args.device )

    # Create optimizer
    params = du.get_params_groups( student )
    if( args.optimizer == "adam" ) :
        opt = optim.Adam( params, lr=args.init_lr )
        scheduler = CosineLRScheduler( opt, t_initial=args.epochs, lr_min=args.init_lr*0.2,
                                       warmup_t=args.warmup_epochs, warmup_lr_init=args.init_lr*0.2, warmup_prefix=True )
    elif( args.optimizer == "adamw" ) :
        opt = optim.AdamW( params, lr=args.init_lr )
        scheduler = CosineLRScheduler( opt, t_initial=args.epochs, lr_min=args.init_lr*0.2,
                                       warmup_t=args.warmup_epochs, warmup_lr_init=args.init_lr*0.2, warmup_prefix=True )
    else :
        print( "error: invalid optimizer:" + args.optimizer )
        quit()

    # Coefficient for momentum update is increased from "momentum_teacher" to 1.0 during training with a cosine schedule
    momentum_schedule = du.cosine_scheduler( args.dino_momentum_teacher, 1.0, args.epochs )

    # Iterate training and evaluation
    best_macro_map = 0.0
    eval_interval = 5
    for epoch in range( args.epochs ) : # for each epoch

        print( "[*] " + str( epoch ) + "-th epoch" )

        # execute evaluation
        if( epoch % eval_interval == 0 ) :
            print( "evaluating..." )
            student.eval()
            teacher.eval()
            with torch.no_grad() :
                macro_map = test_one_epoch( args, test_loader, student, teacher, epoch )

            if( best_macro_map < macro_map ) :
                best_macro_map = macro_map

            print( "MacroMAP: " + str( np.round( macro_map, decimals=4 ) ) )
            print( "current best MacroMAP: " + str( np.round( best_macro_map, decimals=4 ) ) )

        # execute training
        print( "training..." )
        start = time.perf_counter()
        student.train()
        teacher.train()
        avg_loss = train_one_epoch( args, train_loader, student, teacher,
                                    loss_func, opt, momentum_schedule, epoch )
        print( '[Loss] train: ' + str( np.round( avg_loss, decimals=4 ) ) ) # average loss value during this epoch

        print( str( np.round( time.perf_counter() - start, decimals=4 ) ) + " sec.")

        scheduler.step( epoch+1 ) # update learning rate

    # execute evaluation
    print( "evaluating..." )
    student.eval()
    teacher.eval()
    with torch.no_grad() :
        macro_map = test_one_epoch( args, test_loader, student, teacher, epoch )

    if( best_macro_map < macro_map ) :
        best_macro_map = macro_map

    print( "MacroMAP: " + str( np.round( macro_map, decimals=4 ) ) )
    print( "Best MacroMAP: " + str( np.round( best_macro_map, decimals=4 ) ) )


if( __name__ == "__main__" ) :

    # read command-line arguments
    parser = argparse.ArgumentParser(description='Demo of 3D Point Set Deep Learning')
    parser.add_argument('--exp_name', type=str, default='exp', help='ID name of the experiment')
    parser.add_argument('--savedir', type=str, default='out', help='Output directory')
    parser.add_argument('--dataset', type=str, default='', help='dataset of 3D point sets')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points per 3D shape to be used as input to the DNN')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for DNN training')
    parser.add_argument('--use_gpu', type=strtobool, default=True, help='wether to use GPU(s)')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimization algorithm')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='warmup')

    # arguments for rotation scenario ( nr (not rotate) or so3 (all rotations about the origin of 3D space) )
    parser.add_argument('--rot_train', type=str, default='so3', help='Rotation mode for training. nr: not rotate, so3: rotate randomly')
    parser.add_argument('--rot_test', type=str, default='so3', help='Rotation mode for evaluation. nr: not rotate, so3: rotate randomly')

    # arguments for data augmentation
    parser.add_argument('--da_scale_max', type=float, default=1.5, help='[data aug] parameter for anisotropic scaling')
    parser.add_argument('--da_scale_min', type=float, default=0.67, help='[data aug] parameter for anisotropic scaling')
    parser.add_argument('--da_jitter_std', type=float, default=0.01, help='[data aug] parameter for jittering')
    parser.add_argument('--da_jitter_clip', type=float, default=0.05, help='[data aug] parameter for jittering')

    # arguments for encoder
    parser.add_argument('--num_dim_embedding', type=int, default=256, help='Number of dimensions for feature embedding')

    # arguments for multi-crop
    parser.add_argument('--mc_num_lcrop', type=int, default=4, help='[multi-crop] number of local crops')
    parser.add_argument('--mc_gscale_max', type=float, default=1.0, help='[multi-crop] maximum scale of global crops')
    parser.add_argument('--mc_gscale_min', type=float, default=0.7, help='[multi-crop] minimum scale of global crops')
    parser.add_argument('--mc_lscale_max', type=float, default=0.7, help='[multi-crop] maximum scale of local crops')
    parser.add_argument('--mc_lscale_min', type=float, default=0.2, help='[multi-crop] minimum scale of local crops')

    # arguments for point mixup
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='[mixup]')
    parser.add_argument('--mixup_coeff', type=float, default=0.5, help='[mixup]')
    parser.add_argument('--mixup_mode', type=str, default="K", help='[mixup]')

    # arguments for DINO training framework
    # please refer to the original code of DINO: https://github.com/facebookresearch/dino/blob/main/main_dino.py
    parser.add_argument('--dino_out_dim', type=int, default=1024, help='[dino]')
    parser.add_argument('--dino_norm_student_last_layer', type=strtobool, default=False, help='[dino]')
    parser.add_argument('--dino_use_bn_in_head', type=strtobool, default=False, help='[dino]')
    parser.add_argument('--dino_momentum_teacher', type=float, default=0.996, help='[dino]')
    parser.add_argument('--dino_teacher_temp', type=float, default=0.04, help='[dino]')
    parser.add_argument('--dino_student_temp', type=float, default=0.1, help='[dino]')
    parser.add_argument('--dino_head_layers', type=int, default=3, help='[dino]')
    parser.add_argument('--dino_hidden_dim', type=int, default=1024, help='[dino]')
    parser.add_argument('--dino_bottleneck_dim', type=int, default=128, help='[dino]')
    parser.add_argument('--dino_epochs_freeze_last_layer', type=int, default=1, help='[dino]')

    # arguments for RIPT (tokenizer part)
    parser.add_argument('--num_tokens', type=int, default=256, help='Number of tokens per 3D shape')
    parser.add_argument('--token_scale', type=float, default=0.5, help='Determines number of 3D points within each token')
    parser.add_argument('--token_point_sampling_interval', type=int, default=2, help='Determines number of 3D points within each token')
    parser.add_argument('--num_dim_token', type=int, default=64, help='Number of dimensions for each token feature')
    parser.add_argument('--tokenizer_arch_encoder', type=str, default='pointnet', help='Archtecture of encoder in Tokenizer')
    parser.add_argument('--tokenizer_lrf', type=str, default='shot', help='Method for computing local reference frames')
    parser.add_argument('--pod_num_bins', type=int, default=4, help='Grid size for POD feature')

    # arguments for RIPT (transformer part)
    parser.add_argument('--trsfm_num_blocks', type=int, default=1, help='[transformer] Number of transformer blocks')
    parser.add_argument('--trsfm_knn', type=int, default=16, help='[transformer] Number of local neighbors for which self-attention is computed')
    parser.add_argument('--trsfm_attention_mode', type=str, default="", help='[transformer] Mode for self-attention')
    parser.add_argument('--trsfm_subsample_mode', type=str, default="", help='[transformer] Mode for point subsampling')

    args = parser.parse_args()
    print( str( args ) )

    # Determine the device used for training/evaluation
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    args.device = torch.device( "cuda" if args.use_gpu else "cpu" )
    if( args.use_gpu ) :
        print( 'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices' )
    else:
        print( 'Using CPU' )

    # train and evaluate DNN
    train_and_test( args )

    quit()
