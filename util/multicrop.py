# -*- coding: utf-8 -*-
import numpy as np
import copy
import sys
import torch
from sklearn.neighbors import KDTree
from scipy.stats import truncnorm
import transform_3d as t3d

def multicrop_collate_fn( batch ) : # This function is used to overwrite the "collate_fn" function of dataloader
    # Since multi-cropped point sets may have different numbers of 3D points,
    # the dataloader should return a minibatch as a list (not a tensor).
    B = len( batch )
    plist, llist = list(zip(*batch))
    n_crops = len( plist[ 0 ] )

    # initialize lists
    pointsets= []
    for i in range( n_crops ) :
        pointsets.append( [] ) # pointsets[i] will be filled with i-th crops
                               # That is, pointsets[0] is a list of first global crops
                               # pointsets[1] is a list of second global crops
                               # pointsets[2] or later are lists of local crops
    labels = []

    for i in range( B ) : # for each 3D shape
        for j in range( n_crops ) :
            pointsets[ j ].append( plist[ i ][ j ] )
        labels.append( llist[ i ][0] )

    for i in range( n_crops ) :
        pointsets[ i ] = torch.tensor( np.asarray( pointsets[ i ] ) )

    labels = torch.tensor( labels )
    return pointsets, labels

def crop_pointset( pos, ori, query, scale_min, scale_max, tree, p ) :
    # randomly choose scale of cropped 3D point set
    scale = np.random.uniform( low=scale_min, high=scale_max )

    # convert the scale to the number of neighboring points
    knn = int( pos.shape[0] * scale )

    # create cropped point set
    _, knn_idx = tree.query( np.reshape( pos[ query ], (1,3) ), k=knn )
    knn_idx = knn_idx[0]
    crop = np.hstack( [ pos[ knn_idx ], ori[ knn_idx ] ] )

    # duplicate/subsample points so that the cropped 3D point set has p points
    if( p > crop.shape[0] ) :
        n_duplicate = p - crop.shape[0]
        idxs = np.arange( crop.shape[0] )
        np.random.shuffle( idxs )
        idxs = np.tile( idxs, int( np.ceil( n_duplicate / crop.shape[0] ) ) )
        idxs = idxs[ : n_duplicate ]
        crop = np.vstack( [ crop, crop[ idxs ] ] )
    elif( p < crop.shape[0] ) :
        idxs = np.arange( crop.shape[0] )
        np.random.shuffle( idxs )
        idxs = idxs[ : p ]
        crop = crop[ idxs ]

    return crop

def multicrop_pointset( input, args ) :
    # input : An oriented 3D point set [P, 6]
    #         This point set is assumed to be fitted in a unit sphere
    # output : A list of multi-cropped and augmented 3D point sets
    #         The number of 3D point differs depending on global crop or local crop

    P, _ = input.shape
    P_global = P
    P_local = P // 2

    crops = []

    pos = input[ :, 0:3 ]
    ori = input[ :, 3:6 ]
    tree = KDTree( pos )

    # randomly choose center points of cropped 3D points
    pts_g = np.random.randint( low=0, high=P, size=(2) )
    pts_l = np.random.randint( low=0, high=P, size=(args.mc_num_lcrop) )

    # create global crops
    for i in range( 2 ) : # number of global crops is fixed at 2
        if( np.random.rand() < 0.2 ) : # sometimes use all 3D points
            crop = input
        else :
            crop = crop_pointset( pos, ori, pts_g[ i ],
                                  args.mc_gscale_min, args.mc_gscale_max,
                                  tree, P_global )
        crops.append( copy.deepcopy( crop ) )

    # create local crops
    for i in range( args.mc_num_lcrop ) :
        crop = crop_pointset( pos, ori, pts_l[ i ],
                              args.mc_lscale_min, args.mc_lscale_max,
                              tree, P_local )
        crops.append( copy.deepcopy( crop ) )

    # # normalize scale and position of each crop
    # for i in range( len( crops ) ) :
    #     crops[ i ] = t3d.normalize_data( crops[ i ] )

    # randomly augment each crop
    for i in range( len( crops ) ) :
        if( np.random.rand() < 0.8 ) : # augmentation is usually applied

            c = crops[ i ]
            pos = c[ :, 0:3 ]
            ori = c[ :, 3:6 ]

            # anisotropic scaling (orthogonal axes for scaling are chosen randomly)
            S = t3d.generate_randomscale_matrix( args.da_scale_min, args.da_scale_max )
            R = t3d.generate_randomrot_matrix()
            pos = np.matmul( pos, R )
            pos = np.matmul( pos, S )
            pos = np.matmul( pos, R.transpose() )

            # translation
            # does not apply translation since it is canceled (normalized) by the tokenizer

            # jittering
            lowerbound, upperbound = - args.da_jitter_clip, args.da_jitter_clip # clip too large displacements
            mean = 0.0
            std = args.da_jitter_std
            a, b = ( lowerbound - mean ) / std, ( upperbound - mean ) / std # standardize (ref. documentation of scipy.stats.truncnorm)
            noise = truncnorm.rvs( a, b, loc=mean, scale=std, size=pos.shape )
            pos = pos + noise

            crops[ i ] = np.hstack( [ pos, ori ] )

    return crops
