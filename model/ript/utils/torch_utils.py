import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat, rearrange


def normalize_position_and_scale( input ) :
    # Normalize position and scale of the input 3D point sets.
    # input : an input batch of 3D point sets [B, P, 3]
    mu = torch.mean( input, dim=1, keepdim=True ) # gravity centers [B, 1, 3]
    output = input - mu # [ B, P, 3 ]
    norm = torch.linalg.vector_norm( output, ord=2, dim=2, keepdim=True ) # [B, P, 1]
    max_norm, _ = torch.max( norm, dim=1, keepdim=True ) # [B, 1, 1]
    output = torch.div( output, max_norm )
    return output

def nn_search( Q_pts, T_pts, knn ) :
    # Q_pts : an input batch of query 3D points [B, P1, 3]
    # T_pts : an input batch of target 3D point sets [B, P2, 3]

    # The knn_points function in Pytorch3D is slow when knn is very large
    # _, nn_idx, _ = knn_points( Q_pts, T_pts, K=knn ) # [B, P1, K]

    distmats = torch.cdist( Q_pts, T_pts )
    dists, nn_idx = torch.topk( distmats, knn, dim=2, largest=False, sorted=True )
    return nn_idx


def compute_angle_to_refvec( input, refvec=[0, 1] ) :
    # input: 2D points [P, 2]
    # refvec: a reference vector
    # For each 2D point in input, computes an angle between the 2D point and the reference vector refvec.
    # Each angle takes a value in the range [0, 2pi)
    rv = torch.FloatTensor( refvec ).unsqueeze(0).to( input.device ) # a reference vector
    cross_z = ( input[:,0] * rv[0,1] - input[:,1] * rv[0,0] ).unsqueeze(1) # we need to compute only the z element of a cross product vector
    norm = torch.linalg.norm( input, dim=1, keepdims=True )
    cos = torch.matmul( input, torch.t( rv ) ) / norm
    cos = torch.nan_to_num( cos, nan=0.0 ) # each NaN (caused by division by zero) is replaced with zero
    cos = torch.clamp( cos, min=-1.0, max=1.0 ) # avoid feeding out-of-rage value (e.g., 1.0004 ) to the acos function
    theta = torch.acos( cos )
    mask = ( cross_z < 0.0 ).to(torch.float32)
    theta_inversed = 2.0 * np.pi - theta
    theta = ( 1.0 - mask ) * theta + mask * theta_inversed
    return theta

def compute_idx_and_weight_for_circular_binning( val, n_bin, max_val ) :
    # Each value in val is assigned, with weights, to two adjacent bins in a circular histogram.
    # This function computes the two bin indices and their weights for each value.
    # A weight is computed by using linear interpolation.
    v = ( val / max_val ) * n_bin
    rbin_idx = torch.floor( v + 0.5 ) % n_bin # index of the right bin
    lbin_idx = rbin_idx - 1 # index of the left bin
    lbin_idx[ lbin_idx == -1 ] = n_bin - 1
    l = rbin_idx - 1
    l[ v >= n_bin - 0.5 ] = n_bin - 1
    rbin_weight = v - ( l + 0.5 ) # weight for the right bin
    lbin_weight = 1.0 - rbin_weight # weight for the left bin
    rbin_idx = rbin_idx.to( torch.int64 )
    lbin_idx = lbin_idx.to( torch.int64 )
    return lbin_idx, lbin_weight, rbin_idx, rbin_weight

def construct_cylindrical_feature_histogram( perpoint_feat, n_bin,
                                             lbin_idx, lbin_weight,
                                             rbin_idx, rbin_weight ) :
    # perpoint_feat: pointwise features [B*T, num_points_per_token, feat_dim]
    # A cylindrical feature histogram is constructed per token

    # scatter weighted pointwise features to their corresponding bins
    num_all_tokens = perpoint_feat.shape[0]
    feat_dim = perpoint_feat.shape[-1]
    cfh = torch.zeros( ( num_all_tokens, n_bin, feat_dim ), dtype=torch.float32, device=perpoint_feat.device )
    cfh = cfh.scatter_add( 1, torch.tile( rbin_idx, ( 1, 1, feat_dim ) ), rbin_weight * perpoint_feat )
    cfh = cfh.scatter_add( 1, torch.tile( lbin_idx, ( 1, 1, feat_dim ) ), lbin_weight * perpoint_feat )

    # count population for each bin to take average
    idx = rbin_idx.squeeze()

    # freq_r = [ torch.bincount( v, minlength=n_bin ).unsqueeze(0) for v in idx ] # TOO SLOW due to the for loop
    # Instead, use scatter_add for population count
    freq_r = torch.zeros( ( num_all_tokens, n_bin ), dtype=torch.float32, device=perpoint_feat.device )
    ones = torch.ones_like( idx, dtype=torch.float32 )
    freq_r = freq_r.scatter_add( 1, idx, ones ) # count population

    freq_l = torch.cat( ( freq_r[ :, 1: ], freq_r[ :, 0 ].unsqueeze(1) ), dim=1 )
    freq = ( freq_r + freq_l ) / 2.0
    freq = freq.unsqueeze(2)
    cfh = cfh / ( freq + 1 ) # average (+1 to avoid division by zero)

    return cfh

def rods_rotat_formula( a, b ) :
    # borrowed from: https://github.com/The-Learning-And-Vision-Atelier-LAVA/SpinNet/blob/main/script/common.py
    B, _ = a.shape
    device = a.device
    b = b.to(device)
    c = torch.cross(a, b)
    theta = torch.acos(F.cosine_similarity(a, b)).unsqueeze(1).unsqueeze(2)

    c = F.normalize(c, p=2, dim=1)
    one = torch.ones(B, 1, 1).to(device)
    zero = torch.zeros(B, 1, 1).to(device)
    a11 = zero
    a12 = -c[:, 2].unsqueeze(1).unsqueeze(2)
    a13 = c[:, 1].unsqueeze(1).unsqueeze(2)
    a21 = c[:, 2].unsqueeze(1).unsqueeze(2)
    a22 = zero
    a23 = -c[:, 0].unsqueeze(1).unsqueeze(2)
    a31 = -c[:, 1].unsqueeze(1).unsqueeze(2)
    a32 = c[:, 0].unsqueeze(1).unsqueeze(2)
    a33 = zero
    Rx = torch.cat(
        (torch.cat((a11, a12, a13), dim=2), torch.cat((a21, a22, a23), dim=2), torch.cat((a31, a32, a33), dim=2)),
        dim=1)
    I = torch.eye(3).to(device)
    R = I.unsqueeze(0).repeat(B, 1, 1) + torch.sin(theta) * Rx + (1 - torch.cos(theta)) * torch.matmul(Rx, Rx)
    return R.transpose(-1, -2)


def compute_angles( vecs1, vecs2 ) :
    # For each vector pair (stored in vecs1 and vecs2), compute an angle between them by using acos function.
    # Assumes both vecs1 and vecs2 are 4D tensors (B, P, K, 3).
    # Since the arccos function is used, each angle takes a value in the range [0,pi].
    # Angle are returned after normalized in the range [0,1].
    normalized_vecs1 = vecs1 / torch.linalg.norm( vecs1, dim=3, keepdims=True )
    normalized_vecs1 = torch.nan_to_num( normalized_vecs1, nan=0.0 ) # each NaN (caused by division by zero) is replaced with zero
    normalized_vecs2 = vecs2 / torch.linalg.norm( vecs2, dim=3, keepdims=True )
    normalized_vecs2 = torch.nan_to_num( normalized_vecs2, nan=0.0 ) # each NaN (caused by division by zero) is replaced with zero
    cos = torch.sum( normalized_vecs1 * normalized_vecs2, dim=3 ) # inner products
    cos = torch.clamp( cos, min=-1.0, max=1.0 ) # avoid feeding out-of-rage value (e.g., 1.0004 ) to the acos function
    angles = torch.acos( cos ) / np.pi
    angles = angles.unsqueeze(3)
    return angles

def compute_absdot( vecs1, vecs2 ) :
    # For each vector pair (stored in vecs1 and vecs2), compute an angle between them by using acos function.
    # Assumes both vecs1 and vecs2 are 4D tensors (B, P, K, 3).
    normalized_vecs1 = vecs1 / torch.linalg.norm( vecs1, dim=3, keepdims=True )
    normalized_vecs1 = torch.nan_to_num( normalized_vecs1, nan=0.0 ) # each NaN (caused by division by zero) is replaced with zero
    normalized_vecs2 = vecs2 / torch.linalg.norm( vecs2, dim=3, keepdims=True )
    normalized_vecs2 = torch.nan_to_num( normalized_vecs2, nan=0.0 ) # each NaN (caused by division by zero) is replaced with zero
    dot = torch.abs( torch.sum( normalized_vecs1 * normalized_vecs2, dim=3 ) ) # absolute of inner products
    dot = dot.unsqueeze(3)
    return dot

def compute_ppf( points_g, points_l, normals ) :
    # Compute point-pair feature similar to "Point Pair Features Based Object Detection and Pose Estimation Revisited"
    # An angle using normal vector is computed as absolute value of inner product to obtain invariance against flipping of normal vector
    # points_g : 3D points in local regions but their coordinates are not normalized (centered)
    # points_l : 3D points in local regions and their coordinates are normalized (centered)
    # normals : (estimated) normal vectors

    # p1, n1: one 3D point in a local region and its normal vector
    # p2, n2: another 3D point in a local region and its normal vector
    # c: the center of a local region (representative point)
    # o: the origin in the 3D space

    B, P, _ = points_l.shape
    delta = rearrange( points_l, 'b i d -> b i 1 d') - rearrange( points_l, 'b j d -> b 1 j d')
    distance = torch.sqrt( torch.sum( torch.square( delta ), dim=3 ) )
    max_distance, _ = torch.max( torch.reshape( distance, [B,-1]), dim=1, keepdims=True )
    distance = distance / max_distance.unsqueeze(2)

    # normalized delta is used below
    norm_delta = torch.linalg.norm( delta, dim=3, keepdims=True )
    delta = delta / norm_delta
    delta = torch.nan_to_num( delta, nan=0.0, posinf=0.0, neginf=0.0 ) # each NaN or inf is replaced with zero

    n1 = normals.unsqueeze(2)
    n1 = n1.repeat(1,1,P,1)
    n2 = rearrange( n1, 'b i j d -> b j i d' )

    # angle between n1 and delta
    dot = torch.sum( n1 * delta, dim=3 ) # inner products
    abs_angle_n1_delta = torch.abs( dot )

    # angle between n2 and delta
    dot = torch.sum( n2 * delta, dim=3 ) # inner products
    abs_angle_n2_delta = torch.abs( dot )

    # angle between n1 and n2
    dot = torch.sum( n1 * n2, dim=3 ) # inner products
    abs_angle_n1_n2 = torch.abs( dot )

    # angle between p1-c and p2-c (assumes 3D points in points_l are already centered)
    norms = torch.linalg.norm( points_l, dim=2, keepdims=True )
    normalized_points = points_l / norms
    normalized_points = torch.nan_to_num( normalized_points, nan=0.0, posinf=0.0, neginf=0.0 ) # each NaN or inf is replaced with zero
    cos = rearrange( normalized_points, 'b i d -> b i 1 d') * rearrange( normalized_points, 'b j d -> b 1 j d')
    cos = torch.sum( cos, dim=3 ) # inner products
    cos = torch.clamp( cos, min=-1.0, max=1.0 ) # avoid feeding out-of-rage value (e.g., 1.0004 ) to the acos function
    angle_p1_p2 = torch.acos( cos ) / np.pi

    # angle between p1-c and p1-p2
    normalized_points = normalized_points.unsqueeze(2)
    cos = normalized_points * delta
    cos = torch.sum( cos, dim=3 ) # inner products
    cos = torch.clamp( cos, min=-1.0, max=1.0 ) # avoid feeding out-of-rage value (e.g., 1.0004 ) to the acos function
    angle_p1_p1p2 = torch.acos( cos ) / np.pi

    # angle between p1-o and p2-o
    norms = torch.linalg.norm( points_g, dim=2, keepdims=True )
    normalized_points = points_g / norms
    normalized_points = torch.nan_to_num( normalized_points, nan=0.0, posinf=0.0, neginf=0.0 ) # each NaN or inf is replaced with zero
    cos = rearrange( normalized_points, 'b i d -> b i 1 d') * rearrange( normalized_points, 'b j d -> b 1 j d')
    cos = torch.sum( cos, dim=3 ) # inner products
    cos = torch.clamp( cos, min=-1.0, max=1.0 ) # avoid feeding out-of-rage value (e.g., 1.0004 ) to the acos function
    angle_p1o_p2o = torch.acos( cos ) / np.pi

    # angle between p1-o and p1-p2
    normalized_points = normalized_points.unsqueeze(2)
    cos = normalized_points * delta
    cos = torch.sum( cos, dim=3 ) # inner products
    cos = torch.clamp( cos, min=-1.0, max=1.0 ) # avoid feeding out-of-rage value (e.g., 1.0004 ) to the acos function
    angle_p1o_p1p2 = torch.acos( cos ) / np.pi

    distance = distance.unsqueeze(3)
    abs_angle_n1_delta = abs_angle_n1_delta.unsqueeze(3)
    abs_angle_n2_delta = abs_angle_n2_delta.unsqueeze(3)
    abs_angle_n1_n2 = abs_angle_n1_n2.unsqueeze(3)
    angle_p1_p2 = angle_p1_p2.unsqueeze(3)
    angle_p1_p1p2 = angle_p1_p1p2.unsqueeze(3)
    angle_p1o_p2o = angle_p1o_p2o.unsqueeze(3)
    angle_p1o_p1p2 = angle_p1o_p1p2.unsqueeze(3)
    ppf = torch.cat( ( distance, abs_angle_n1_delta, abs_angle_n2_delta, abs_angle_n1_n2, angle_p1_p2, angle_p1_p1p2, angle_p1o_p2o, angle_p1o_p1p2 ), dim=3 )
    return ppf

def disambiguate_vector_directions( lps, vecs ) :
    # disambiguate sign of normals in the SHOT manner
    # the codes below are borrowed from: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_normals.html
    knn = lps.shape[2]
    proj = ( vecs[:, :, None] * lps ).sum(3) # projection of the difference on the principal direction
    n_pos = (proj > 0).to(torch.float32).sum(2, keepdim=True) # check how many projections are positive
    # flip the principal directions where number of positive correlations
    flip = (n_pos < (0.5 * knn)).to(torch.float32)
    vecs = (1.0 - 2.0 * flip) * vecs
    return vecs

def sparse_encode( f, n_bin ) :
    # f is a 2D tensor
    # each element in the input feature (f) must be normalized in the range [0,1]

    N, C = f.shape

    # for each element, determine its two adjacent indices ( right index and left index ) for binning
    v = f * ( n_bin - 2 )
    rbin_idx = torch.floor( v + 0.5 ) # index of the right bin
    lbin_idx = rbin_idx - 1 # index of the left bin

    # for each element, determine its weights for the left bin and the right bin
    width_bin = ( 1.0 / ( n_bin - 2 ) )
    half_width_bin = width_bin / 2
    lbin_center = width_bin * lbin_idx + half_width_bin
    rbin_weight = ( f - lbin_center ) * ( n_bin - 2 ) # weight for the right bin
    lbin_weight = 1.0 - rbin_weight # weight for the left bin

    rbin_idx = rbin_idx + 1 # add 1 since index starts with -1
    lbin_idx = lbin_idx + 1 # add 1 since index starts with -1
    rbin_idx = rbin_idx.to( torch.int64 )
    lbin_idx = lbin_idx.to( torch.int64 )

    # scatter weights to their corresponding bins
    f = f.unsqueeze( 2 )
    rbin_idx = rbin_idx.unsqueeze( 2 )
    rbin_weight = rbin_weight.unsqueeze( 2 )
    lbin_idx = lbin_idx.unsqueeze( 2 )
    lbin_weight = lbin_weight.unsqueeze( 2 )
    sparse_f = torch.zeros( ( N, C, n_bin ), dtype=torch.float32, device=f.device )
    sparse_f = sparse_f.scatter_add( 2, rbin_idx, rbin_weight )
    sparse_f = sparse_f.scatter_add( 2, lbin_idx, lbin_weight )
    sparse_f = torch.reshape( sparse_f, [ N, C * n_bin ] )

    return sparse_f


def ksparse( input, k ) :
    # scatter_ operation is very slow
    # d = len( input.shape ) - 1
    # _, indices = torch.topk( input, k )
    # mask = torch.zeros( input.size() ).to( input.device )
    # mask.scatter_( d, indices, 1 )
    # output = torch.mul( input, mask )
    # return output
    d = len( input.shape )
    values, indices = torch.topk( input, k+1 )
    if( d == 2 ) :
        thresholds = values[ :, -1 ].unsqueeze(1)
    elif( d == 3 ) :
        thresholds = values[ :, :, -1 ].unsqueeze(2)
    elif( d == 4 ) :
        thresholds = values[ :, :, :, -1 ].unsqueeze(3)
    else :
        print( "[ksparse] error: unsupported input tensor dimension." )
        quit()
    mask = F.relu( input - thresholds )
    mask = mask > 0.0
    output = input * mask
    return output


def chamfer_distance( set1, set2, metric="L2" ) :
    # Each input set has the shape [B,N,D]

    # metric = "L2"
    # metric = "Cosine"
    # metric = "Softmax"
    # metric = "FocalCosine"

    if( metric == "L2" ) :
        distmats = torch.cdist( set1, set2, p=2.0 )
        d1, _ = torch.min( distmats, dim=2 )
        d2, _ = torch.min( distmats, dim=1 )
        cd = torch.mean( d1 ) + torch.mean( d2 )
    elif( metric == "Cosine" ) :
        set1 = F.normalize( set1, dim=2 )
        set2 = F.normalize( set2, dim=2 )
        set2 = torch.transpose( set2, 1, 2 )
        distmats = 1.0 - torch.bmm( set1, set2 )
        d1, _ = torch.min( distmats, dim=2 )
        d2, _ = torch.min( distmats, dim=1 )
        cd = torch.mean( d1 ) + torch.mean( d2 )
    elif( metric == "Softmax" ) :
        T = 0.2
        set1 = F.normalize( set1, dim=2 )
        set2 = F.normalize( set2, dim=2 )
        set2 = torch.transpose( set2, 1, 2 )
        simmats = torch.bmm( set1, set2 )
        distmats1 = - 1.0 * F.softmax( simmats / T, dim=2 )
        distmats2 = - 1.0 * F.softmax( simmats / T, dim=1 )
        d1, _ = torch.min( distmats1, dim=2 )
        d2, _ = torch.min( distmats2, dim=1 )
        cd = torch.mean( d1 ) + torch.mean( d2 )
    elif( metric == "FocalCosine" ) :
        GAMMA = 0.0
        # GAMMA = 0.2
        set1 = F.normalize( set1, dim=2 )
        set2 = F.normalize( set2, dim=2 )
        set2 = torch.transpose( set2, 1, 2 )
        distmats = 1.0 - torch.bmm( set1, set2 )
        d1, _ = torch.min( distmats, dim=2 )
        d2, _ = torch.min( distmats, dim=1 )
        d1 = - torch.pow( d1, GAMMA ) * torch.log( torch.clamp( 1.0 - d1, min=1e-6 ) )
        d2 = - torch.pow( d2, GAMMA ) * torch.log( torch.clamp( 1.0 - d2, min=1e-6 ) )
        cd = torch.mean( d1 ) + torch.mean( d2 )

    return cd

    # if( metric == "L2" ) :
    #     distmats = torch.cdist( set1, set2, p=2.0 )
    # elif( metric == "Cosine" ) :
    #     set1 = F.normalize( set1, dim=2 )
    #     set2 = F.normalize( set2, dim=2 )
    #     set2 = torch.transpose( set2, 1, 2 )
    #     distmats = 1.0 - torch.bmm( set1, set2 )
    #
    #
    # # elif( metric == "Softmax" ) :
    # #     T = 1.0
    # #     set1 = F.normalize( set1, dim=2 )
    # #     set2 = F.normalize( set2, dim=2 )
    # #     set2 = torch.transpose( set2, 1, 2 )
    # #     simmats = torch.bmm( set1, set2 )
    # #     simmats / T
    #
    #
    #
    #
    # d1, _ = torch.min( distmats, dim=2 )
    # d2, _ = torch.min( distmats, dim=1 )
    # cd = torch.mean( d1 ) + torch.mean( d2 )
    # return cd


if( __name__ == '__main__' ) :

    device = torch.device( "cuda" )
    P1 = np.random.rand( 2, 8, 6 )
    P2 = np.random.rand( 2, 8, 6 )
    P1 = torch.from_numpy( P1 ).to( device )
    P2 = torch.from_numpy( P2 ).to( device )
    chamfer_distance( P1, P2 )
    quit()




    # P = np.random.rand( 16, 2048, 1024 )
    # # P = np.random.rand( 2, 8, 6 )
    # P = P.astype( np.float32 )
    # P = torch.from_numpy( P )
    # device = torch.device( "cuda" )
    # P = P.to( device )
    #
    # import time
    # start_time = time.time()
    # for i in range( 10 ) :
    #     Q = ksparse( P, 3 )
    # elapsed_time = time.time() - start_time
    # print( elapsed_time )
    #
    # quit()

    # P = np.array( [ [ 0.0, 0.0, 0.0 ],
    #                 [ 0.1, 0.2, 0.3 ],
    #                 [ 0.4, 0.5, 0.6 ] ] )
    # N = np.array( [ [ 0.0, 0.0, 0.0 ],
    #                 [ 0.0, 1.0, 0.0 ],
    #                 [ 1.0, 0.0, 0.0 ] ] )
    # # P = np.array( [ [ 0.0, 0.5, 0.1 ],
    # #                 [ 0.1, 0.2, 0.3 ],
    # #                 [ 0.4, 0.5, 0.6 ] ] )
    # # N = np.array( [ [ 0.0, 0.0, 1.0 ],
    # #                 [ 0.0, 1.0, 0.0 ],
    # #                 [ 1.0, 0.0, 0.0 ] ] )
    # P = np.expand_dims( P, 0 )
    # N = np.expand_dims( N, 0 )
    # P = np.tile( P, ( 2, 1, 1 ) )
    # N = np.tile( N, ( 2, 1, 1 ) )
    #
    # # Numpyテンソルを Pytorchテンソルへ変換
    # P = P.astype( np.float32 )
    # P = torch.from_numpy( P )
    # N = N.astype( np.float32 )
    # N = torch.from_numpy( N )
    # print( P.shape )
    # print( N.shape )
    # device = torch.device( "cuda" )
    # P = P.to( device )
    # N = N.to( device )
    #
    # result = compute_ppf( P, N )
    # print( result )
    # quit()
