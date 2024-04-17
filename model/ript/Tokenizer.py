import sys

sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from pytorch3d.ops import sample_farthest_points, estimate_pointcloud_normals, knn_points, knn_gather
from pytorch3d.ops.utils import get_point_covariances
from pytorch3d.common.workaround import symeig3x3
from pytorch3d.ops.points_normals import _disambiguate_vector_directions
from pytorch3d.transforms.rotation_conversions import _axis_angle_rotation
from pytorch3d.loss import chamfer_distance

import numpy as np
import time
from scipy.stats import truncnorm
import utils.torch_utils as tu
from utils.building_blocks import PointNet, ShellNet
from utils.pod_extractor import HandcraftedPodExtractor


class Tokenizer( nn.Module ) :
    def __init__( self, args ) :
        super( Tokenizer, self ).__init__()

        self.args = args
        self.arch_encoder = args.tokenizer_arch_encoder

        if( self.arch_encoder == "pointnet" ) :
            self.encoder = PointNet( self.args )
        elif( self.arch_encoder == "shellnet" ) :
            self.encoder = ShellNet( self.args )
        elif( self.arch_encoder == "pod" ) :
            pod_extractor = HandcraftedPodExtractor( self.args )
            self.encoder = nn.Sequential( pod_extractor,
                                          nn.Linear( pod_extractor.get_outdim(), self.args.num_dim_token ) )
        else :
            print( "error: invalid arch_encoder: " + str( self.arch_encoder ) )
            quit()

    def compute_lrf_SHOT( self, pos_q, pos_t, nn_idx ) :
        # pos_q: [B, P1, 3]
        # pos_t: [B, P2, 6]

        B, P1, C = pos_q.shape
        knn = nn_idx.shape[2]

        # Partition each 3D point set into multiple local point sets (LPS)
        lps = knn_gather( pos_t, nn_idx )
        lps_pos = lps[ :, :, :, 0:3 ]
        lps_ori = lps[ :, :, :, 3:6 ]

        # For each LPS, normalize coordinates of the 3D points
        pos_q = pos_q.unsqueeze(2)
        lps_pos = lps_pos - pos_q

        # covs = torch.einsum( "bijk,bijl->bikl", lps_pos, lps_pos )

        norms = torch.linalg.norm( lps_pos, dim=3, keepdims=True )
        max_norms, _ = torch.max( norms, dim=2, keepdims=True )
        w = max_norms - norms
        w = w / torch.sum( w, dim=2, keepdims=True )
        scaled_lps = 100.0 * lps_pos # for numerical stability
        covs = torch.einsum( "bijk,bijl->bikl", w * scaled_lps, scaled_lps )

        # compute local reference frames
        _, lrfs = symeig3x3( covs, eigenvectors=True )
        # lrfs: [B, , 3, 3], where [:, i, :, 0] corresponds to the normal vector for the local point set i

        # disambiguate normal
        n = tu.disambiguate_vector_directions( lps_pos, lrfs[ :, :, :, 0 ] )

        # disambiguate the main curvature
        z = tu.disambiguate_vector_directions( lps_pos, lrfs[ :, :, :, 2 ] )

        # the secondary curvature is just a cross between n and z
        y = torch.cross(n, z, dim=2)

        # cat to form the set of principal directions
        lrfs = torch.stack( (n, y, z), dim=3 )

        # normalize orientation of each LPS
        lps_pos = torch.reshape( lps_pos, [ B * P1, knn, 3 ] )
        lps_pos = torch.bmm( lps_pos, torch.reshape( lrfs, [ B * P1, 3, 3 ] ) )
        lps_pos = torch.reshape( lps_pos, [ B, P1, knn, 3 ] )

        lps_ori = torch.reshape( lps_ori, [ B * P1, knn, 3 ] )
        lps_ori = torch.bmm( lps_ori, torch.reshape( lrfs, [ B * P1, 3, 3 ] ) )
        lps_ori = torch.reshape( lps_ori, [ B, P1, knn, 3 ] )

        lps = torch.cat( [ lps_pos, lps_ori ], dim=3 )

        return lps, lrfs

    def compute_lrf_EM( self, pos_q, pos_t, ori_q, nn_idx ) :
        # pos_q: [B, P1, 3]
        # pos_t: [B, P2, 6]
        # ori_q: [B, P1, 3]

        B, P1, C = pos_q.shape
        knn = nn_idx.shape[2]

        # Partition each 3D point set into multiple local point sets (LPS)
        lps = knn_gather( pos_t, nn_idx )
        lps_pos = lps[ :, :, :, 0:3 ]
        lps_ori = lps[ :, :, :, 3:6 ]

        # For each LPS, normalize coordinates of the 3D points
        pos_q = pos_q.unsqueeze(2)
        lps_pos = lps_pos - pos_q

        # z-axis corresponds to the normal vector
        # z_axis = torch.reshape( ori_q, [ B, P1, 3, 1 ] )
        z_axis = ori_q

        # x-axis is computed by PCA
        norms = torch.linalg.norm( lps_pos, dim=3, keepdims=True )
        max_norms, _ = torch.max( norms, dim=2, keepdims=True )
        w = max_norms - norms
        w = w / torch.sum( w, dim=2, keepdims=True )
        scaled_lps = 100.0 * lps_pos # for numerical stability
        covs = torch.einsum( "bijk,bijl->bikl", w * scaled_lps, scaled_lps )
        _, lrfs = symeig3x3( covs, eigenvectors=True )
        # lrfs: [B, , 3, 3], where [:, i, :, 0] corresponds to the normal vector for the local point set i
        x_axis = lrfs[ :, :, :, 2 ] # eigen vectors associated with the largest eigen values

        # each principal axis is projected onto the tangent plane of a z-axis
        dot = torch.sum( torch.mul( z_axis, x_axis ), dim=2, keepdim=True )
        x_axis = x_axis - dot * z_axis
        x_axis = F.normalize( x_axis, dim=2 )

        # disambiguate direction of the axes
        z_axis = tu.disambiguate_vector_directions( lps_pos, z_axis )
        x_axis = tu.disambiguate_vector_directions( lps_pos, x_axis )

        # y-axis is just a cross between z and x
        y_axis = torch.cross( z_axis, x_axis, dim=2 )

        # cat to form the set of principal directions
        lrfs = torch.stack( ( z_axis, y_axis, x_axis ), dim=3 )

        # normalize orientation of each LPS
        lps_pos = torch.reshape( lps_pos, [ B * P1, knn, 3 ] )
        lps_pos = torch.bmm( lps_pos, torch.reshape( lrfs, [ B * P1, 3, 3 ] ) )
        lps_pos = torch.reshape( lps_pos, [ B, P1, knn, 3 ] )

        lps_ori = torch.reshape( lps_ori, [ B * P1, knn, 3 ] )
        lps_ori = torch.bmm( lps_ori, torch.reshape( lrfs, [ B * P1, 3, 3 ] ) )
        lps_ori = torch.reshape( lps_ori, [ B, P1, knn, 3 ] )

        lps = torch.cat( [ lps_pos, lps_ori ], dim=3 )

        return lps, lrfs

    def kmeans( self, init, points, n_iter=10 ) :

        B, n_clusters, n_dim = init.shape

        centers = init
        for i in range( n_iter ) :

            # find nearest center
            _, nn_idx, _ = knn_points( points, centers, K=1 ) # [B, P, K]

            # for each cluster, compute sum of the points belonging to the cluster
            new_centers = torch.zeros( ( B, n_clusters, n_dim ), dtype=torch.float32, device=points.device )
            new_centers = new_centers.scatter_add( 1, torch.tile( nn_idx, ( 1, 1, n_dim ) ), points )

            # population count by using scatter_add
            nn_idx = nn_idx.squeeze()
            freq = torch.zeros( ( B, n_clusters ), dtype=torch.float32, device=points.device )
            ones = torch.ones_like( nn_idx, dtype=torch.float32 )
            freq = freq.scatter_add( 1, nn_idx, ones ) # count population
            freq[ freq == 0 ] = 1.0
            freq = freq.unsqueeze(2)

            # update centers
            centers = new_centers / freq

        return centers

    def forward( self, input ) :
        # input : an input batch of 3D point sets [B, P, 6]

        B, P, _ = input.shape # B: batch size, P: number of 3D point per shape
        T = self.args.num_tokens

        pos = input[ :, :, 0:3 ] # position of points
        ori = input[ :, :, 3:6 ] # orientation of points

        # Compute center point for each token
        _, rep_idx = sample_farthest_points( pos, K=self.args.num_tokens, random_start_point=True )
        ctr_pos = torch.gather( pos, dim=1, index=torch.unsqueeze( rep_idx, dim=2 ).repeat(1,1,3) )
        # ctr_pos = self.kmeans( ctr_pos, pos, n_iter=10 ) # run k-means to obtain stable center points

        # Find neighboring points of each center point
        knn = int( self.args.token_scale * P )
        nn_idx = tu.nn_search( ctr_pos, pos, knn )

        # To reduce the number points within each token,
        # neighbor indices are subsampled at equal intervals
        nn_idx = nn_idx.unfold( dimension=2, size=1, step=self.args.token_point_sampling_interval )
        nn_idx = nn_idx.squeeze()
        num_points_per_token = nn_idx.shape[2]

        # Compute token point sets and their reference frames
        if( self.args.tokenizer_lrf == "shot" ) :
            tps, trf = self.compute_lrf_SHOT( ctr_pos, input, nn_idx )
        elif( self.args.tokenizer_lrf == "em" ) :
            ctr_ori = torch.gather( ori, dim=1, index=torch.unsqueeze( rep_idx, dim=2 ).repeat(1,1,3) )
            tps, trf = self.compute_lrf_EM( ctr_pos, input, ctr_ori, nn_idx )
        else :
            print( "error: invalid tokenizer_lrf: " + tokenizer_lrf )
            quit()

        # Extract features from token point sets
        if( self.arch_encoder == "pointnet" ) :
            tokens = self.encoder( tps )
        elif( self.arch_encoder == "shellnet" ) :
            tokens = self.encoder( tps )
        elif( self.arch_encoder == "pod" ) :
            tokens = self.encoder( torch.reshape( tps, [ B * T, num_points_per_token, 6 ] ) )

        tokens = torch.reshape( tokens, [ B, T, self.args.num_dim_token ] )

        return tokens, ctr_pos, trf

    def get_outdim( self ) :
        return self.args.num_dim_token

    def get_parameter( self ) :
        return list( self.parameters() )
