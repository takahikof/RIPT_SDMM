import sys

sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


def generate_grids( num_bins ) :
    num_cells = num_bins * num_bins * num_bins
    grids = np.zeros( ( num_cells, 3 ), dtype=np.float32 )
    grid_interval = 1.0 / num_bins
    offset = grid_interval / 2.0
    for x in range( num_bins ) :
        coord_x = grid_interval * ( x + 1 ) - offset
        for y in range( num_bins ) :
            coord_y = grid_interval * ( y + 1 ) - offset
            for z in range( num_bins ) :
                coord_z = grid_interval * ( z + 1 ) - offset
                idx = ( x * num_bins + y ) * num_bins + z
                grids[ idx, 0 ] = coord_x
                grids[ idx, 1 ] = coord_y
                grids[ idx, 2 ] = coord_z
    return grids

class HandcraftedPodExtractor( nn.Module ) :
    def __init__( self, args ) :
        super( HandcraftedPodExtractor, self ).__init__()
        self.args = args
        self.num_bins = args.pod_num_bins
        self.num_dim_feat = self.num_bins * self.num_bins * self.num_bins * 10
        self.grids = torch.from_numpy( generate_grids( self.num_bins ) ).to( self.args.device )

    def forward( self, input ) :
        # input : an input batch of oriented 3D point sets [B, P, 6]
        #         scale of each point set should be normalized while position of each point set doesn't need to be noramlized

        B, P, _ = input.shape # B: batch size, P: number of 3D point per shape

        pos = input[ :, :, 0:3 ]
        ori = input[ :, :, 3:6 ]

        # Each point set is "voxelized" by using a regular 3D grid.
        # And in each cell of the 3D grid, handcrafted feature (POD) is computed.

        # Firstly, normalize scale of token point sets so that each token point set is fit in a unit cube.
        bbox_max, _ = torch.max( torch.abs( pos ), dim=1, keepdim=True )
        bbox_max, _ = torch.max( bbox_max, dim=2, keepdim=True )
        bbox_max = torch.tile( bbox_max, ( 1, 1, 3 ) )
        bbox_min = - bbox_max
        thickness = torch.clamp( bbox_max - bbox_min, min=1e-5 )
        pos_in_cube = ( pos - bbox_min ) / thickness

        # Assign a cell ID to each point (cell ID indicates which cell in a 3D grid a point belongs to.)
        cell_idx = ( pos_in_cube * self.num_bins ).to( torch.int64 )
        cell_idx = torch.clamp( cell_idx, min=0, max=self.num_bins-1 )
        cell_idx = ( cell_idx[ :, :, 0 ] * self.num_bins + cell_idx[ :, :, 1 ] ) * self.num_bins + cell_idx[ :, :, 2 ]

        # Compute POD feature (ref: Diffusion-on-Manifold Aggregation of Local Features for Shape-based 3D Model Retrieval)
        # frequency feature for each cell
        freq = torch.zeros( ( B, self.num_bins*self.num_bins*self.num_bins ), dtype=torch.float32, device=self.args.device )
        ones = torch.ones_like( cell_idx, dtype=torch.float32 )
        freq = freq.scatter_add( 1, cell_idx, ones ) # count population
        freq = freq.unsqueeze(2)
        uprefix = 1.0 / torch.sqrt( torch.clamp( freq, min=1 ) ) # clamp to avoid division by zero
        f_freq = 0.001 * freq * uprefix

        # gravity center for each cell
        cell_centers = torch.gather( torch.unsqueeze( self.grids, dim=0 ).repeat(B,1,1).to( self.args.device ), dim=1, index=torch.unsqueeze( cell_idx, dim=2 ).repeat(1,1,3).to( self.args.device ) )
        f_mean = torch.zeros( ( B, self.num_bins*self.num_bins*self.num_bins, 3 ), dtype=torch.float32, device=self.args.device )
        f_mean = f_mean.scatter_add( 1, torch.tile( cell_idx.unsqueeze(2), (1,1,3) ), pos_in_cube - cell_centers )
        f_mean *= uprefix

        # covariance of normal vectors for each cell
        cov = torch.matmul( ori.unsqueeze(3), ori.unsqueeze(2) )
        cov = cov.view( B, P, 9 )
        cov = torch.cat( [ cov[:,:,0:3], cov[:,:,4:6], cov[:,:,8].unsqueeze(2) ], dim=2 )
        f_normcov = torch.zeros( ( B, self.num_bins*self.num_bins*self.num_bins, 6 ), dtype=torch.float32, device=self.args.device )
        f_normcov = f_normcov.scatter_add( 1, torch.tile( cell_idx.unsqueeze(2), (1,1,6) ), cov )
        f_normcov /= torch.clamp( freq, min=1 )

        f_all = torch.cat( [ f_freq, f_mean, f_normcov ], dim=2 )
        f_all = f_all.view( B, -1 )
        f_all = F.normalize( f_all ) # L2 normalize

        return f_all

    def get_outdim( self ) :
        return self.num_dim_feat


# class HandcraftedPodExtractor( nn.Module ) :
#     def __init__( self, args ) :
#         super( HandcraftedPodExtractor, self ).__init__()
#         self.args = args
#         self.num_bins = args.pod_num_bins
#         self.num_dim_feat = self.num_bins * self.num_bins * self.num_bins * 12
#         self.grids = torch.from_numpy( generate_grids( self.num_bins ) ).to( self.args.device )
#
#     def forward( self, input ) :
#         # input : an input batch of oriented 3D point sets [B, P, 6]
#         #         scale of each point set should be normalized while position of each point set doesn't need to be noramlized
#
#         B, P, _ = input.shape # B: batch size, P: number of 3D point per shape
#
#         pos = input[ :, :, 0:3 ]
#         ori = input[ :, :, 3:6 ]
#
#         # Each point set is "voxelized" by using a regular 3D grid.
#         # And in each cell of the 3D grid, handcrafted feature is computed.
#
#         # Firstly, normalize scale of token point sets so that each token point set is fit in a unit cube.
#         bbox_max, _ = torch.max( torch.abs( pos ), dim=1, keepdim=True )
#         bbox_max, _ = torch.max( bbox_max, dim=2, keepdim=True )
#         bbox_max = torch.tile( bbox_max, ( 1, 1, 3 ) )
#         bbox_min = - bbox_max
#         thickness = torch.clamp( bbox_max - bbox_min, min=1e-5 )
#         pos_in_cube = ( pos - bbox_min ) / thickness
#
#         # Assign a cell ID to each point (cell ID indicates which cell in a 3D grid a point belongs to.)
#         cell_idx = ( pos_in_cube * self.num_bins ).to( torch.int64 )
#         cell_idx = torch.clamp( cell_idx, min=0, max=self.num_bins-1 )
#         cell_idx = ( cell_idx[ :, :, 0 ] * self.num_bins + cell_idx[ :, :, 1 ] ) * self.num_bins + cell_idx[ :, :, 2 ]
#
#         # Compute handcrafted feature
#         # count the number of 3D points within each cell
#         freq = torch.zeros( ( B, self.num_bins*self.num_bins*self.num_bins ), dtype=torch.float32, device=self.args.device )
#         ones = torch.ones_like( cell_idx, dtype=torch.float32 )
#         freq = freq.scatter_add( 1, cell_idx, ones ) # count population
#         freq = freq.unsqueeze(2)
#
#         # normalize position and scale of 3D points for each cell
#         cell_centers = torch.gather( torch.unsqueeze( self.grids, dim=0 ).repeat(B,1,1), dim=1, index=torch.unsqueeze( cell_idx, dim=2 ).repeat(1,1,3) )
#         pos_in_cube = ( pos_in_cube - cell_centers ) / ( 1.0 / self.num_bins ) # divide by grid interval to normalize the scale of 3D points
#
#         # compute covariance of points and covariance of normal vectors
#         pos_ori = torch.cat( [ pos_in_cube, ori ], dim=0 )
#         cov = torch.matmul( pos_ori.unsqueeze(3), pos_ori.unsqueeze(2) )
#         cov = cov.view( B*2, P, 9 )
#         cov = torch.cat( [ cov[:,:,0:3], cov[:,:,4:6], cov[:,:,8].unsqueeze(2) ], dim=2 ) # use only the upper triangle elements of the 3x3 covariance matrix
#         pos_cov, ori_cov = cov.chunk( 2 )
#
#         # covariance of points for each cell
#         f_poscov = torch.zeros( ( B, self.num_bins*self.num_bins*self.num_bins, 6 ), dtype=torch.float32, device=self.args.device )
#         f_poscov = f_poscov.scatter_add( 1, torch.tile( cell_idx.unsqueeze(2), (1,1,6) ), pos_cov )
#         f_poscov /= torch.clamp( freq, min=1 )
#
#         # covariance of normal vectors for each cell
#         f_normcov = torch.zeros( ( B, self.num_bins*self.num_bins*self.num_bins, 6 ), dtype=torch.float32, device=self.args.device )
#         f_normcov = f_normcov.scatter_add( 1, torch.tile( cell_idx.unsqueeze(2), (1,1,6) ), ori_cov )
#         f_normcov /= torch.clamp( freq, min=1 )
#
#         f_all = torch.cat( [ f_poscov, f_normcov ], dim=2 )
#         f_all = f_all.view( B, -1 )
#         f_all = F.normalize( f_all ) # L2 normalize
#
#         return f_all
#
#     def get_outdim( self ) :
#         return self.num_dim_feat


# class HandcraftedPodExtractor( nn.Module ) :
#     def __init__( self, args ) :
#         super( HandcraftedPodExtractor, self ).__init__()
#         self.args = args
#         self.num_bins = args.pod_num_bins
#         self.num_dim_feat = self.num_bins * self.num_bins * self.num_bins * 10
#         self.grids = torch.from_numpy( generate_grids( self.num_bins ) ).to( self.args.device )
#
#     def forward( self, input ) :
#         # input : an input batch of oriented 3D point sets [B, P, 6]
#         #         scale of each point set should be normalized while position of each point set doesn't need to be noramlized
#
#         B, P, _ = input.shape # B: batch size, P: number of 3D point per shape
#
#         pos = input[ :, :, 0:3 ]
#         ori = input[ :, :, 3:6 ]
#
#         # Each point set is "voxelized" by using a regular 3D grid.
#         # And in each cell of the 3D grid, handcrafted, POD feature is computed.
#
#         # Firstly, normalize scale of token point sets so that each token point set is fit in a unit cube.
#         bbox_max, _ = torch.max( torch.abs( pos ), dim=1, keepdim=True )
#         bbox_max, _ = torch.max( bbox_max, dim=2, keepdim=True )
#         bbox_max = torch.tile( bbox_max, ( 1, 1, 3 ) )
#         bbox_min = - bbox_max
#         thickness = torch.clamp( bbox_max - bbox_min, min=1e-5 )
#         pos_in_cube = ( pos - bbox_min ) / thickness
#
#         # Assign a cell ID to each point (cell ID indicates which cell in a 3D grid a point belongs to.)
#         cell_idx = ( pos_in_cube * self.num_bins ).to( torch.int64 )
#         cell_idx = torch.clamp( cell_idx, min=0, max=self.num_bins-1 )
#         cell_idx = ( cell_idx[ :, :, 0 ] * self.num_bins + cell_idx[ :, :, 1 ] ) * self.num_bins + cell_idx[ :, :, 2 ]
#
#         # Compute POD feature (ref: Diffusion-on-Manifold Aggregation of Local Features for Shape-based 3D Model Retrieval)
#         # frequency feature for each cell
#         freq = torch.zeros( ( B, self.num_bins*self.num_bins*self.num_bins ), dtype=torch.float32, device=self.args.device )
#         ones = torch.ones_like( cell_idx, dtype=torch.float32 )
#         freq = freq.scatter_add( 1, cell_idx, ones ) # count population
#         freq = freq.unsqueeze(2)
#         uprefix = 1.0 / torch.sqrt( torch.clamp( freq, min=1 ) ) # clamp to avoid division by zero
#         f_freq = 0.001 * freq * uprefix
#
#         # gravity center for each cell
#         cell_centers = torch.gather( torch.unsqueeze( self.grids, dim=0 ).repeat(B,1,1), dim=1, index=torch.unsqueeze( cell_idx, dim=2 ).repeat(1,1,3) )
#         f_mean = torch.zeros( ( B, self.num_bins*self.num_bins*self.num_bins, 3 ), dtype=torch.float32, device=self.args.device )
#         f_mean = f_mean.scatter_add( 1, torch.tile( cell_idx.unsqueeze(2), (1,1,3) ), pos_in_cube - cell_centers )
#         f_mean *= uprefix
#
#         # covariance of normal vectors for each cell
#         cov = torch.matmul( ori.unsqueeze(3), ori.unsqueeze(2) )
#         cov = cov.view( B, P, 9 )
#         cov = torch.cat( [ cov[:,:,0:3], cov[:,:,4:6], cov[:,:,8].unsqueeze(2) ], dim=2 )
#         f_normcov = torch.zeros( ( B, self.num_bins*self.num_bins*self.num_bins, 6 ), dtype=torch.float32, device=self.args.device )
#         f_normcov = f_normcov.scatter_add( 1, torch.tile( cell_idx.unsqueeze(2), (1,1,6) ), cov )
#         f_normcov /= torch.clamp( freq, min=1 )
#
#         f_all = torch.cat( [ f_freq, f_mean, f_normcov ], dim=2 )
#         f_all = f_all.view( B, -1 )
#         f_all = F.normalize( f_all ) # L2 normalize
#
#         return f_all
#
#     def get_outdim( self ) :
#         return self.num_dim_feat


# class HandcraftedPodExtractor( nn.Module ) :
#     def __init__( self, args ) :
#         super( HandcraftedPodExtractor, self ).__init__()
#         self.args = args
#         self.num_bins = args.pod_num_bins
#         self.num_dim_feat = self.num_bins * self.num_bins * self.num_bins * 7
#         self.grids = torch.from_numpy( generate_grids( self.num_bins ) ).to( self.args.device )
#
#     def forward( self, input ) :
#         # input : an input batch of oriented 3D point sets [B, P, 6]
#         #         scale of each point set should be normalized while position of each point set doesn't need to be noramlized
#
#         B, P, _ = input.shape # B: batch size, P: number of 3D point per shape
#
#         pos = input[ :, :, 0:3 ]
#         ori = input[ :, :, 3:6 ]
#
#         # Each point set is "voxelized" by using a regular 3D grid.
#         # And in each cell of the 3D grid, handcrafted feature is computed.
#
#         # Firstly, normalize scale of token point sets so that each token point set is fit in a unit cube.
#         bbox_max, _ = torch.max( torch.abs( pos ), dim=1, keepdim=True )
#         bbox_max, _ = torch.max( bbox_max, dim=2, keepdim=True )
#         bbox_max = torch.tile( bbox_max, ( 1, 1, 3 ) )
#         bbox_min = - bbox_max
#         thickness = torch.clamp( bbox_max - bbox_min, min=1e-5 )
#         pos_in_cube = ( pos - bbox_min ) / thickness
#
#         # Assign a cell ID to each point (cell ID indicates which cell in a 3D grid a point belongs to.)
#         cell_idx = ( pos_in_cube * self.num_bins ).to( torch.int64 )
#         cell_idx = torch.clamp( cell_idx, min=0, max=self.num_bins-1 )
#         cell_idx = ( cell_idx[ :, :, 0 ] * self.num_bins + cell_idx[ :, :, 1 ] ) * self.num_bins + cell_idx[ :, :, 2 ]
#
#         # Compute handcrafted feature
#         # count the number of 3D points within each cell
#         freq = torch.zeros( ( B, self.num_bins*self.num_bins*self.num_bins ), dtype=torch.float32, device=self.args.device )
#         ones = torch.ones_like( cell_idx, dtype=torch.float32 )
#         freq = freq.scatter_add( 1, cell_idx, ones ) # count population
#         freq = freq.unsqueeze(2)
#
#         # point density (probability) for each cell
#         f_density = freq / P
#
#         # covariance of normal vectors for each cell
#         cov = torch.matmul( ori.unsqueeze(3), ori.unsqueeze(2) )
#         cov = cov.view( B, P, 9 )
#         cov = torch.cat( [ cov[:,:,0:3], cov[:,:,4:6], cov[:,:,8].unsqueeze(2) ], dim=2 )
#         f_normcov = torch.zeros( ( B, self.num_bins*self.num_bins*self.num_bins, 6 ), dtype=torch.float32, device=self.args.device )
#         f_normcov = f_normcov.scatter_add( 1, torch.tile( cell_idx.unsqueeze(2), (1,1,6) ), cov )
#         f_normcov /= torch.clamp( freq, min=1 )
#
#         f_all = torch.cat( [ f_density, f_normcov ], dim=2 )
#         f_all = f_all.view( B, -1 )
#         f_all = F.normalize( f_all ) # L2 normalize
#
#         return f_all
#
#     def get_outdim( self ) :
#         return self.num_dim_feat
