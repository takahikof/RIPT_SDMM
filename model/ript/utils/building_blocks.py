import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat, rearrange
import utils.torch_utils as tu

class PointNet( nn.Module ) :
    def __init__( self, args ) :
        super( PointNet, self ).__init__()

        self.args = args

        self.input_channel = 6

        self.net1 = nn.Sequential( nn.Linear( self.input_channel, 64 ),
                                   nn.BatchNorm1d( 64 ),
                                   nn.ReLU( inplace=True ),
                                   nn.Linear( 64, 128 ),
                                   nn.BatchNorm1d( 128 ),
                                   nn.ReLU( inplace=True ),
                                   nn.Linear( 128, 256 ),
                                   nn.BatchNorm1d( 256 ),
                                   nn.ReLU( inplace=True ) )

        self.net2 = nn.Sequential( nn.Linear( 256, self.args.num_dim_token ) )

    def forward( self, input ) :
        # input: [ B, P, K, 6 ], where B: batch size, P: number of tokens, K: number of 3D points within each token
        # output: [ B, P, C ], C-dimensional tokenwise features
        B, P, K, _ = input.shape
        h = torch.reshape( input, [ -1, self.input_channel ] )
        h = self.net1( h )
        h = torch.reshape( h, [ B, P, K, -1 ] )
        h, _ = torch.max( h, dim=2 )
        output = self.net2( h )
        return output


class ShellNet( nn.Module ) :
    def __init__( self, args ) :
        super( ShellNet, self ).__init__()

        self.args = args

        self.input_channel = 6

        self.liftup = nn.Sequential( nn.Linear( self.input_channel, 64 ),
                                     nn.BatchNorm1d( 64 ),
                                     nn.ReLU( inplace=True ),
                                     nn.Linear( 64, 128 ),
                                     nn.BatchNorm1d( 128 ),
                                     nn.ReLU( inplace=True ) )

        self.num_division = 4 # number of shells
        # num_points_per_div = int( self.args.num_points_per_token / self.num_division ) # number of feaures per division
        # self.shellpool = nn.MaxPool2d( ( 1, num_points_per_div ), stride=( 1, num_points_per_div ) )
        # # self.shellpool = nn.AvgPool2d( ( 1, num_points_per_div ), stride=( 1, num_points_per_div ) )

        kernel_size = 2
        self.conv1d = nn.Sequential(
                        nn.Conv2d( 128, 128, ( 1, kernel_size ) ),
                        nn.BatchNorm2d( 128 ),
                        nn.ReLU( inplace=True ),
                        nn.Conv2d( 128, 256, ( 1, kernel_size ) ),
                        nn.BatchNorm2d( 256 ),
                        nn.ReLU( inplace=True ),
                        nn.Conv2d( 256, 512, ( 1, kernel_size ) ),
                        nn.BatchNorm2d( 512 ),
                        nn.ReLU( inplace=True )
                        )

        self.embed = nn.Sequential( nn.Linear( 512, self.args.num_dim_token ) )


    def forward( self, input ) :
        # input: [ B, P, K, 6 ], where B: batch size, P: number of tokens, K: number of 3D points within each token
        # output: [ B, P, C ], C-dimensional tokenwise features

        # !!! For each local region, positions of the 3D points are already normalized (centered).
        # Also, the K points in each local point set (i.e., input[ , , :, :]) are ordered according to their norm.
        # Therefore, there is no need to sort the K points.
        B, P, K, _ = input.shape
        h = torch.reshape( input, [ B*P*K, self.input_channel ] )
        h = self.liftup( h )
        h = torch.reshape( h, [ B, P, K, -1 ] )

        h = h.permute(0,3,1,2)
        # h = self.shellpool( h )

        num_points_per_div = int( K / self.num_division ) # number of feaures per division
        h = F.max_pool2d( h, kernel_size=( 1, num_points_per_div ), stride=( 1, num_points_per_div ) )

        h = self.conv1d( h )
        h = h.permute(0,2,3,1)
        h, _ = torch.max( h, dim=2 )
        output = self.embed( h )

        return output
