# -*- coding: utf-8 -*-
import numpy as np
import torch
import transform_3d as t3d

def point_mixup( pointsets, mixup_alpha, mixup_mode, inv=False ) :
    B, P, C = pointsets.shape
    mixup_pointsets = []
    mixup_lambdas = []

    for i in range( B ) :
        if( inv == True ) :
            idx1 = i
            idx2 = i - 1
            if( idx2 < 0 ) :
                idx2 = B - 1
        else :
            idx1 = i
            idx2 = ( i + 1 ) % B

        ps1 = pointsets[ idx1 ]
        ps2 = pointsets[ idx2 ].detach().clone()

        # before mixup, one point set (ps2) is randomly rotated
        R = t3d.generate_randomrot_matrix()
        R = torch.from_numpy( np.asarray( R, dtype=np.float32 ) ).to( ps2.device )
        if( C == 6 ) :
            ps2 = torch.cat( [ torch.matmul( ps2[ :, 0:3 ], R ), torch.matmul( ps2[ :, 3:6 ], R ) ], dim=1 )
        else :
            ps2 = torch.matmul( ps2, R )

        # determine lambda, i.e., mixup weight
        lamb = np.random.beta( mixup_alpha, mixup_alpha )
        num_pts1 = int( np.round( P * lamb ) )
        num_pts2 = P - num_pts1
        mixup_lambdas.append( lamb )

        # randomly choose indices of 3D points
        if( mixup_mode == "R" ) : # random and uniform mixup
            indices1 = torch.randperm( P )[:num_pts1]
            indices2 = torch.randperm( P )[:num_pts2]
        elif( mixup_mode == "K" ) : # random yet spatially-constrained mixup using k-nearest neighbors
            randidx = np.random.randint( 0, P )
            query_point = ps1[ randidx, 0:3 ].unsqueeze(0)
            distmat = torch.cdist( query_point, ps1[ :, 0:3 ] )
            _, indices1 = torch.topk( distmat, num_pts1, dim=1, largest=False, sorted=False )
            distmat = torch.cdist( query_point, ps2[ :, 0:3 ] )
            _, indices2 = torch.topk( distmat, num_pts2, dim=1, largest=True, sorted=False )
            indices1 = indices1.squeeze()
            indices2 = indices2.squeeze()

        # create two subsets to be mixed
        subset1 = ps1[ indices1 ]
        subset2 = ps2[ indices2 ]
        if( subset1.dim() == 1 ) :
            subset1 = subset1.unsqueeze(0)
        if( subset2.dim() == 1 ) :
            subset2 = subset2.unsqueeze(0)

        # mixup
        mixup_pointsets.append( torch.cat( [ subset1, subset2 ], dim=0 ).unsqueeze(0) )

        # print( num_pts1 )
        # print( num_pts2 )
        # result = torch.cat( [ subset1, subset2 ], dim=0 )
        # print( result.shape )
        # result = result.to('cpu').detach().numpy().copy()
        # np.savetxt( "mixed.xyz", result )
        # quit()

    mixup_pointsets = torch.cat( mixup_pointsets, dim=0 )
    return mixup_pointsets, mixup_lambdas
