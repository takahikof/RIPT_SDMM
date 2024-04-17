import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import utils.torch_utils as tu
from utils.transformer import PointTransformer
from utils.feature_normalizer import IterNorm, DecorrelatedBatchNorm
from pytorch3d.ops import sample_farthest_points, knn_points, knn_gather
from pytorch3d.common.workaround import symeig3x3

class TokenSetTransformer( nn.Module ) :
    def __init__( self, args ) :
        super( TokenSetTransformer, self ).__init__()
        self.args = args

        self.num_center_pts = [] # number of tokens for each transformer block
        for i in range( self.args.trsfm_num_blocks ) :
            nt = int( self.args.num_tokens // np.power(2,(i+1)) )
            if( nt <= 8 ) : # set a lower bound for the number of tokens at deep layers
                nt = 8
            self.num_center_pts.append( nt )
        print( "The number of 3D points at each resolution level: " )
        print( self.num_center_pts )

        self.num_nns = [] # number of nearest neighbors for each transformer block
        for i in range( self.args.trsfm_num_blocks ) :
            knn = self.args.trsfm_knn * np.power( 2, i )
            if( i > 0 and self.num_center_pts[ i-1 ] <= knn ) : # avoid that knn becomes larger than the number of tokens
                knn = self.num_center_pts[ i-1 ]
            self.num_nns.append( knn )

        print( "The number of nearest neighbors at each resolution level: " )
        print( self.num_nns )

        # layers for self-attention
        self.attention_mode = self.args.trsfm_attention_mode
        self.inner_dim = self.args.num_dim_token

        self.to_q = nn.ModuleList([])
        self.to_kv = nn.ModuleList([])
        self.pos_encoder_qk = nn.ModuleList([])
        self.ori_encoder_qk = nn.ModuleList([])
        self.pos_encoder_v = nn.ModuleList([])
        self.ori_encoder_v = nn.ModuleList([])
        self.mlp = nn.ModuleList([])
        self.normalizer1 = nn.ModuleList([])
        self.normalizer2 = nn.ModuleList([])
        self.attention_mlp = nn.ModuleList([])
        self.coeff = []
        self.relu = nn.ReLU( inplace=True )
        for i in range( self.args.trsfm_num_blocks ) :

            self.to_q.append( nn.Linear( self.args.num_dim_token, self.inner_dim, bias = False ) )
            self.to_kv.append( nn.Linear( self.args.num_dim_token, self.inner_dim * 2, bias = False ) )

            self.mlp.append( nn.Sequential( nn.Linear( self.inner_dim, self.inner_dim * 2 ),
                                            nn.ReLU( inplace=True ),
                                            nn.Linear( self.inner_dim * 2, self.inner_dim ) ) )
            self.normalizer1.append( nn.BatchNorm1d( self.inner_dim ) )
            self.normalizer2.append( nn.BatchNorm1d( self.inner_dim ) )

            self.attention_mlp.append( nn.Sequential( nn.Linear( self.inner_dim, self.inner_dim ),
                                                      nn.ReLU(),
                                                      nn.Linear( self.inner_dim, self.inner_dim ) ) )

        # layers for feature embedding
        self.embedder = nn.Sequential( nn.Linear( self.inner_dim, self.args.num_dim_embedding ),
                                       nn.BatchNorm1d( self.args.num_dim_embedding ),
                                       nn.ReLU( inplace=True ) )


    def self_attention( self, Q, K, V, layer_id ) :

        att_mode = self.attention_mode
        b, t, c = V.shape

        if( att_mode == "scalar_sa" ) : # scalar self-attention
            QK = torch.bmm( Q, torch.transpose( K, 1, 2 ) )
            QK = QK / np.sqrt( self.inner_dim )
            A = F.softmax( QK, dim=2 ) # attention map
            outputs = torch.bmm( A, V )

        elif( att_mode == "vector_sa" ) : # vector self-attention
            # code below is based on: https://github.com/lucidrains/point-transformer-pytorch/blob/main/point_transformer_pytorch/point_transformer_pytorch.py
            QK = Q[:, :, None, :] - K[:, None, :, :] # [B, 1, T, C]
            QK = QK.squeeze()
            sim = self.attention_mlp[ layer_id ]( QK )
            A = sim.softmax( dim=1 ) # attention map
            outputs = torch.sum( A * V, dim=1 )

        else :
            print( "error: invalid attention mode:" + att_mode )
            quit()

        return outputs


    def update_centers_and_features( self, init, points, feats, n_iter=10 ) :

        # compute center 3D points by k-means clustering
        B, n_clusters, n_dim = init.shape
        _, _, C = feats.shape

        centers = init
        for i in range( n_iter ) :

            # find nearest center
            _, nn_idx, _ = knn_points( points, centers, K=1 ) # [B, P, K]

            # for each cluster, compute sum of the points belonging to the cluster
            new_centers = torch.zeros( ( B, n_clusters, n_dim ), dtype=torch.float32, device=points.device )
            new_centers = new_centers.scatter_add( 1, torch.tile( nn_idx, ( 1, 1, n_dim ) ), points )

            # population count by using scatter_add
            nn_idx_sq = nn_idx.squeeze()
            freq = torch.zeros( ( B, n_clusters ), dtype=torch.float32, device=points.device )
            ones = torch.ones_like( nn_idx_sq, dtype=torch.float32 )
            freq = freq.scatter_add( 1, nn_idx_sq, ones ) # count population
            freq[ freq == 0 ] = 1.0
            freq = freq.unsqueeze(2)

            # update centers
            centers = new_centers / freq

        # compute average feature of each cluster
        avg_feats = torch.zeros( ( B, n_clusters, C ), dtype=torch.float32, device=points.device )
        avg_feats = avg_feats.scatter_add( 1, torch.tile( nn_idx, ( 1, 1, C ) ), feats )
        avg_feats = avg_feats / freq

        return centers, avg_feats

    def compute_lrf_SHOT( self, pos_q, pos_t ) :
        # pos_q: [B, P1, 3]
        # pos_t: [B, P2, 3]

        B, P1, C = pos_q.shape

        # Use all 3D points in pos_t as neighbors
        lps_pos = torch.tile( pos_t.unsqueeze( 1 ), ( 1, P1, 1, 1 ) )

        # For each LPS, normalize coordinates of the 3D points
        pos_q = pos_q.unsqueeze(2)
        lps_pos = lps_pos - pos_q

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

        return lrfs

    def forward( self, tokens, centers, lrfs ) :
        # tokens : an input batch of token features [B, T, C]
        # centers : an input batch of center coordinates of tokens [B, T, 3]
        # lrfs : an input batch of local reference frames of tokens [B, T, 3, 3]
        # B: batch size, T: number of tokens per shape, C: number of feature dimensions
        B, T, C = tokens.shape

        T_target = T
        centers_target = centers
        lrfs_target = torch.reshape( lrfs, [ B, T, 3*3 ] )
        tokens_target = tokens

        for i in range( self.args.trsfm_num_blocks ) :

            T_query = self.num_center_pts[ i ]

            # print( "T_target:"+str(T_target)+", T_query:"+str(T_query))
            # print( tokens_target.shape )

            if( self.args.trsfm_subsample_mode == "fps" ) :
                # Choose representative tokens by using farthest point sampling on the 3D space
                _, rep_idx = sample_farthest_points( centers_target, K=T_query, random_start_point=True )
                centers_query = torch.gather( centers_target, dim=1, index=torch.unsqueeze( rep_idx, dim=2 ).repeat(1,1,3) )
                lrfs_query = torch.gather( lrfs_target, dim=1, index=torch.unsqueeze( rep_idx, dim=2 ).repeat(1,1,3*3) )
                tokens_query = torch.gather( tokens_target, dim=1, index=torch.unsqueeze( rep_idx, dim=2 ).repeat(1,1,self.inner_dim) ) # tokens of representative points

            elif( self.args.trsfm_subsample_mode == "kmeans" ) :
                # Choose representative tokens by using farthest point sampling on the 3D space
                _, rep_idx = sample_farthest_points( centers_target, K=T_query, random_start_point=True )
                centers_query = torch.gather( centers_target, dim=1, index=torch.unsqueeze( rep_idx, dim=2 ).repeat(1,1,3) )

                # Recompute positions and features of representative tokens
                centers_query, tokens_query = self.update_centers_and_features( centers_query, centers_target, tokens_target, n_iter=10 )

                # Recompute LRFs of representative tokens
                lrfs_query = torch.reshape( self.compute_lrf_SHOT( centers_query, centers_target ), ( B, T_query, 3*3 ) )
            else :
                print( "error: invalid trsfm_subsample_mode : " + self.args.trsfm_subsample_mode )
                quit()

            # Find neighbors of representative tokens
            nn_idx = tu.nn_search( centers_query, centers_target, self.num_nns[ i ] )
            centers_target_knn = knn_gather( centers_target, nn_idx )
            lrfs_target_knn = knn_gather( lrfs_target, nn_idx )

            # Preserve token features for skip connection
            skip = torch.reshape( tokens_query, [ B*T_query, self.inner_dim ] )

            # Normalize features
            if( i > 0 ) : # skip normalization at the first layer because input features are already normalized.
                t = torch.cat( [ tokens_query, tokens_target ], dim=1 )
                t = torch.reshape( t, [ B*(T_query+T_target), self.inner_dim ] )
                t = self.normalizer1[ i ]( t )
                t = torch.reshape( t, [ B, T_query+T_target, self.inner_dim ] )
                tokens_query = t[ :, 0:T_query, : ]
                tokens_target = t[ :, T_query:, : ]

            # Map tokens to Queries, Keys, and Values
            Q = self.to_q[ i ]( tokens_query )
            K, V = self.to_kv[ i ]( tokens_target ).chunk( 2, dim=-1 )

            # neighbors of each representative point are used as keys and values
            K = knn_gather( K, nn_idx )
            V = knn_gather( V, nn_idx )

            Q = torch.reshape( Q, [ B*T_query, 1, self.inner_dim ] )
            K = torch.reshape( K, [ B*T_query, self.num_nns[ i ], self.inner_dim ] )
            V = torch.reshape( V, [ B*T_query, self.num_nns[ i ], self.inner_dim ] )

            # Self-attention
            tokens_query = self.self_attention( Q, K, V, i )
            tokens_query = torch.reshape( tokens_query, [ B*T_query, self.inner_dim ] )
            tokens_query += skip

            # MLP after self-attention
            skip = tokens_query # preserve token features for skip connection
            tokens_query = self.normalizer2[ i ]( tokens_query )
            tokens_query = self.mlp[ i ]( tokens_query )
            tokens_query += skip

            # Update information for the next layer
            T_target = T_query
            centers_target = torch.reshape( centers_query, [ B, T_query, 3 ] )
            lrfs_target = torch.reshape( lrfs_query, [ B, T_query, 9 ] )
            tokens_target = torch.reshape( tokens_query, [ B, T_query, self.inner_dim ] )

        global_feats = torch.mean( tokens_target, dim=1 )
        global_feats = self.embedder( global_feats )
        global_feats = tu.ksparse( global_feats, k=64 )
        global_feats = F.normalize( global_feats, dim=1 )

        return global_feats

    def get_outdim( self ) :
        return self.args.token_num_dims

    def get_parameter( self ) :
        return list( self.parameters() )
