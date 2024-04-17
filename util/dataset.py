# -*- coding: utf-8 -*-
import h5py
import numpy as np
from torch.utils.data import Dataset
from multicrop import multicrop_pointset, multicrop_collate_fn
import sys
from scipy.stats import truncnorm
import transform_3d as t3d

def normalize_data( pointset ) :
    # 点群形状1個の位置と大きさを正規化する

    pos = pointset[ :, 0:3 ] # 座標値 (x,y,z)
    ori = pointset[ :, 3:6 ] # 法線ベクトル

    # 位置を正規化
    mean = np.mean( pos, axis=0 )
    pos = pos - mean

    # 大きさを正規化 (単位球に収まるようスケーリング)
    radius = 1.0
    norms = np.linalg.norm( pos, axis=1 )
    max_norm = np.max( norms )
    pos = pos * ( radius / ( max_norm + 1e-6 ) )

    # ついでに，法線ベクトルの長さが1になるよう正規化
    norms = np.linalg.norm( ori, axis=1, keepdims=True )
    ori = ori / ( norms + 1e-6 )

    output = np.hstack( [ pos, ori ] )

    return output;

def load_data( path_dataset, partition ) :

    # データセットを読み込む
    h5py_file = path_dataset + "_" + partition + ".h5"
    f = h5py.File( h5py_file, 'r' )
    all_data = f['data'][:].astype('float32') # 点群データ．テンソルのshapeは[形状数, 点数, 6]であり，6の最初の3次元は座標を表し，残りの3次元は法線ベクトルを表す
    all_label = f['label'][:].astype('int64') # ラベル．整数値(カテゴリ番号)の配列．
    all_label = np.reshape( all_label, ( -1, 1 ) )
    f.close()

    for i in range( all_data.shape[ 0 ] ) : # 各点群データについて
        all_data[ i ] = normalize_data( all_data[ i ] ) # 大きさと位置の正規化を行う

    return all_data, all_label

# def augment_pointcloud( pointcloud ) :
#     xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3]) # (0.67, 1.5)の範囲でスケーリング倍率を決定
#     xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3]) # (-0.2, 0.2)の範囲で平行移動量を決定
#     augmented_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
#     return augmented_pointcloud

class PointCloudDataset( Dataset ) : # データセットクラス

    def __init__( self, args, path_dataset, num_points, partition, data_aug, rotation ) :
        self.args = args
        self.data, self.label = load_data( path_dataset, partition )
        self.num_points = num_points
        self.partition = partition
        self.data_aug = data_aug
        self.rotation = rotation

    def __getitem__( self, item ) : # item番目の形状データを取得

        # self.num_points個の点をランダムにサブサンプリング
        idxs = np.arange( self.data[item].shape[0] )
        np.random.shuffle( idxs )
        idxs = idxs[ : self.num_points ]
        pointcloud = self.data[item][ idxs ]
        label = self.label[item]
        pointcloud = pointcloud.astype('float32')

        if( self.rotation == "nr" ) : # no rotation
            pass
        elif( self.rotation == "so3" ) : # SO3 rotation
            pos = pointcloud[ :, 0:3 ]
            ori = pointcloud[ :, 3:6 ]
            R = t3d.generate_randomrot_matrix().astype('float32')
            pointcloud = np.hstack( [ np.matmul( pos, R ), np.matmul( ori, R ) ] )
        else :
            print( "error: invalid rotation mode: " + str( self.rotation ) )
            quit()

        if( self.data_aug ) :
            pointcloud = multicrop_pointset( pointcloud, self.args )
            for i in range( len( pointcloud ) ) :
                pointcloud[ i ] = pointcloud[ i ].astype('float32')

        return pointcloud, label

        # if( self.data_aug ) :
        #     if( np.random.rand() < 0.8 ) : # augmentation is usually applied
        #
        #         pos = pointcloud[ :, 0:3 ]
        #         ori = pointcloud[ :, 3:6 ]
        #
        #         # anisotropic scaling
        #         S = t3d.generate_randomscale_matrix( self.args.da_scale_min, self.args.da_scale_max )
        #         pos = np.matmul( pos, S )
        #
        #         # jittering
        #         lowerbound, upperbound = - self.args.da_jitter_clip, self.args.da_jitter_clip # clip too large displacements
        #         mean = 0.0
        #         std = self.args.da_jitter_std
        #         a, b = ( lowerbound - mean ) / std, ( upperbound - mean ) / std # standardize (ref. documentation of scipy.stats.truncnorm)
        #         noise = truncnorm.rvs( a, b, loc=mean, scale=std, size=pos.shape )
        #         pos = pos + noise
        #
        #         pointcloud = np.hstack( [ pos, ori ] )
        #         pointcloud = pointcloud.astype('float32')
        #
        # # return pointcloud, label
        # return pointcloud, label, item # item is used as a pseudo label for instance discrimination


    def __len__(self):
        return self.data.shape[0]
