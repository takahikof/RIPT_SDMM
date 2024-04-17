# -*- coding: utf-8 -*-
import numpy as np
import math

def normalize_data( pointset ) :
    # Normalizes position and scale of a point set

    pos = pointset[ :, 0:3 ];
    ori = pointset[ :, 3:6 ];

    # Normalizes position
    mean = np.mean( pos, axis=0 );
    pos = pos - mean;

    # Normalizes scale of a 3D model so that it is enclosed by a sphere whose radius is 1.0
    radius = 1.0;
    norms = np.linalg.norm( pos, axis=1 );
    max_norm = np.max( norms );
    pos = pos * ( radius / ( max_norm + 1e-6 ) );

    # Normalizes length of normal vectors
    norms = np.linalg.norm( ori, axis=1, keepdims=True );
    ori = ori / ( norms + 1e-6 );
    output = np.hstack( [ pos, ori ] );

    return output;

# https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/axangles.py
def axangle2mat(axis, angle, is_normalized=False):
    ''' Rotation matrix for rotation angle `angle` around `axis`
    Parameters
    ----------
    axis : 3 element sequence
       vector specifying axis for rotation.
    angle : scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False.
    Returns
    -------
    mat : array shape (3,3)
       rotation matrix for specified rotation
    Notes
    -----
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    '''
    x, y, z = axis
    if not is_normalized:
        n = math.sqrt(x*x + y*y + z*z)
        x = x/n
        y = y/n
        z = z/n
    c = math.cos(angle); s = math.sin(angle); C = 1-c
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC
    return np.array([
            [ x*xC+c,   xyC-zs,   zxC+ys ],
            [ xyC+zs,   y*yC+c,   yzC-xs ],
            [ zxC-ys,   yzC+xs,   z*zC+c ]])



# def randomrot_azimuth( oript ) :
#     # uprightベクトルはz軸方向とする
#     axis = np.array( [ 0.0, 0.0, 1.0 ] );
#     angle_degree = np.random.rand() * 360.0; # [0, 360)
#     angle_radian = angle_degree * np.pi / 180.0;
#
#     R = np.transpose( axangle2mat( axis, angle_radian, is_normalized=True ) );
#
#     return np.hstack( [ np.matmul( oript[ :, 0:3 ], R ), np.matmul( oript[ :, 3:6 ], R ) ] );
#
# def randomrot_so3( oript ) :
#     axis_x = np.array( [ 1.0, 0.0, 0.0 ] );
#     axis_y = np.array( [ 0.0, 1.0, 0.0 ] );
#     axis_z = np.array( [ 0.0, 0.0, 1.0 ] );
#
#     angle_degree_x = np.random.rand() * 360.0; # [0, 360)
#     angle_radian_x = angle_degree_x * np.pi / 180.0;
#     angle_degree_y = np.random.rand() * 360.0; # [0, 360)
#     angle_radian_y = angle_degree_y * np.pi / 180.0;
#     angle_degree_z = np.random.rand() * 360.0; # [0, 360)
#     angle_radian_z = angle_degree_z * np.pi / 180.0;
#
#     R_x = np.transpose( axangle2mat( axis_x, angle_radian_x, is_normalized=True ) );
#     R_y = np.transpose( axangle2mat( axis_y, angle_radian_y, is_normalized=True ) );
#     R_z = np.transpose( axangle2mat( axis_z, angle_radian_z, is_normalized=True ) );
#     R = np.matmul( np.matmul( R_x, R_y ), R_z );
#
#     return np.hstack( [ np.matmul( oript[ :, 0:3 ], R ), np.matmul( oript[ :, 3:6 ], R ) ] );



def generate_randomrot_matrix( angle_max = 180 ) :
    # x, y, z 軸のそれぞれについて[ -angle_max, +angle_max ] (単位は°)の範囲でランダム回転する行列を作成

    axis_x = np.array( [ 1.0, 0.0, 0.0 ] );
    axis_y = np.array( [ 0.0, 1.0, 0.0 ] );
    axis_z = np.array( [ 0.0, 0.0, 1.0 ] );

    angle_degree_x = ( np.random.rand() * 2.0 - 1.0 ) * angle_max;
    angle_radian_x = angle_degree_x * np.pi / 180.0;
    angle_degree_y = ( np.random.rand() * 2.0 - 1.0 ) * angle_max;
    angle_radian_y = angle_degree_y * np.pi / 180.0;
    angle_degree_z = ( np.random.rand() * 2.0 - 1.0 ) * angle_max;
    angle_radian_z = angle_degree_z * np.pi / 180.0;

    R_x = np.transpose( axangle2mat( axis_x, angle_radian_x, is_normalized=True ) );
    R_y = np.transpose( axangle2mat( axis_y, angle_radian_y, is_normalized=True ) );
    R_z = np.transpose( axangle2mat( axis_z, angle_radian_z, is_normalized=True ) );
    R = np.matmul( np.matmul( R_x, R_y ), R_z );

    return R;



# def generate_randomrot_matrix_one_axis( axis = None ) :
#     # axis (ノルムは1で正規化済みとする)を中心としたランダム回転行列を作成
#
#     # デフォルトではz軸方向 (ModelNet10/40に含まれる形状データの上向き)
#     if( axis is None ) :
#         axis = np.array( [ 0.0, 0.0, 1.0 ] );
#
#     angle_degree = np.random.rand() * 360.0; # [0, 360)
#     angle_radian = angle_degree * np.pi / 180.0;
#
#     R = np.transpose( axangle2mat( axis, angle_radian, is_normalized=True ) );
#
#     return R;
#
# def generate_randomrot_matrix( angle_max = 180 ) :
#     # x, y, z 軸のそれぞれについて[ -angle_max, +angle_max ] (単位は°)の範囲でランダム回転する行列を作成
#
#     axis_x = np.array( [ 1.0, 0.0, 0.0 ] );
#     axis_y = np.array( [ 0.0, 1.0, 0.0 ] );
#     axis_z = np.array( [ 0.0, 0.0, 1.0 ] );
#
#     angle_degree_x = ( np.random.rand() * 2.0 - 1.0 ) * angle_max;
#     angle_radian_x = angle_degree_x * np.pi / 180.0;
#     angle_degree_y = ( np.random.rand() * 2.0 - 1.0 ) * angle_max;
#     angle_radian_y = angle_degree_y * np.pi / 180.0;
#     angle_degree_z = ( np.random.rand() * 2.0 - 1.0 ) * angle_max;
#     angle_radian_z = angle_degree_z * np.pi / 180.0;
#
#     R_x = np.transpose( axangle2mat( axis_x, angle_radian_x, is_normalized=True ) );
#     R_y = np.transpose( axangle2mat( axis_y, angle_radian_y, is_normalized=True ) );
#     R_z = np.transpose( axangle2mat( axis_z, angle_radian_z, is_normalized=True ) );
#     R = np.matmul( np.matmul( R_x, R_y ), R_z );
#
#     R = np.concatenate( [ R, np.array( [ [ 0.0 ], [ 0.0 ], [ 0.0] ] ) ], axis=1 );
#     a = np.reshape( np.array( [ 0.0, 0.0, 0.0, 1.0 ] ), [ 1, 4 ] );
#     R = np.concatenate( [ R, a ], axis=0 );
#
#     return R;
#
# def generate_randomshear_matrix( param_max = 0.5 ) :
#     # x, y, z 軸のそれぞれについてランダムせん断する行列を作成．shearing factorは[ -param_max, +param_max ] の範囲でランダムに決定
#     # shearing参考：https://www.gatevidyalay.com/3d-shearing-in-computer-graphics-definition-examples/
#     s = ( np.random.rand() * 2.0 - 1.0 ) * param_max;
#     t = ( np.random.rand() * 2.0 - 1.0 ) * param_max;
#     shear_mat_xy = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [s, t, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] # shearing in z-axis (z does not change while x and y change.)
#     shear_mat_xz = [[1.0, 0.0, 0.0, 0.0], [s, 1.0, t, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] # shearing in y-axis
#     shear_mat_yz = [[1.0, s, t, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] # shearing in x-axis
#     shear_mat = np.dot(np.dot(shear_mat_xy, shear_mat_xz), shear_mat_yz)
#     return shear_mat;
#
# def generate_randomscale_matrix( param_max = 0.2 ) :
#     # x, y, z 軸のそれぞれについて[ 1-param_max, 1+param_max ] 範囲でランダム非等方スケーリングする行列を作成
#     k_x = 1.0 + ( np.random.rand() * 2.0 - 1.0 ) * param_max;
#     k_y = 1.0 + ( np.random.rand() * 2.0 - 1.0 ) * param_max;
#     k_z = 1.0 + ( np.random.rand() * 2.0 - 1.0 ) * param_max;
#     scale_mat = np.array( [[k_x, 0.0, 0.0 ], [0.0, k_y, 0.0], [0.0, 0.0, k_z]] );
#     return scale_mat;

def generate_randomscale_matrix( param_min=2/3.0, param_max = 3/2.0 ) :
    # x, y, z 軸のそれぞれについて[ param_min, param_max ] 範囲でランダム非等方スケーリングする行列を作成
    s = np.random.uniform( low=param_min, high=param_max, size=3 )
    scale_mat = np.array( [[s[0], 0.0, 0.0 ], [0.0, s[1], 0.0], [0.0, 0.0, s[2]]] );
    return scale_mat;

# def generate_randomtranslate_matrix( param_max = 0.2 ) :
#     # x, y, z 軸のそれぞれについて[ -param_max, +param_max ] 範囲でランダム平行移動する行列を作成
#     tx = ( np.random.rand() * 2.0 - 1.0 ) * param_max;
#     ty = ( np.random.rand() * 2.0 - 1.0 ) * param_max;
#     tz = ( np.random.rand() * 2.0 - 1.0 ) * param_max;
#
#     translate_mat = np.array( [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [tx, ty, tz, 1.0]] );
#     return translate_mat;
