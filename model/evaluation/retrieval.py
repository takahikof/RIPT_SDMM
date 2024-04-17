# -*- coding: utf-8 -*-
import numpy as np
import time
import random
import math
import sys
import os

from sklearn.metrics import pairwise_distances
from sklearn.metrics import label_ranking_average_precision_score
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

def retrieval( F, L ) :
    # Evaluate retrieavl accuracy.
    # Inputs:
    #   F: An N-by-D distance matrix. N is the number of data samples. D is the number of feature dimensions.
    #   L: A label vector with N elements. Each element is integer value indicating a semantic class.
    # Outputs:
    #   NN: Nearest Neighbor
    #   MAP: Mean Average Precision
    #   RP: values for Recall-Precision curve

    recall_step = 0.05;
    mean_nearest_neighbor = 0.0;
    mean_average_precision = 0.0;

    mean_recall = np.zeros( int( 1.0 / recall_step ) );
    mean_precision = np.zeros( int( 1.0 / recall_step ) );

    n_data = F.shape[0];
    n_class = np.unique( L ).shape[0]
    categorywise_ap = []
    for i in range( n_class ) :
        categorywise_ap.append([])

    D = pairwise_distances( F, metric="cosine" );

    n_valid_queries = 0 # In some datasets, there exists a data sample whose category size is 1.
                        # Such data samples are excluded from queries for retrieval.
    for i in range( n_data ) : # for each query

        # compute Euclidean distances among the query and the other features
        dist_vec = D[ i ];
        gt_vec = np.asarray( L==L[i], dtype=np.int32 ); # 1 if the retrieval target belongs to the same category with the query, 0 otherwise

        dist_vec_woq = np.delete( dist_vec, i ); # distance vector without query
        gt_vec_woq = np.delete( gt_vec, i );     # groundtruth vector without query
        gt_vec_woq_sp = csr_matrix( gt_vec_woq );   # convert to sparse matrix

        relevant = gt_vec_woq_sp.indices;
        n_correct = gt_vec_woq_sp.nnz; # number of correct targets for the query

        if( n_correct > 0 ) :

            n_valid_queries += 1

            rank = rankdata( dist_vec_woq, 'max')[ relevant ]; # positions where correct data appear in a retrieval ranking
            rank_sorted = np.sort( rank );

            # nearest neighbor
            if( rank_sorted[ 0 ] == 1 ) : # correct data appears at the top of the retrieval ranking
                mean_nearest_neighbor += 1.0

            # average precision
            rd = rankdata( dist_vec_woq[relevant], 'max');
            ap = (rd / rank).mean();
            mean_average_precision += ap;
            categorywise_ap[ L[i] ].append( ap )

            # recall-precision curve
            one_to_n = ( np.arange( n_correct ) + 1 ).astype( np.float32 );
            precision = one_to_n / rank_sorted;
            recall = one_to_n / n_correct;
            recall_interp = np.arange( recall_step, 1.01, recall_step );
            precision_interp = np.interp( recall_interp, recall, precision );
            mean_recall = recall_interp; # no need to average
            mean_precision += precision_interp;
        else :
            pass

    mean_nearest_neighbor /= n_valid_queries;
    mean_average_precision /= n_valid_queries; # This is Micro-averaged MAP
    mean_precision /= n_valid_queries;

    # Compute Macro-averaged MAP
    macro_map = 0.0
    n_valid_class = 0
    for i in range( n_class ) :
        if( len( categorywise_ap[ i ] ) > 0 ) :
            n_valid_class += 1
            macro_map += np.mean( np.asarray( categorywise_ap[ i ] ) )
    macro_map /= n_valid_class

    return( mean_nearest_neighbor, mean_average_precision, macro_map, ( mean_recall, mean_precision ) );
