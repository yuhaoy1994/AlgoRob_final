from sklearn.metrics.pairwise import additive_chi2_kernel
from scipy.optimize import linear_sum_assignment
from pfh import *

import numpy as np


def pc_registeration(src_p, dest_p):
    # find R,t = argmin sum |R*src_p + t - dest_p|^2

    X = np.concatenate(src_p, axis=1)
    Y = np.concatenate(dest_p, axis=1)
    mx = np.mean(X, axis=1)
    my = np.mean(Y, axis=1)

    S = np.matmul(X-mx, (Y-my).T) # 3-by-3
    U,s,V = np.linalg.svd(S)

    tmp = np.linalg.det(np.dot(V.T, U.T))
    R = np.dot(np.dot(V.T, np.diag(np.array([1,1,tmp]))), U.T)
    # print np.dot(R, R.T)
    # print np.linalg.det(R)
    # print np.dot(R.T, R)

    t = my - np.dot(R, mx);
    T = np.zeros((4,4))
    T[:3,:3] = R
    T[:3, 3:] = t
    T[3,3] = 1 
    return T


def correspondence(h1, idx1, h2, idx2, d_thresh=None):
    '''
    compute distance between two histogram - h1, h2
    based on distance, determine the correspondence
    '''

    corr = list()

    # chi2 distance
    D = -0.5*additive_chi2_kernel(np.concatenate(h1, axis=0), np.concatenate(h2, axis=0))

    # determine correspondence using Munkres Algorithm , also known as Hungarian Algorithm
    row_idx, col_idx = linear_sum_assignment(D)

    for r,c in zip(row_idx, col_idx):
        if d_thresh is not None and D[r,c] >= d_thresh:
            # filter out some correspondence with high cost(large distance) 
            # considering the issue that target and source may be only partialy overlapped
            continue
        corr.append((idx1[r], idx2[c]))

    return corr

def icp_pfh(p_src, p_tgt, bmin, bmax, nbins, r_neigh=0.01, a=0.3, b=0.2, d_thresh=None):
    # compute pfh feature
    h_src, idx_src = get_feature(np.concatenate(p_src,axis=1), bmin, bmax, nbins, r_neigh, a, b)
    h_tgt, idx_tgt = get_feature(np.concatenate(p_tgt,axis=1), bmin, bmax, nbins, r_neigh, a, b)

    # determine correspondence
    corr = correspondence(h_src, idx_src, h_tgt, idx_tgt, d_thresh)
    p1 = [p_src[corr[i][0]] for i in range(len(corr))]
    p2 = [p_tgt[corr[i][1]] for i in range(len(corr))]

    T = pc_registeration(p1, p2)

    trans_p_src = [np.dot(T,np.concatenate((pi,[[1]]), axis=0))[:3] for pi in p_src]


    # while not STOP:
    #     # compute transform matrix
    #     T_t = pc_registeration(p1, p2)
    #     T = np.dot(T_t, T)
    #     # transform p_src
    #     p_src = [np.dot(T_t,np.concatenate((pi,[[1]]), axis=0))[:3] for pi in p_src]

    #     if iter < maxIter:
    #         # recompute src point cloud feature
    #         h_src, idx_src = get_feature(np.concatenate(p_src,axis=1), bmin, bmax, nbins)
    #         # recompute correspondence
    #         corr = correspondence(h_src, idx_src, h_tgt, idx_tgt)
    #         p1 = [p_src[corr[i][0]] for i in range(len(corr))]
    #         p2 = [p_tgt[corr[i][1]] for i in range(len(corr))]
    #     else:
    #         STOP = true

    return T, trans_p_src, idx_src




