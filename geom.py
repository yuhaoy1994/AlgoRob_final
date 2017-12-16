import numpy as np

import fit

def m_gau_curvature_eval(f, p):
    '''
    evaluate mean, gaussian curvature, given a point and quadratic fit of it
    
    f: 6-by-1 matrix, quadratic fit of local surface
    p: 3-d point

    f = ax^2 + by^2 + cxy + dx + ey + f
    '''

    x = float(p[0][0])
    y = float(p[1][0])
    a = float(f[0][0])
    b = float(f[1][0])
    c = float(f[2][0])
    d = float(f[3][0])
    e = float(f[4][0])
    f = float(f[5][0])

    fxx = 2*a
    fyy = 2*b
    fxy = c
    fx  = 2*a*x + c*y + d
    fy  = 2*b*y + c*x + e

    H = (fxx + fyy + fxx*(fy**2) + fyy*(fx**2) - 2*fx*fy*fxy) / ((1 + fx**2 + fy**2)**1.5) / 2
    K = (fxx*fyy - fxy**2) / ((1 + fx**2 + fy**2)**2)

    return H, K


def principle_curv_eval(H, K):
    '''
    H = (kmin + kmax) / 2
    K = kmin * kmax
    '''    

    kmin = H - np.sqrt(H**2 - K)
    kmax = H + np.sqrt(H**2 - K)

    return kmin, kmax


def shape_index(kmin, kmax):
    '''

    shape index refer to :
    Dorai, C., & Jain, A. K. (1997). COSMOS-A representation scheme for 3D 
    free-form objects. IEEE transactions on pattern analysis and machine intelligence, 19(10), 1115-1130.
    
    shape index basically maps local shape of point into [0, 1] range

    if index is close to 1, then it is more like a convex shape
    if index is close to 0, then it is more like a concave shape
    if index is in somewhere around 0.5, then it may have a shape with less 
    variation around the neighbor, thus may not be a good 3-d keypoint
    
    this method is used in following paper
    refer to:
    Chen, H., & Bhanu, B. (2007). 3D free-form object recognition in range 
    images using local surface patches. Pattern Recognition Letters, 28(10), 1252-1262.
    
    '''

    p = np.pi

    return 0.5 - 1/p * np.arctan((kmin + kmax) / (kmax - kmin))


def shape_index_eval(f, p):
    H, K = m_gau_curvature_eval(f, p) 
    kmin, kmax = principle_curv_eval(H, K)
    return shape_index(kmin, kmax) 

def is_equal_float(testval, val):
    return abs(testval - val) < 1e-5

def extract_keypoint(p_neighbors, points, a, b):
    '''

    extract 3-D keypoints based on shape index

    '''

    # step1. calcualate shape indices for all points
    s = list()
    for i in range(points.shape[1]):
        pn = p_neighbors[i]
        p = points[:,i]

        yi, xi = fit.prepare_fit_quad(pn)
        f      = fit.least_square_fit(yi, xi)

        s.append(shape_index_eval(f, p))

    # step2. compute mean value of shape indices, and range for a point to be 
    # keypoint

    ms = sum(s) / len(s)
    up_b = (1 + a) * ms
    bot_b = (1 - b) * ms


    # step3. find keypoint
    maxs = max(s)
    mins = min(s)

    idx = list()

    for i in range(points.shape[1]):
        if is_equal_float(s[i], maxs) or is_equal_float(s[i],mins):
            idx.append(i)
        elif s[i] > up_b or s[i] < bot_b:
            idx.append(i)


    return idx


