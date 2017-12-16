import numpy as np

def prepare_fit_quad(points):
    '''
    prepare data for fitting 3-d points into quadratic funcition
    
    return:
    y: n-by-1 matrix that corresponds z-coordinates
    x: 6-by-n matrix, each column corresponds to [x^2, y^2, xy, x, y, 1]

    '''

    if len(points) < 6:
        print "WARNING: to fit quadratic function at least 6 points are needed!!!"

    y = list()
    x = list()
    for p in points:
        y.append(p[2][0])
        x.append(np.concatenate([p[0][0]**2, p[1][0]**2, p[0][0]*p[1][0], p[0][0], p[1][0], np.matrix(1)], axis=0))

        return np.concatenate(y, axis=0), np.concatenate(x, axis=1)


def least_square_fit(y, x):
    '''
    optimize function : min_w || y - x^T * W ||

    
    y: # of observation - by - 1 matrix
    x: # of param - by - # of observation

    '''

    xshape = x.shape
    yshape = y.shape

    if xshape[1] != yshape[0]:
        print "dimension mismatch (%d, %d) is not compatible with (%d,%d)" % (xshape[0],xshape[1],yshape[0],yshape[1])


    xTpinv = np.linalg.pinv(x.T)

    return xTpinv.dot(y)

