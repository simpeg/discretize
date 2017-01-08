from __future__ import print_function, division
import numpy as np
import sys
if sys.version_info < (3,):
    scalarTypes = [float, int, long, np.float_, np.int_]
else:
    scalarTypes = [float, int, np.float_, np.int_]


def isScalar(f):
    if type(f) in scalarTypes:
        return True
    elif isinstance(f, np.ndarray) and f.size == 1 and type(f[0]) in scalarTypes:
        return True
    return False


def asArray_N_x_Dim(pts, dim):
        if type(pts) == list:
            pts = np.array(pts)
        assert isinstance(pts, np.ndarray), "pts must be a numpy array"

        if dim > 1:
            pts = np.atleast_2d(pts)
        elif len(pts.shape) == 1:
            pts = pts[:, np.newaxis]

        assert pts.shape[1] == dim, "pts must be a column vector of shape (nPts, {0:d}) not ({1:d}, {2:d})".format(*((dim,)+pts.shape))

        return pts
