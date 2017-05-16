from __future__ import print_function
import numpy as np
import scipy.sparse as sp
from .matutils import mkvc, sub2ind

try:
    from . import interputils_cython as pyx
    _interp_point_1D = pyx._interp_point_1D
    _interpmat1D = pyx._interpmat1D
    _interpmat2D = pyx._interpmat2D
    _interpmat3D = pyx._interpmat3D
    _interpCython = True
except ImportError:
    print("""Interpolation will not work, use setup.py to compile the cython:

        python setup.py install""")
    _interpCython = False


def interpmat(locs, x, y=None, z=None):
    """Local interpolation computed for each receiver point in turn

    :param numpy.ndarray loc: Location of points to interpolate to
    :param numpy.ndarray x: Tensor of 1st dimension of grid.
    :param numpy.ndarray y: Tensor of 2nd dimension of grid. None by default.
    :param numpy.ndarray z: Tensor of 3rd dimension of grid. None by default.
    :rtype: scipy.sparse.csr_matrix
    :return: Interpolation matrix

    .. plot::

        import discretize
        import numpy as np
        import matplotlib.pyplot as plt
        locs = np.random.rand(50)*0.8+0.1
        x = np.linspace(0, 1, 7)
        dense = np.linspace(0, 1, 200)
        fun = lambda x: np.cos(2*np.pi*x)
        Q = discretize.utils.interpmat(locs, x)
        plt.plot(x, fun(x), 'bs-')
        plt.plot(dense, fun(dense), 'y:')
        plt.plot(locs, Q*fun(x), 'mo')
        plt.plot(locs, fun(locs), 'rx')
        plt.show()

    """

    npts = locs.shape[0]
    locs = locs.astype(float)
    x = x.astype(float)
    if y is None and z is None:
        shape = [x.size]
        inds, vals = _interpmat1D(mkvc(locs), x)
    elif z is None:
        y = y.astype(float)
        shape = [x.size, y.size]
        inds, vals = _interpmat2D(locs, x, y)
    else:
        y = y.astype(float)
        z = z.astype(float)
        shape = [x.size, y.size, z.size]
        inds, vals = _interpmat3D(locs, x, y, z)

    I = np.repeat(range(npts), 2**len(shape))
    J = sub2ind(shape, inds)
    Q = sp.csr_matrix((vals, (I, J)),
                      shape=(npts, np.prod(shape)))
    return Q
