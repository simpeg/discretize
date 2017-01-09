import numpy as np
from .matutils import ndgrid
from .codeutils import asArray_N_x_Dim
from .codeutils import isScalar
import scipy.ndimage as ndi
import scipy.sparse as sp

import sys
if sys.version_info < (3,):
    num_types = [int, long, float]
else:
    num_types = [int, float]


def exampleLrmGrid(nC, exType):
    assert type(nC) == list, "nC must be a list containing the number of nodes"
    assert len(nC) == 2 or len(nC) == 3, "nC must either two or three dimensions"
    exType = exType.lower()

    possibleTypes = ['rect', 'rotate']
    assert exType in possibleTypes, "Not a possible example type."

    if exType == 'rect':
        return list(ndgrid([np.cumsum(np.r_[0, np.ones(nx)/nx]) for nx in nC], vector=False))
    elif exType == 'rotate':
        if len(nC) == 2:
            X, Y = ndgrid([np.cumsum(np.r_[0, np.ones(nx)/nx]) for nx in nC], vector=False)
            amt = 0.5-np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
            amt[amt < 0] = 0
            return [X + (-(Y - 0.5))*amt, Y + (+(X - 0.5))*amt]
        elif len(nC) == 3:
            X, Y, Z = ndgrid([np.cumsum(np.r_[0, np.ones(nx)/nx]) for nx in nC], vector=False)
            amt = 0.5-np.sqrt((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2)
            amt[amt < 0] = 0
            return [X + (-(Y - 0.5))*amt, Y + (-(Z - 0.5))*amt, Z + (-(X - 0.5))*amt]


def random_model(shape, seed=None, anisotropy=None, its=100, bounds=None):
    """
        Create a random model by convolving a kernel with a
        uniformly distributed model.

        :param tuple shape: shape of the model.
        :param int seed: pick which model to produce, prints the seed if you don't choose.
        :param numpy.ndarray anisotropy: this is the (3 x n) blurring kernel that is used.
        :param int its: number of smoothing iterations
        :param list bounds: bounds on the model, len(list) == 2
        :rtype: numpy.ndarray
        :return: M, the model


        .. plot::

            import matplotlib.pyplot as plt
            import discretize
            plt.colorbar(plt.imshow(discretize.utils.random_model((50, 50), bounds=[-4, 0])))
            plt.title('A very cool, yet completely random model.')
            plt.show()


    """
    if bounds is None:
        bounds = [0, 1]

    if seed is None:
        seed = np.random.randint(1e3)
        print('Using a seed of: ', seed)

    if type(shape) in num_types:
        shape = (shape, ) # make it a tuple for consistency

    np.random.seed(seed)
    mr = np.random.rand(*shape)
    if anisotropy is None:
        if len(shape) is 1:
            smth = np.array([1, 10., 1], dtype=float)
        elif len(shape) is 2:
            smth = np.array([[1, 7, 1], [2, 10, 2], [1, 7, 1]], dtype=float)
        elif len(shape) is 3:
            kernal = np.array([1, 4, 1], dtype=float).reshape((1, 3))
            smth = np.array(sp.kron(sp.kron(kernal, kernal.T).todense()[:], kernal).todense()).reshape((3, 3, 3))
    else:
        assert len(anisotropy.shape) is len(shape), 'Anisotropy must be the same shape.'
        smth = np.array(anisotropy, dtype=float)

    smth = smth/smth.sum() # normalize
    mi = mr
    for i in range(its):
        mi = ndi.convolve(mi, smth)

    # scale the model to live between the bounds.
    mi = (mi - mi.min())/(mi.max()-mi.min()) # scaled between 0 and 1
    mi = mi*(bounds[1]-bounds[0])+bounds[0]

    return mi



def meshTensor(value):
    """**meshTensor** takes a list of numbers and tuples
    that have the form::

        mT = [ float, (cellSize, numCell), (cellSize, numCell, factor) ]

    For example, a time domain mesh code needs
    many time steps at one time::

        [(1e-5, 30), (1e-4, 30), 1e-3]

    Means take 30 steps at 1e-5 and then 30 more at 1e-4,
    and then one step of 1e-3.

    Tensor meshes can also be created by increase factors::

        [(10.0, 5, -1.3), (10.0, 50), (10.0, 5, 1.3)]

    When there is a third number in the tuple, it
    refers to the increase factor, if this number
    is negative this section of the tensor is flipped right-to-left.

    .. plot::

        import discretize
        tx = [(10.0, 10, -1.3), (10.0, 40), (10.0, 10, 1.3)]
        ty = [(10.0, 10, -1.3), (10.0, 40)]
        M = discretize.TensorMesh([tx, ty])
        M.plotGrid(showIt=True)

    """
    if type(value) is not list:
        raise Exception('meshTensor must be a list of scalars and tuples.')

    proposed = []
    for v in value:
        if isScalar(v):
            proposed += [float(v)]
        elif type(v) is tuple and len(v) == 2:
            proposed += [float(v[0])]*int(v[1])
        elif type(v) is tuple and len(v) == 3:
            start = float(v[0])
            num = int(v[1])
            factor = float(v[2])
            pad = ((np.ones(num)*np.abs(factor))**(np.arange(num)+1))*start
            if factor < 0: pad = pad[::-1]
            proposed += pad.tolist()
        else:
            raise Exception('meshTensor must contain only scalars and len(2) or len(3) tuples.')

    return np.array(proposed)


def closestPoints(mesh, pts, gridLoc='CC'):
    """Move a list of points to the closest points on a grid.

    :param BaseMesh mesh: The mesh
    :param numpy.ndarray pts: Points to move
    :param string gridLoc: ['CC', 'N', 'Fx', 'Fy', 'Fz', 'Ex', 'Ex', 'Ey', 'Ez']
    :rtype: numpy.ndarray
    :return: nodeInds
    """

    pts = asArray_N_x_Dim(pts, mesh.dim)
    grid = getattr(mesh, 'grid' + gridLoc)
    nodeInds = np.empty(pts.shape[0], dtype=int)

    for i, pt in enumerate(pts):
        if mesh.dim == 1:
            nodeInds[i] = ((pt - grid)**2).argmin()
        else:
            nodeInds[i] = ((np.tile(pt, (grid.shape[0], 1)) - grid)**2).sum(axis=1).argmin()

    return nodeInds


def ExtractCoreMesh(xyzlim, mesh, meshType='tensor'):
    """Extracts Core Mesh from Global mesh

    :param numpy.ndarray xyzlim: 2D array [ndim x 2]
    :param BaseMesh mesh: The mesh

    This function ouputs::

        - actind: corresponding boolean index from global to core
        - meshcore: core mesh

    Warning: 1D and 2D has not been tested
    """
    import discretize
    if mesh.dim == 1:
        xyzlim = xyzlim.flatten()
        xmin, xmax = xyzlim[0], xyzlim[1]

        xind = np.logical_and(mesh.vectorCCx > xmin, mesh.vectorCCx < xmax)

        xc = mesh.vectorCCx[xind]

        hx = mesh.hx[xind]

        x0 = [xc[0]-hx[0]*0.5, yc[0]-hy[0]*0.5]

        meshCore = discretize.TensorMesh([hx, hy], x0=x0)

        actind = (mesh.gridCC[:, 0] > xmin) & (mesh.gridCC[:, 0] < xmax)

    elif mesh.dim == 2:
        xmin, xmax = xyzlim[0, 0], xyzlim[0, 1]
        ymin, ymax = xyzlim[1, 0], xyzlim[1, 1]

        yind = np.logical_and(mesh.vectorCCy > ymin, mesh.vectorCCy < ymax)
        zind = np.logical_and(mesh.vectorCCz > zmin, mesh.vectorCCz < zmax)

        xc = mesh.vectorCCx[xind]
        yc = mesh.vectorCCy[yind]

        hx = mesh.hx[xind]
        hy = mesh.hy[yind]

        x0 = [xc[0]-hx[0]*0.5, yc[0]-hy[0]*0.5]

        meshCore = discretize.TensorMesh([hx, hy], x0=x0)

        actind = (mesh.gridCC[:, 0] > xmin) & (mesh.gridCC[:, 0] < xmax) \
               & (mesh.gridCC[:, 1] > ymin) & (mesh.gridCC[:, 1] < ymax) \

    elif mesh.dim == 3:
        xmin, xmax = xyzlim[0, 0], xyzlim[0, 1]
        ymin, ymax = xyzlim[1, 0], xyzlim[1, 1]
        zmin, zmax = xyzlim[2, 0], xyzlim[2, 1]

        xind = np.logical_and(mesh.vectorCCx > xmin, mesh.vectorCCx < xmax)
        yind = np.logical_and(mesh.vectorCCy > ymin, mesh.vectorCCy < ymax)
        zind = np.logical_and(mesh.vectorCCz > zmin, mesh.vectorCCz < zmax)

        xc = mesh.vectorCCx[xind]
        yc = mesh.vectorCCy[yind]
        zc = mesh.vectorCCz[zind]

        hx = mesh.hx[xind]
        hy = mesh.hy[yind]
        hz = mesh.hz[zind]

        x0 = [xc[0]-hx[0]*0.5, yc[0]-hy[0]*0.5, zc[0]-hz[0]*0.5]

        meshCore = discretize.TensorMesh([hx, hy, hz], x0=x0)

        actind = (
            (mesh.gridCC[:, 0] > xmin) & (mesh.gridCC[:, 0] < xmax) &
            (mesh.gridCC[:, 1] > ymin) & (mesh.gridCC[:, 1] < ymax) &
            (mesh.gridCC[:, 2] > zmin) & (mesh.gridCC[:, 2] < zmax)
        )

    else:
        raise Exception("Not implemented!")

    return actind, meshCore
