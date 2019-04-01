import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp

from .matutils import ndgrid
from .codeutils import asArray_N_x_Dim
from .codeutils import isScalar
import discretize
from scipy.spatial import cKDTree, Delaunay
from scipy import interpolate

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
    :param str gridLoc: ['CC', 'N', 'Fx', 'Fy', 'Fz', 'Ex', 'Ex', 'Ey', 'Ez']
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
    """
    Extracts Core Mesh from Global mesh

    :param numpy.ndarray xyzlim: 2D array [ndim x 2]
    :param BaseMesh mesh: The mesh

    This function ouputs::

        - actind: corresponding boolean index from global to core
        - meshcore: core sdiscretize mesh
    """

    if mesh.dim == 1:
        xyzlim = xyzlim.flatten()
        xmin, xmax = xyzlim[0], xyzlim[1]

        xind = np.logical_and(mesh.vectorCCx > xmin, mesh.vectorCCx < xmax)

        xc = mesh.vectorCCx[xind]

        hx = mesh.hx[xind]

        x0 = [xc[0]-hx[0]*0.5]

        meshCore = discretize.TensorMesh([hx], x0=x0)

        actind = (mesh.gridCC > xmin) & (mesh.gridCC < xmax)

    elif mesh.dim == 2:
        xmin, xmax = xyzlim[0, 0], xyzlim[0, 1]
        ymin, ymax = xyzlim[1, 0], xyzlim[1, 1]

        xind = np.logical_and(mesh.vectorCCx > xmin, mesh.vectorCCx < xmax)
        yind = np.logical_and(mesh.vectorCCy > ymin, mesh.vectorCCy < ymax)

        xc = mesh.vectorCCx[xind]
        yc = mesh.vectorCCy[yind]

        hx = mesh.hx[xind]
        hy = mesh.hy[yind]

        x0 = [xc[0]-hx[0]*0.5, yc[0]-hy[0]*0.5]

        meshCore = discretize.TensorMesh([hx, hy], x0=x0)

        actind = (
            (mesh.gridCC[:, 0] > xmin) & (mesh.gridCC[:, 0] < xmax) &
            (mesh.gridCC[:, 1] > ymin) & (mesh.gridCC[:, 1] < ymax)
        )

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


def meshBuilderXYZ(
    xyz, h,
    padX=[0, 0], padY=[0, 0], padZ=[0, 0],
    baseMesh=None,
    expFact=1.3,
    meshType='TENSOR',
    verticalAlignment='top'
):
    """
        Function to quickly generate a Tensor mesh
        given a cloud of xyz points, finest core cell size
        and padding distance.
        If a baseMesh is provided, the core cells will be centered
        on the underlaying mesh to reduce interpolation errors.

        :param numpy.ndarray xyz: n x 3 array of locations [x, y, z]
        :param numpy.ndarray h: 1 x 3 cell size for the core mesh
        :param numpy.ndarray padDist: 2 x 3 padding distances [W,E,S,N,Down,Up]
        [OPTIONAL]
        :param numpy.ndarray padCore: Number of core cells around the xyz locs
        :object SimPEG.Mesh: Base mesh used to shift the new mesh for overlap
        :param float expFact: Expension factor for padding cells [1.3]
        :param string meshType: Specify output mesh type: "TensorMesh"

        RETURNS:
        :object SimPEG.Mesh: Mesh object

    """

    assert meshType in ['TENSOR', 'TREE'], ('Revise meshType. Only ' +
                                            ' TENSOR | TREE mesh ' +
                                            'are implemented')

    assert verticalAlignment in ['center', 'top'], ("verticalAlignment must be 'center' | 'top'")

    # Get extent of points
    limx = np.r_[xyz[:, 0].max(), xyz[:, 0].min()]
    limy = np.r_[xyz[:, 1].max(), xyz[:, 1].min()]
    limz = np.r_[xyz[:, 2].max(), xyz[:, 2].min()]

    # Get center of the mesh
    midX = np.mean(limx)
    midY = np.mean(limy)

    if verticalAlignment == 'center':
        midZ = np.mean(limz)
    else:
        midZ = limz[0]

    nCx = int(limx[0]-limx[1]) / h[0]
    nCy = int(limy[0]-limy[1]) / h[1]
    nCz = int(limz[0]-limz[1]+int(np.min(np.r_[nCx, nCy])/3)) / h[2]

    if meshType == 'TENSOR':
        # Make sure the core has odd number of cells for centereing
        # on global mesh
        if baseMesh is not None:
            nCx += 1 - int(nCx % 2)
            nCy += 1 - int(nCy % 2)
            nCz += 1 - int(nCz % 2)

        # Figure out padding cells from distance
        def expand(dx, pad):
            L = 0
            nC = 0
            while L < pad:
                nC += 1
                L = np.sum(dx * expFact**(np.asarray(range(nC))+1))

            return nC

        # Figure number of padding cells required to fill the space
        npadEast = expand(h[0], padX[0])
        npadWest = expand(h[0], padX[1])
        npadSouth = expand(h[1], padY[0])
        npadNorth = expand(h[1], padY[1])
        npadDown = expand(h[2], padZ[0])
        npadUp = expand(h[2], padZ[1])

        # Create discretization
        hx = [(h[0], npadWest, -expFact),
              (h[0], nCx),
              (h[0], npadEast, expFact)]
        hy = [(h[1], npadSouth, -expFact),
              (h[1], nCy), (h[1],
              npadNorth, expFact)]
        hz = [(h[2], npadDown, -expFact),
              (h[2], nCz),
              (h[2], npadUp, expFact)]

        # Create mesh
        mesh = discretize.TensorMesh([hx, hy, hz], 'CC0')

        # Re-set the mesh at the center of input locations
        # Set origin
        x0 = [midX-np.sum(mesh.hx)/2., midY-np.sum(mesh.hy)/2., midZ - np.sum(mesh.hz)]

        if verticalAlignment == 'center':
            x0[2] = midZ - np.sum(mesh.hz)/2.

        mesh.x0 = x0

    elif meshType == 'TREE':

        # Figure out full extent required from input
        extent = np.max(np.r_[nCx * h[0] + np.sum(padX),
                              nCy * h[1] + np.sum(padY),
                              nCz * h[2] + np.sum(padZ)])

        # Number of cells at the small octree level
        # For now equal in 3D
        maxLevel = int(np.log2(extent/h[0]))+1
        nCx, nCy, nCz = 2**(maxLevel), 2**(maxLevel), 2**(maxLevel)

        # Define the mesh and origin
        mesh = discretize.TreeMesh([
            np.ones(nCx)*h[0],
            np.ones(nCx)*h[1],
            np.ones(nCx)*h[2]
        ])

        # Shift mesh if global mesh is used
        center = np.r_[midX, midY, midZ]
        if baseMesh is not None:

            tree = cKDTree(baseMesh.gridCC)
            _, ind = tree.query(center, k=1)
            center = baseMesh.gridCC[ind, :]

        # Set origin
        x0 = np.r_[
                center[0] - (nCx)*h[0]/2.,
                center[1] - (nCy)*h[1]/2.,
                center[2] - (nCz)*h[2]
            ]

        if verticalAlignment == 'center':
            x0[2] = center[2] - (nCz)*h[2]/2.

        mesh.x0 = x0

    return mesh


def refineTreeXYZ(
            mesh, xyz,
            method="radial",
            octreeLevels=[1, 1, 1],
            octreeLevels_XY=None,
            finalize=False,
):
    """
    Refine a TreeMesh based on XYZ point locations

    :param BaseMesh mesh: The mesh
    :param numpy.ndarray xyz: 2D array of points

    [OPTIONAL]
    :param method: Method used to refine the mesh based on xyz locations
        'radial' (default): Based on radial distance xyz and cell centers
        'surface': Along triangulated surface repeated vertically
        'box': Inside limits defined by outer xyz locations
    :param octreeLevels: List defining the minimum number of cells
        in each octree level.
    :param octreeLevels_XY: List defining the minimum number of padding cells
        added outside the data hull of each octree levels. Optionanl for
        method='surface' and 'box' only
    :param finalize: True | False (default) Finalize the TreeMesh


    This function ouputs::

        - Refined TreeMesh


    """

    if octreeLevels_XY is not None:
        assert len(octreeLevels_XY) == len(octreeLevels), (
            "'octreeLevels_XY' must be the length %i" % len(octreeLevels)
        )

    else:
        octreeLevels_XY = np.zeros_like(octreeLevels)

    # Prime the refinement against large cells
    mesh.insert_cells(
        xyz,
        [mesh.max_level - np.nonzero(octreeLevels)[0][0]]*xyz.shape[0],
        finalize=False
    )

    # Trigger different refine methods
    if method == "radial":

        # Build a cKDTree for fast nearest lookup
        tree = cKDTree(xyz)

        # Compute the outer limits of each octree level
        rMax = np.cumsum(
            mesh.hx.min() *
            np.asarray(octreeLevels) *
            2**np.arange(len(octreeLevels))
        )

        # Radial function
        def inBall(cell):
            xyz = cell.center
            r, ind = tree.query(xyz)

            for ii, nC in enumerate(octreeLevels):

                if r < rMax[ii]:

                    return mesh.max_level-ii

            return 0

        mesh.refine(inBall, finalize=finalize)

    elif method == 'surface':

        # Compute centroid
        centroid = np.mean(xyz, axis=0)

        if mesh.dim == 2:
            rOut = np.abs(centroid[0]-xyz).max()
            hz = mesh.hy.min()
        else:
            # Largest outer point distance
            rOut = np.linalg.norm(np.r_[
                np.abs(centroid[0]-xyz[:, 0]).max(),
                np.abs(centroid[1]-xyz[:, 1]).max()
                ]
            )
            hz = mesh.hz.min()

        # Compute maximum depth of refinement
        zmax = np.cumsum(
            hz *
            np.asarray(octreeLevels) *
            2**np.arange(len(octreeLevels))
        )

        # Compute maximum horizontal padding offset
        padWidth = np.cumsum(
            mesh.hx.min() *
            np.asarray(octreeLevels_XY) *
            2**np.arange(len(octreeLevels_XY))
        )

        # Increment the vertical offset
        zOffset = 0
        xyPad = -1
        depth = zmax[-1]
        # Cycle through the Tree levels backward
        for ii in range(len(octreeLevels)-1, -1, -1):

            dx = mesh.hx.min() * 2**ii

            if mesh.dim == 3:
                dy = mesh.hy.min() * 2**ii
                dz = mesh.hz.min() * 2**ii
            else:
                dz = mesh.hy.min() * 2**ii

            # Increase the horizontal extent of the surface
            if xyPad != padWidth[ii]:
                xyPad = padWidth[ii]

                # Calculate expansion for padding XY cells
                expFactor = (rOut + xyPad) / rOut
                xLoc = (xyz - centroid)*expFactor + centroid

                if mesh.dim == 3:
                    # Create a new triangulated surface
                    tri2D = Delaunay(xLoc[:, :2])
                    F = interpolate.LinearNDInterpolator(tri2D, xLoc[:, 2])
                else:
                    F = interpolate.interp1d(xLoc[:, 0], xLoc[:, 1])

            limx = np.r_[xLoc[:, 0].max(), xLoc[:, 0].min()]
            nCx = int(np.ceil((limx[0]-limx[1]) / dx))

            if mesh.dim == 3:
                limy = np.r_[xLoc[:, 1].max(), xLoc[:, 1].min()]
                nCy = int(np.ceil((limy[0]-limy[1]) / dy))

                # Create a grid at the octree level in xy
                CCx, CCy = np.meshgrid(
                    np.linspace(
                        limx[1], limx[0], nCx
                        ),
                    np.linspace(
                        limy[1], limy[0], nCy
                        )
                )

                xy = np.c_[CCx.reshape(-1), CCy.reshape(-1)]

                # Only keep points within triangulation
                indexTri = tri2D.find_simplex(xy)

            else:
                xy = np.linspace(
                        limx[1], limx[0], nCx
                        )
                indexTri = np.ones_like(xy, dtype='bool')

            # Interpolate the elevation linearly
            z = F(xy[indexTri != -1])

            # Apply vertical padding for current octree level
            zOffset = 0
            while zOffset < depth:

                mesh.insert_cells(
                    np.c_[xy[indexTri != -1], z-zOffset],
                    np.ones_like(z)*mesh.max_level-ii,
                    finalize=False
                )

                zOffset += dz

            depth -= dz * octreeLevels[ii]

        if finalize:
            mesh.finalize()

    elif method == 'box':

        # Define the data extend [bottom SW, top NE]
        bsw = np.min(xyz, axis=0)
        tne = np.max(xyz, axis=0)

        hx = mesh.hx.min()

        if mesh.dim == 2:
            hz = mesh.hy.min()
        else:
            hz = mesh.hz.min()

        # Pre-calculate max depth of each level
        zmax = np.cumsum(
            hz * np.asarray(octreeLevels) *
            2**np.arange(len(octreeLevels))
        )

        if mesh.dim == 2:
            # Pre-calculate outer extent of each level
            padWidth = np.cumsum(
                    mesh.hx.min() *
                    np.asarray(octreeLevels_XY) *
                    2**np.arange(len(octreeLevels_XY))
                )

            # Make a list of outer limits
            BSW = [
                bsw - np.r_[padWidth[ii], zmax[ii]]
                for ii, (octZ, octXY) in enumerate(
                        zip(octreeLevels, octreeLevels_XY)
                )
            ]

            TNE = [
                tne + np.r_[padWidth[ii], zmax[ii]]
                for ii, (octZ, octXY) in enumerate(
                    zip(octreeLevels, octreeLevels_XY)
                )
            ]

        else:
            hy = mesh.hy.min()

            # Pre-calculate outer X extent of each level
            padWidth_x = np.cumsum(
                    hx * np.asarray(octreeLevels_XY) *
                    2**np.arange(len(octreeLevels_XY))
                )

            # Pre-calculate outer Y extent of each level
            padWidth_y = np.cumsum(
                    hy * np.asarray(octreeLevels_XY) *
                    2**np.arange(len(octreeLevels_XY))
                )

            # Make a list of outer limits
            BSW = [
                bsw - np.r_[padWidth_x[ii], padWidth_y[ii], zmax[ii]]
                for ii, (octZ, octXY) in enumerate(
                        zip(octreeLevels, octreeLevels_XY)
                )
            ]

            TNE = [
                tne + np.r_[padWidth_x[ii], padWidth_y[ii], zmax[ii]]
                for ii, (octZ, octXY) in enumerate(
                    zip(octreeLevels, octreeLevels_XY)
                )
            ]

        def inBox(cell):

            xyz = cell.center

            for ii, (nC, bsw, tne) in enumerate(zip(octreeLevels, BSW, TNE)):

                if np.all([xyz > bsw, xyz < tne]):
                    return mesh.max_level-ii

            return cell._level

        mesh.refine(inBox, finalize=finalize)

    else:
        NotImplementedError(
            "Only method= 'radial', 'surface'"
            " or 'box' have been implemented"
        )

    return mesh
