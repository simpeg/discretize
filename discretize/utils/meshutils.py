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

    Parameters
    ----------
    shape: tuple
        shape of the model.
    seed: int
        pick which model to produce, prints the seed if you don't choose.
    anisotropy: numpy.ndarray
        this is the (3 x n) blurring kernel that is used.
    its: int
        number of smoothing iterations
    bounds: list
        bounds on the model, len(list) == 2

    Returns
    -------
    numpy.ndarray
        M, the model

    Examples
    --------

    .. plot::
        :include-source:

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
        :include-source:

        import discretize
        tx = [(10.0, 10, -1.3), (10.0, 40), (10.0, 10, 1.3)]
        ty = [(10.0, 10, -1.3), (10.0, 40)]
        mesh = discretize.TensorMesh([tx, ty])
        mesh.plotGrid(show_it=True)

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

    Parameters
    ----------
    mesh: BaseMesh
        The mesh
    pts: numpy.ndarray
        Points to move
    gridLoc: str
        ['CC', 'N', 'Fx', 'Fy', 'Fz', 'Ex', 'Ex', 'Ey', 'Ez']

    Returns
    -------
    numpy.ndarray
        nodeInds

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


def ExtractCoreMesh(xyzlim, mesh, mesh_type='tensor'):
    """
    Extracts Core Mesh from Global mesh

    Parameters
    ----------
    xyzlim: numpy.ndarray
        2D array [ndim x 2]
    mesh: BaseMesh
        The mesh

    Returns
    -------
    tuple
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


def mesh_builder_xyz(
    xyz, h,
    padding_distance=[[0, 0], [0, 0], [0, 0]],
    base_mesh=None,
    depth_core=None,
    expansion_factor=1.3,
    mesh_type='tensor'
):
    """
    Function to quickly generate a Tensor or Tree mesh
    given a cloud of xyz points, finest core cell size
    and padding distance.
    If a base_mesh is provided, the core cells will be centered
    on the underlaying mesh to reduce interpolation errors.
    The core extent is set by the bounding box of the xyz location.

    Parameters
    ----------
    xyz: numpy.ndarray
        Location points [n x dim]
    h: list
        Cell size for the core mesh [1 x ndim]
    padding_distance: list
        Padding distances [[W,E], [N,S], [Down,Up]]
    base_mesh: discretize.BaseMesh
        discretize mesh used to center the core mesh
    depth_core: float
        Depth of core mesh below xyz
    expansion_factor: float
        Expension factor for padding cells [1.3]
    mesh_type: str
        Specify output mesh type: ["TENSOR"] or "TREE"

    Returns
    --------
    discretize.BaseMesh
        Mesh of "mesh_type"

    Examples
    --------
    .. plot::
        :include-source:

        import discretize
        import matplotlib.pyplot as plt
        import numpy as np

        xyLoc = np.random.randn(8,2)

        mesh = discretize.utils.meshutils.mesh_builder_xyz(
            xyLoc, [0.1, 0.1], depth_core=0.5,
            padding_distance=[[1,2], [1,0]],
            mesh_type='tensor',
        )


        axs = plt.subplot()
        mesh.plotImage(mesh.vol, grid=True, ax=axs)
        axs.scatter(xyLoc[:,0], xyLoc[:,1], 15, c='w', zorder=3)
        axs.set_aspect('equal')
        plt.show()

    """
    if mesh_type.lower() not in ['tensor', 'tree']:
        raise ValueError(
            'Revise mesh_type. Only TENSOR | TREE mesh are implemented'
        )

    # Get extent of points
    limits = []
    center = []
    nC = []
    for dim in range(xyz.shape[1]):

        max_min = np.r_[xyz[:, dim].max(), xyz[:, dim].min()]
        limits += [max_min]
        center += [np.mean(max_min)]
        nC += [int((max_min[0] - max_min[1]) / h[dim])]

    if depth_core is not None:
        nC[-1] += int(depth_core / h[-1])
        limits[-1][1] -= depth_core

    if mesh_type.lower() == 'tensor':

        # Figure out padding cells from distance
        def expand(dx, pad):
            length = 0
            nc = 0
            while length < pad:
                nc += 1
                length = np.sum(dx * expansion_factor**(np.asarray(range(nc))+1))

            return nc

        # Define h along each dimension
        h_dim = []
        nC_x0 = []
        for dim in range(xyz.shape[1]):
            h_dim += [[
                (
                    h[dim],
                    expand(h[dim], padding_distance[dim][0]),
                    -expansion_factor
                ),
                (h[dim], nC[dim]),
                (
                    h[dim],
                    expand(h[dim], padding_distance[dim][1]),
                    expansion_factor
                )
            ]]

            nC_x0 += [h_dim[-1][0][1]]

        # Create mesh
        mesh = discretize.TensorMesh(h_dim)

    elif mesh_type.lower() == 'tree':

        # Figure out full extent required from input
        h_dim = []
        nC_x0 = []
        for ii, cc in enumerate(nC):
            extent = (
                limits[ii][0] -
                limits[ii][1] +
                np.sum(padding_distance[ii])
            )

            # Number of cells at the small octree level
            maxLevel = int(np.log2(extent / h[ii])) + 1
            h_dim += [np.ones(2**maxLevel) * h[ii]]

        # Define the mesh and origin
        mesh = discretize.TreeMesh(h_dim)

        for ii, cc in enumerate(nC):
            core = limits[ii][0] - limits[ii][1]
            pad2 = int(np.log2(padding_distance[ii][0] / h[ii] + 1))

            nC_x0 += [int(np.ceil((mesh.h[ii].sum() - core) / h[ii] / 2))]


    # Set origin
    x0 = []
    for ii, hi in enumerate(mesh.h):
        x0 += [limits[ii][1] - np.sum(hi[:nC_x0[ii]])]

    mesh.x0 = np.hstack(x0)

    # Shift mesh if global mesh is used based on closest to centroid
    axis = ["x", "y", "z"]
    if base_mesh is not None:
        for dim in range(base_mesh.dim):

            cc_base = getattr(
                    base_mesh,
                    "vectorCC{orientation}".format(
                            orientation=axis[dim]
                        )
            )

            cc_local = getattr(
                    mesh,
                    "vectorCC{orientation}".format(
                            orientation=axis[dim]
                        )
            )

            shift = (
                cc_base[np.max([np.searchsorted(cc_base, center[dim])-1, 0])] -
                cc_local[np.max([np.searchsorted(cc_local, center[dim])-1, 0])]
            )

            x0[dim] += shift

            mesh.x0 = np.hstack(x0)

    return mesh


def refine_tree_xyz(
    mesh, xyz,
    method="radial",
    octree_levels=[1, 1, 1],
    octree_levels_padding=None,
    finalize=False,
    min_level=0,
    max_distance=np.inf
):
    """
    Refine a TreeMesh based on xyz point locations

    Parameters
    ----------
    mesh: TreeMesh
        The TreeMesh object to be refined
    xyz: numpy.ndarray
        2D array of points
    method: str
        Method used to refine the mesh based on xyz locations

        - "radial": Based on radial distance xyz and cell centers
        - "surface": Along triangulated surface repeated vertically
        - "box": Inside limits defined by outer xyz locations

    octree_levels: list
        Minimum number of cells around points in each k octree level
        [N(k), N(k-1), ...]
    octree_levels_padding: list
        Padding cells added to the outer limits of the data each octree levels
        used for method= "surface" and "box" [N(k), N(k-1), ...].
    finalize: bool
        True | [False]    Finalize the TreeMesh
    max_distance: float
        Maximum refinement distance from xyz locations.
        Used for method="surface" to reduce interpolation distance.

    Returns
    --------
    discretize.TreeMesh
        mesh

    """

    if octree_levels_padding is not None:

        if len(octree_levels_padding) != len(octree_levels):
            raise ValueError(
                "'octree_levels_padding' must be the length %i" % len(octree_levels)
            )

    else:
        octree_levels_padding = np.zeros_like(octree_levels)

    # Prime the refinement against large cells
    mesh.insert_cells(
        xyz,
        [mesh.max_level - np.nonzero(octree_levels)[0][0]]*xyz.shape[0],
        finalize=False
    )

    # Trigger different refine methods
    if method.lower() == "radial":

        # Build a cKDTree for fast nearest lookup
        tree = cKDTree(xyz)

        # Compute the outer limits of each octree level
        rMax = np.cumsum(
            mesh.hx.min() *
            np.asarray(octree_levels) *
            2**np.arange(len(octree_levels))
        )

        # Radial function
        def inBall(cell):
            xyz = cell.center
            r, ind = tree.query(xyz)

            for ii, nC in enumerate(octree_levels):

                if r < rMax[ii]:

                    return mesh.max_level-ii

            return min_level

        mesh.refine(inBall, finalize=finalize)

    elif method.lower() == 'surface':

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
            np.asarray(octree_levels) *
            2**np.arange(len(octree_levels))
        )

        # Compute maximum horizontal padding offset
        padWidth = np.cumsum(
            mesh.hx.min() *
            np.asarray(octree_levels_padding) *
            2**np.arange(len(octree_levels_padding))
        )

        # Increment the vertical offset
        zOffset = 0
        xyPad = -1
        depth = zmax[-1]
        # Cycle through the Tree levels backward
        for ii in range(len(octree_levels)-1, -1, -1):

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
                expansion_factor = (rOut + xyPad) / rOut
                xLoc = (xyz - centroid)*expansion_factor + centroid

                if mesh.dim == 3:
                    # Create a new triangulated surface
                    tri2D = Delaunay(xLoc[:, :2])
                    F = interpolate.LinearNDInterpolator(tri2D, xLoc[:, 2])
                else:
                    F = interpolate.interp1d(xLoc[:, 0], xLoc[:, 1], fill_value='extrapolate')

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

            newLoc = np.c_[xy[indexTri != -1], z]

            # Only keep points within max_distance
            tree = cKDTree(xyz)
            r, ind = tree.query(newLoc)

            # Apply vertical padding for current octree level
            dim = mesh.dim - 1
            zOffset = 0
            while zOffset < depth:
                indIn = r < (max_distance + padWidth[ii])
                nnz = int(np.sum(indIn))
                if nnz > 0:
                    mesh.insert_cells(
                        np.c_[
                                newLoc[indIn, :dim],
                                newLoc[indIn, -1]-zOffset],
                        np.ones(nnz)*mesh.max_level-ii,
                        finalize=False
                    )

                zOffset += dz

            depth -= dz * octree_levels[ii]

        if finalize:
            mesh.finalize()

    elif method.lower() == 'box':

        # Define the data extent [bottom SW, top NE]
        bsw = np.min(xyz, axis=0)
        tne = np.max(xyz, axis=0)

        hx = mesh.hx.min()

        if mesh.dim == 2:
            hz = mesh.hy.min()
        else:
            hz = mesh.hz.min()

        # Pre-calculate max depth of each level
        zmax = np.cumsum(
            hz * np.asarray(octree_levels) *
            2**np.arange(len(octree_levels))
        )

        if mesh.dim == 2:
            # Pre-calculate outer extent of each level
            padWidth = np.cumsum(
                    mesh.hx.min() *
                    np.asarray(octree_levels_padding) *
                    2**np.arange(len(octree_levels_padding))
                )

            # Make a list of outer limits
            BSW = [
                bsw - np.r_[padWidth[ii], zmax[ii]]
                for ii, (octZ, octXY) in enumerate(
                        zip(octree_levels, octree_levels_padding)
                )
            ]

            TNE = [
                tne + np.r_[padWidth[ii], zmax[ii]]
                for ii, (octZ, octXY) in enumerate(
                    zip(octree_levels, octree_levels_padding)
                )
            ]

        else:
            hy = mesh.hy.min()

            # Pre-calculate outer X extent of each level
            padWidth_x = np.cumsum(
                    hx * np.asarray(octree_levels_padding) *
                    2**np.arange(len(octree_levels_padding))
                )

            # Pre-calculate outer Y extent of each level
            padWidth_y = np.cumsum(
                    hy * np.asarray(octree_levels_padding) *
                    2**np.arange(len(octree_levels_padding))
                )

            # Make a list of outer limits
            BSW = [
                bsw - np.r_[padWidth_x[ii], padWidth_y[ii], zmax[ii]]
                for ii, (octZ, octXY) in enumerate(
                        zip(octree_levels, octree_levels_padding)
                )
            ]

            TNE = [
                tne + np.r_[padWidth_x[ii], padWidth_y[ii], zmax[ii]]
                for ii, (octZ, octXY) in enumerate(
                    zip(octree_levels, octree_levels_padding)
                )
            ]

        def inBox(cell):

            xyz = cell.center

            for ii, (nC, bsw, tne) in enumerate(zip(octree_levels, BSW, TNE)):

                if np.all([xyz > bsw, xyz < tne]):
                    return mesh.max_level-ii

            return cell._level

        mesh.refine(inBox, finalize=finalize)

    else:
        raise NotImplementedError(
            "Only method= 'radial', 'surface'"
            " or 'box' have been implemented"
        )

    return mesh


def active_from_xyz(mesh, xyz, grid_reference='CC', method='linear'):
    """Returns an active cell index array below a surface

    Get active cells in the `mesh` that are below the surface create by
    interpolating over the last dimension of the input points. This method will
    uses scipy's interpolation routine to interpolate between input points. This
    will use a nearest neighbour interpolation for cell values outside the convex
    hull of points.

    For `grid_reference='CC'`, this will check if the center of a given cell in
    the mesh is below the interpolation surface to determine if that cell is
    active.

    For `grid_reference='N'`, this will check if **every** node of a given cell
    in the mesh is below the interpolation surface.

    For the

    Parameters
    ----------

    mesh : TensorMesh or TreeMesh or CylMesh
        Mesh object, (if CylMesh: mesh must be symmetric).
    xyz : numpy.ndarray
        Points coordinates shape (*, mesh.dim).
    grid_reference : {'CC', 'N'}
        Use cell coordinates from cells-center 'CC' or nodes 'N'.
    method : {'linear', 'nearest'}
        Interpolation method for the xyz points.

    Returns
    -------

    active : numpy.ndarray
        1D mask array of `bool` for the active cells below xyz.

    Examples
    --------

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from discretize import TensorMesh
        from discretize.utils import active_from_xyz

        mesh = TensorMesh([5, 5])
        topo_func = lambda x: -3*(x-0.2)*(x-0.8)+.5
        topo_points = np.linspace(0, 1)
        topo_vals = topo_func(topo_points)

        active_cc = active_from_xyz(mesh, np.c_[topo_points, topo_vals], grid_reference='CC')
        ax = plt.subplot(121)
        mesh.plotImage(active_cc, ax=ax)
        mesh.plotGrid(centers=True, ax=ax)
        ax.plot(np.linspace(0,1), topo_func(np.linspace(0,1)), color='C3')
        ax.set_title("CC")

        active_n = active_from_xyz(mesh, np.c_[topo_points, topo_vals], grid_reference='N')
        ax = plt.subplot(122)
        mesh.plotImage(active_n, ax=ax)
        mesh.plotGrid(nodes=True, ax=ax)
        ax.plot(np.linspace(0,1), topo_func(np.linspace(0,1)), color='C3')
        ax.set_title("N")
        plt.show()

    """
    if isinstance(mesh, discretize.CylMesh) and not mesh.isSymmetric:
        raise NotImplementedError('Unsymmetric CylMesh is not yet supported')

    if not isinstance(mesh, (discretize.TensorMesh, discretize.TreeMesh, discretize.CylMesh)):
        raise TypeError("Mesh must be either TensorMesh, TreeMesh, or Symmetric CylMesh")

    if grid_reference not in ["N", "CC"]:
        raise ValueError("Value of grid_reference must be 'N' (nodal) or 'CC' (cell center)")

    dim = mesh.dim - 1

    if mesh.dim == 3:
        if xyz.shape[1] != 3:
            raise ValueError("xyz locations of shape (*, 3) required for 3D mesh")
        if method == 'linear':
            tri2D = Delaunay(xyz[:, :2])
            z_interpolate = interpolate.LinearNDInterpolator(tri2D, xyz[:, 2])
        else:
            z_interpolate = interpolate.NearestNDInterpolator(xyz[:, :2], xyz[:, 2])
    elif mesh.dim == 2:
        if xyz.shape[1] != 2:
            raise ValueError("xyz locations of shape (*, 2) required for 2D mesh")
        z_interpolate = interpolate.interp1d(
            xyz[:, 0], xyz[:, 1], bounds_error=False, fill_value=np.nan, kind=method
        )
    else:
        if xyz.ndim != 1:
            raise ValueError("xyz locations of shape (*, ) required for 1D mesh")

    if grid_reference == 'CC':
        locations = mesh.gridCC

        if mesh.dim == 1:
            active = np.zeros(mesh.nC, dtype='bool')
            active[np.searchsorted(mesh.vectorCCx, xyz).max():] = True
            return active

    elif grid_reference == 'N':

        if mesh.dim == 3:
            locations = np.vstack([
                mesh.gridCC + (np.c_[-1, 1, 1][:, None] * mesh.h_gridded / 2.).squeeze(),
                mesh.gridCC + (np.c_[-1, -1, 1][:, None] * mesh.h_gridded / 2.).squeeze(),
                mesh.gridCC + (np.c_[1, 1, 1][:, None] * mesh.h_gridded / 2.).squeeze(),
                mesh.gridCC + (np.c_[1, -1, 1][:, None] * mesh.h_gridded / 2.).squeeze()
            ])

        elif mesh.dim == 2:
            locations = np.vstack([
                mesh.gridCC + (np.c_[-1, 1][:, None] * mesh.h_gridded / 2.).squeeze(),
                mesh.gridCC + (np.c_[1, 1][:, None] * mesh.h_gridded / 2.).squeeze(),
            ])

        else:
            active = np.zeros(mesh.nC, dtype='bool')
            active[np.searchsorted(mesh.vectorNx, xyz).max():] = True

            return active

    # Interpolate z values on CC or N
    z_xyz = z_interpolate(locations[:, :-1]).squeeze()

    # Apply nearest neighbour if in extrapolation
    ind_nan = np.isnan(z_xyz)
    if any(ind_nan):
        tree = cKDTree(xyz)
        _, ind = tree.query(locations[ind_nan, :])
        z_xyz[ind_nan] = xyz[ind, dim]

    # Create an active bool of all True
    active = np.all(
        (locations[:, dim] < z_xyz).reshape((mesh.nC, -1), order='F'), axis=1
    )

    return active.ravel()
