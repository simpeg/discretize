"""Useful tools for working with meshes."""
import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp

from discretize.utils.code_utils import is_scalar
from scipy.spatial import cKDTree, Delaunay
from scipy import interpolate
import discretize
from discretize.utils.code_utils import deprecate_function
import warnings

num_types = [int, float]


def random_model(shape, seed=None, anisotropy=None, its=100, bounds=None):
    """Create random tensor model.

    Creates a random tensor model by convolving a kernel function with a
    uniformly distributed model. The user specifies the number of cells
    along the x, (y and z) directions with the input argument *shape* and
    the function outputs a tensor model with the same shape. Afterwards,
    the user may use the :py:func:`~discretize.utils.mkvc` function
    to convert the tensor to a vector which can be plotting on a
    corresponding tensor mesh.

    Parameters
    ----------
    shape : (dim) tuple of int
        shape of the model.
    seed : int, optional
        pick which model to produce, prints the seed if you don't choose
    anisotropy : numpy.ndarray, optional
        this is the kernel that is convolved with the model
    its : int, optional
        number of smoothing iterations
    bounds : list, optional
        Lower and upper bounds on the model. Has the form [lower_bound, upper_bound].

    Returns
    -------
    numpy.ndarray
        A random generated model whose shape was specified by the input parameter *shape*

    Examples
    --------
    Here, we generate a random model for a 2D tensor mesh and plot.

    >>> from discretize import TensorMesh
    >>> from discretize.utils import random_model, mkvc
    >>> import matplotlib as mpl
    >>> import matplotlib.pyplot as plt

    >>> h = [(1., 50)]
    >>> vmin, vmax = 0., 1.
    >>> mesh = TensorMesh([h, h])

    >>> model = random_model(mesh.shape_cells, seed=4, bounds=[vmin, vmax])

    >>> fig = plt.figure(figsize=(5, 4))
    >>> ax = plt.subplot(111)
    >>> im, = mesh.plot_image(model, grid=False, ax=ax, clim=[vmin, vmax])
    >>> cbar = plt.colorbar(im)
    >>> ax.set_title('Random Tensor Model')
    >>> plt.show()
    """
    if bounds is None:
        bounds = [0, 1]

    if seed is None:
        seed = np.random.randint(1e3)
        print("Using a seed of: ", seed)

    if type(shape) in num_types:
        shape = (shape,)  # make it a tuple for consistency

    np.random.seed(seed)
    mr = np.random.rand(*shape)
    if anisotropy is None:
        if len(shape) == 1:
            smth = np.array([1, 10.0, 1], dtype=float)
        elif len(shape) == 2:
            smth = np.array([[1, 7, 1], [2, 10, 2], [1, 7, 1]], dtype=float)
        elif len(shape) == 3:
            kernal = np.array([1, 4, 1], dtype=float).reshape((1, 3))
            smth = np.array(
                sp.kron(sp.kron(kernal, kernal.T).todense()[:], kernal).todense()
            ).reshape((3, 3, 3))
    else:
        if len(anisotropy.shape) != len(shape):
            raise ValueError("Anisotropy must be the same shape.")
        smth = np.array(anisotropy, dtype=float)

    smth = smth / smth.sum()  # normalize
    mi = mr
    for _i in range(its):
        mi = ndi.convolve(mi, smth)

    # scale the model to live between the bounds.
    mi = (mi - mi.min()) / (mi.max() - mi.min())  # scaled between 0 and 1
    mi = mi * (bounds[1] - bounds[0]) + bounds[0]

    return mi


def unpack_widths(value):
    """Unpack a condensed representation of cell widths or time steps.

    For a list of numbers, if the same value is repeat or expanded by a constant
    factor, it may be represented in a condensed form using list of floats
    and/or tuples. **unpack_widths** takes a list of floats and/or tuples in
    condensed form, e.g.:

        [ float, (cellSize, numCell), (cellSize, numCell, factor) ]

    and expands the representation to a list containing all widths in order. That is:

        [ w1, w2, w3, ..., wn ]

    Parameters
    ----------
    value : list of float and/or tuple
        The list of floats and/or tuples that are to be unpacked

    Returns
    -------
    numpy.ndarray
        The unpacked list with all widths in order

    Examples
    --------
    Time stepping for time-domain codes can be represented in condensed form, e.g.:

    >>> from discretize.utils import unpack_widths
    >>> dt = [ (1e-5, 10), (1e-4, 4), 1e-3 ]

    The above means to take 10 steps at a step width of 1e-5 s and then
    4 more at 1e-4 s, and then one step of 1e-3 s. When unpacked, the output is
    of length 15 and is given by:

    >>> unpack_widths(dt)
    array([1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05,
           1.e-05, 1.e-05, 1.e-04, 1.e-04, 1.e-04, 1.e-04, 1.e-03])

    Each axis of a tensor mesh can also be defined as a condensed list of floats
    and/or tuples. When a third number is defined in any tuple, the width value
    is successively expanded by that factor, e.g.:

    >>> dt = [ 6., 8., (10.0, 3), (8.0, 4, 2.) ]
    >>> unpack_widths(dt)
    array([  6.,   8.,  10.,  10.,  10.,  16.,  32.,  64., 128.])
    """
    if type(value) is not list:
        raise Exception("unpack_widths must be a list of scalars and tuples.")

    proposed = []
    for v in value:
        if is_scalar(v):
            proposed += [float(v)]
        elif type(v) is tuple and len(v) == 2:
            proposed += [float(v[0])] * int(v[1])
        elif type(v) is tuple and len(v) == 3:
            start = float(v[0])
            num = int(v[1])
            factor = float(v[2])
            pad = ((np.ones(num) * np.abs(factor)) ** (np.arange(num) + 1)) * start
            if factor < 0:
                pad = pad[::-1]
            proposed += pad.tolist()
        else:
            raise Exception(
                "unpack_widths must contain only scalars and len(2) or len(3) tuples."
            )

    return np.array(proposed)


def closest_points_index(mesh, pts, grid_loc="CC", **kwargs):
    """Find the indicies for the nearest grid location for a set of points.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        An instance of *discretize.base.BaseMesh*
    pts : (n, dim) numpy.ndarray
        Points to query.
    grid_loc : {'CC', 'N', 'Fx', 'Fy', 'Fz', 'Ex', 'Ex', 'Ey', 'Ez'}
        Specifies the grid on which points are being moved to.

    Returns
    -------
    (n ) numpy.ndarray of int
        Vector of length *n* containing the indicies for the closest
        respective cell center, node, face or edge.

    Examples
    --------
    Here we define a set of random (x, y) locations and find the closest
    cell centers and nodes on a mesh.

    >>> from discretize import TensorMesh
    >>> from discretize.utils import closest_points_index
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> h = 2*np.ones(5)
    >>> mesh = TensorMesh([h, h], x0='00')

    Define some random locations, grid cell centers and grid nodes,

    >>> xy_random = np.random.uniform(0, 10, size=(4,2))
    >>> xy_centers = mesh.cell_centers
    >>> xy_nodes = mesh.nodes

    Find indicies of closest cell centers and nodes,

    >>> ind_centers = closest_points_index(mesh, xy_random, 'CC')
    >>> ind_nodes = closest_points_index(mesh, xy_random, 'N')

    Plot closest cell centers and nodes

    >>> fig = plt.figure(figsize=(5, 5))
    >>> ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    >>> mesh.plot_grid(ax=ax)
    >>> ax.scatter(xy_random[:, 0], xy_random[:, 1], 50, 'k')
    >>> ax.scatter(xy_centers[ind_centers, 0], xy_centers[ind_centers, 1], 50, 'r')
    >>> ax.scatter(xy_nodes[ind_nodes, 0], xy_nodes[ind_nodes, 1], 50, 'b')
    >>> plt.show()
    """
    if "gridLoc" in kwargs:
        warnings.warn(
            "The gridLoc keyword argument has been deprecated, please use grid_loc. "
            "This will be removed in discretize 1.0.0",
            FutureWarning,
        )
        grid_loc = kwargs["gridLoc"]
    warnings.warn(
        "The closest_points_index utilty function has been moved to be a method of "
        "a class object. Please access it as mesh.closest_points_index(). This will "
        "be removed in a future version of discretize",
        DeprecationWarning,
    )
    return mesh.closest_points_index(pts, grid_loc=grid_loc, discard=True)


def extract_core_mesh(xyzlim, mesh, mesh_type="tensor"):
    """Extract the core mesh from a global mesh.

    Parameters
    ----------
    xyzlim : (dim, 2) numpy.ndarray
        2D array defining the x, y and z cutoffs for the core mesh region. Each
        row contains the minimum and maximum limit for the x, y and z axis,
        respectively.
    mesh : discretize.TensorMesh
        The mesh
    mesh_type : str, optional
        Unused currently

    Returns
    -------
    tuple: (**active_index**, **core_mesh**)
        **active_index** is a boolean array that maps from the global the mesh
        to core mesh. **core_mesh** is a *discretize.base.BaseMesh* object representing
        the core mesh.

    Examples
    --------
    Here, we define a 2D tensor mesh that has both a core region and padding.
    We use the function **extract_core_mesh** to return a mesh which contains
    only the core region.

    >>> from discretize.utils import extract_core_mesh
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib as mpl
    >>> mpl.rcParams.update({"font.size": 14})

    Form a mesh of a uniform cube

    >>> h = [(1., 5, -1.5), (1., 20), (1., 5, 1.5)]
    >>> mesh = TensorMesh([h, h], origin='CC')

    Plot original mesh

    >>> fig = plt.figure(figsize=(7, 7))
    >>> ax = fig.add_subplot(111)
    >>> mesh.plot_grid(ax=ax)
    >>> ax.set_title('Original Tensor Mesh')
    >>> plt.show()

    Set the limits for the cutoff of the core mesh (dim, 2)

    >>> xlim = np.c_[-10., 10]
    >>> ylim = np.c_[-10., 10]
    >>> core_limits = np.r_[xlim, ylim]

    Extract indices of core mesh cells and the core mesh, then plot

    >>> core_ind, core_mesh = extract_core_mesh(core_limits, mesh)
    >>> fig = plt.figure(figsize=(4, 4))
    >>> ax = fig.add_subplot(111)
    >>> core_mesh.plot_grid(ax=ax)
    >>> ax.set_title('Core Mesh')
    >>> plt.show()
    """
    if not isinstance(mesh, discretize.TensorMesh):
        raise Exception("Only implemented for class TensorMesh")

    if mesh.dim == 1:
        xyzlim = xyzlim.flatten()
        xmin, xmax = xyzlim[0], xyzlim[1]

        xind = np.logical_and(mesh.cell_centers_x > xmin, mesh.cell_centers_x < xmax)

        xc = mesh.cell_centers_x[xind]

        hx = mesh.h[0][xind]

        origin = [xc[0] - hx[0] * 0.5]

        meshCore = discretize.TensorMesh([hx], origin=origin)

        actind = (mesh.cell_centers > xmin) & (mesh.cell_centers < xmax)

    elif mesh.dim == 2:
        xmin, xmax = xyzlim[0, 0], xyzlim[0, 1]
        ymin, ymax = xyzlim[1, 0], xyzlim[1, 1]

        xind = np.logical_and(mesh.cell_centers_x > xmin, mesh.cell_centers_x < xmax)
        yind = np.logical_and(mesh.cell_centers_y > ymin, mesh.cell_centers_y < ymax)

        xc = mesh.cell_centers_x[xind]
        yc = mesh.cell_centers_y[yind]

        hx = mesh.h[0][xind]
        hy = mesh.h[1][yind]

        origin = [xc[0] - hx[0] * 0.5, yc[0] - hy[0] * 0.5]

        meshCore = discretize.TensorMesh([hx, hy], origin=origin)

        actind = (
            (mesh.cell_centers[:, 0] > xmin)
            & (mesh.cell_centers[:, 0] < xmax)
            & (mesh.cell_centers[:, 1] > ymin)
            & (mesh.cell_centers[:, 1] < ymax)
        )

    elif mesh.dim == 3:
        xmin, xmax = xyzlim[0, 0], xyzlim[0, 1]
        ymin, ymax = xyzlim[1, 0], xyzlim[1, 1]
        zmin, zmax = xyzlim[2, 0], xyzlim[2, 1]

        xind = np.logical_and(mesh.cell_centers_x > xmin, mesh.cell_centers_x < xmax)
        yind = np.logical_and(mesh.cell_centers_y > ymin, mesh.cell_centers_y < ymax)
        zind = np.logical_and(mesh.cell_centers_z > zmin, mesh.cell_centers_z < zmax)

        xc = mesh.cell_centers_x[xind]
        yc = mesh.cell_centers_y[yind]
        zc = mesh.cell_centers_z[zind]

        hx = mesh.h[0][xind]
        hy = mesh.h[1][yind]
        hz = mesh.h[2][zind]

        origin = [xc[0] - hx[0] * 0.5, yc[0] - hy[0] * 0.5, zc[0] - hz[0] * 0.5]

        meshCore = discretize.TensorMesh([hx, hy, hz], origin=origin)

        actind = (
            (mesh.cell_centers[:, 0] > xmin)
            & (mesh.cell_centers[:, 0] < xmax)
            & (mesh.cell_centers[:, 1] > ymin)
            & (mesh.cell_centers[:, 1] < ymax)
            & (mesh.cell_centers[:, 2] > zmin)
            & (mesh.cell_centers[:, 2] < zmax)
        )

    else:
        raise Exception("Not implemented!")

    return actind, meshCore


def mesh_builder_xyz(
    xyz,
    h,
    padding_distance=None,
    base_mesh=None,
    depth_core=None,
    expansion_factor=1.3,
    mesh_type="tensor",
):
    """Generate a tensor or tree mesh using a cloud of points.

    For a cloud of (x,y[,z]) locations and specified minimum cell widths
    (hx,hy,[hz]), this function creates a tensor or a tree mesh.
    The lateral extent of the core region is determine by the cloud of points.
    Other properties of the mesh can be defined automatically or by the user.
    If *base_mesh* is an instance of :class:`~discretize.TensorMesh` or
    :class:`~discretize.TreeMesh`, the core cells will be centered
    on the underlying mesh to reduce interpolation errors.

    Parameters
    ----------
    xyz : (n, dim) numpy.ndarray
        Location points
    h : (dim ) list
        Cell size(s) for the core mesh
    padding_distance : list, optional
        Padding distances [[W,E], [N,S], [Down,Up]], default is no padding.
    base_mesh : discretize.TensorMesh or discretize.TreeMesh, optional
        discretize mesh used to center the core mesh
    depth_core : float, optional
        Depth of core mesh below xyz
    expansion_factor : float. optional
        Expansion factor for padding cells. Ignored if *mesh_type* = *tree*
    mesh_type : {'tensor', 'tree'}
        Specify output mesh type

    Returns
    -------
    discretize.TensorMesh or discretize.TreeMesh
        Mesh of type specified by *mesh_type*

    Examples
    --------
    >>> import discretize
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> xy_loc = np.random.randn(8,2)
    >>> mesh = discretize.utils.mesh_builder_xyz(
    ...     xy_loc, [0.1, 0.1], depth_core=0.5,
    ...     padding_distance=[[1,2], [1,0]],
    ...     mesh_type='tensor',
    ... )

    >>> axs = plt.subplot()
    >>> mesh.plot_image(mesh.cell_volumes, grid=True, ax=axs)
    >>> axs.scatter(xy_loc[:,0], xy_loc[:,1], 15, c='w', zorder=3)
    >>> axs.set_aspect('equal')
    >>> plt.show()
    """
    if mesh_type.lower() not in ["tensor", "tree"]:
        raise ValueError("Revise mesh_type. Only TENSOR | TREE mesh are implemented")

    if padding_distance is None:
        padding_distance = [[0, 0], [0, 0], [0, 0]]
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

    if mesh_type.lower() == "tensor":
        # Figure out padding cells from distance
        def expand(dx, pad):
            length = 0
            nc = 0
            while length < pad:
                nc += 1
                length = np.sum(dx * expansion_factor ** (np.asarray(range(nc)) + 1))

            return nc

        # Define h along each dimension
        h_dim = []
        nC_origin = []
        for dim in range(xyz.shape[1]):
            h_dim += [
                [
                    (
                        h[dim],
                        expand(h[dim], padding_distance[dim][0]),
                        -expansion_factor,
                    ),
                    (h[dim], nC[dim]),
                    (
                        h[dim],
                        expand(h[dim], padding_distance[dim][1]),
                        expansion_factor,
                    ),
                ]
            ]

            nC_origin += [h_dim[-1][0][1]]

        # Create mesh
        mesh = discretize.TensorMesh(h_dim)

    elif mesh_type.lower() == "tree":
        # Figure out full extent required from input
        h_dim = []
        nC_origin = []
        for ii, _cc in enumerate(nC):
            extent = limits[ii][0] - limits[ii][1] + np.sum(padding_distance[ii])

            # Number of cells at the small octree level
            maxLevel = int(np.log2(extent / h[ii])) + 1
            h_dim += [np.ones(2**maxLevel) * h[ii]]

        # Define the mesh and origin
        mesh = discretize.TreeMesh(h_dim)

        for ii, _cc in enumerate(nC):
            core = limits[ii][0] - limits[ii][1]

            nC_origin += [int(np.ceil((mesh.h[ii].sum() - core) / h[ii] / 2))]

    # Set origin
    origin = []
    for ii, hi in enumerate(mesh.h):
        origin += [limits[ii][1] - np.sum(hi[: nC_origin[ii]])]

    mesh.origin = np.hstack(origin)

    # Shift mesh if global mesh is used based on closest to centroid
    axis = ["x", "y", "z"]
    if base_mesh is not None:
        for dim in range(base_mesh.dim):
            cc_base = getattr(
                base_mesh,
                "cell_centers_{orientation}".format(orientation=axis[dim]),
            )

            cc_local = getattr(
                mesh, "cell_centers_{orientation}".format(orientation=axis[dim])
            )

            shift = (
                cc_base[np.max([np.searchsorted(cc_base, center[dim]) - 1, 0])]
                - cc_local[np.max([np.searchsorted(cc_local, center[dim]) - 1, 0])]
            )

            origin[dim] += shift

            mesh.origin = np.hstack(origin)

    return mesh


def refine_tree_xyz(
    mesh,
    xyz,
    method="radial",
    octree_levels=(1, 1, 1),
    octree_levels_padding=None,
    finalize=False,
    min_level=0,
    max_distance=np.inf,
):
    """Refine region within a :class:`~discretize.TreeMesh`.

    This function refines the specified region of a tree mesh using
    one of several methods. These are summarized below:

    **radial:** refines based on radial distances from a set of xy[z] locations.
    Consider a tree mesh whose smallest cell size has a width of *h* . And
    *octree_levels = [nc1, nc2, nc3, ...]* . Within a distance of *nc1 x h*
    from any of the points supplied, the smallest cell size is used. Within a distance of
    *nc2 x (2h)* , the cells will have a width of *2h* . Within a distance of *nc3 x (4h)* ,
    the cells will have a width of *4h* . Etc...

    **surface:** refines downward from a triangulated surface.
    Consider a tree mesh whose smallest cell size has a width of *h*. And
    *octree_levels = [nc1, nc2, nc3, ...]* . Within a downward distance of *nc1 x h*
    from the topography (*xy[z]* ) supplied, the smallest cell size is used. The
    topography is triangulated if the points supplied are coarser than the cell
    size. No refinement is done above the topography. Within a vertical distance of
    *nc2 x (2h)* , the cells will have a width of *2h* . Within a vertical distance
    of *nc3 x (4h)* , the cells will have a width of *4h* . Etc...

    **box:** refines inside the convex hull defined by the xy[z] locations.
    Consider a tree mesh whose smallest cell size has a width of *h*. And
    *octree_levels = [nc1, nc2, nc3, ...]* . Within the convex hull defined by *xyz* ,
    the smallest cell size is used. Within a distance of *nc2 x (2h)* from that convex
    hull, the cells will have a width of *2h* . Within a distance of *nc3 x (4h)* ,
    the cells will have a width of *4h* . Etc...

    .. deprecated:: 0.9.0
          `refine_tree_xyz` will be removed in a future version of discretize. It is
          replaced by `discretize.TreeMesh.refine_surface`, `discretize.TreeMesh.refine_bounding_box`,
          and `discretize.TreeMesh.refine_points`, to separate the calling convetions,
          and improve the individual documentation. Those methods are more explicit
          about which levels of the TreeMesh you are refining, and provide more
          flexibility for padding cells in each dimension.

    Parameters
    ----------
    mesh : discretize.TreeMesh
        The tree mesh object to be refined
    xyz : numpy.ndarray
        2D array of points (n, dim)
    method : {'radial', 'surface', 'box'}
        Method used to refine the mesh based on xyz locations.

        - *radial:* Based on radial distance xy[z] and cell centers
        - *surface:* Refines downward from a triangulated surface
        - *box:* Inside limits defined by outer xy[z] locations

    octree_levels : list of int, optional
        Minimum number of cells around points in each *k* octree level
        starting from the smallest cells size; i.e. *[nc(k), nc(k-1), ...]* .
        Note that you *can* set entries to 0; e.g. you don't want to discretize
        using the smallest cell size.
    octree_levels_padding : list of int, optional
        Padding cells added to extend the region of refinement at each level.
        Used for *method = surface* and *box*. Has the form *[nc(k), nc(k-1), ...]*
    finalize : bool, optional
        Finalize the tree mesh.
    min_level : int, optional
        Sets the largest cell size allowed in the mesh. The default (*0*),
        allows the largest cell size to be used.
    max_distance : float
        Maximum refinement distance from xy[z] locations.
        Used if *method* = "surface" to reduce interpolation distance

    Returns
    -------
    discretize.TreeMesh
        The refined tree mesh

    See Also
    --------
    discretize.TreeMesh.refine_surface
        Recommended to use instead of this function for the `surface` option.
    discretize.TreeMesh.refine_bounding_box
        Recommended to use instead of this function for the `box` option.
    discretize.TreeMesh.refine_points
        Recommended to use instead of this function for the `radial` option.

    Examples
    --------
    Here we use the **refine_tree_xyz** function refine a tree mesh
    based on topography as well as a cluster of points.

    >>> from discretize import TreeMesh
    >>> from discretize.utils import mkvc, refine_tree_xyz
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> dx = 5  # minimum cell width (base mesh cell width) in x
    >>> dy = 5  # minimum cell width (base mesh cell width) in y
    >>> x_length = 300.0  # domain width in x
    >>> y_length = 300.0  # domain width in y

    Compute number of base mesh cells required in x and y

    >>> nbcx = 2 ** int(np.round(np.log(x_length / dx) / np.log(2.0)))
    >>> nbcy = 2 ** int(np.round(np.log(y_length / dy) / np.log(2.0)))

    Define the base mesh

    >>> hx = [(dx, nbcx)]
    >>> hy = [(dy, nbcy)]
    >>> mesh = TreeMesh([hx, hy], x0="CC")

    Refine surface topography

    >>> xx = mesh.vectorNx
    >>> yy = -3 * np.exp((xx ** 2) / 100 ** 2) + 50.0
    >>> pts = np.c_[mkvc(xx), mkvc(yy)]
    >>> mesh = refine_tree_xyz(
    ...     mesh, pts, octree_levels=[2, 4], method="surface", finalize=False
    ... )

    Refine mesh near points

    >>> xx = np.array([-10.0, 10.0, 10.0, -10.0])
    >>> yy = np.array([-40.0, -40.0, -60.0, -60.0])
    >>> pts = np.c_[mkvc(xx), mkvc(yy)]
    >>> mesh = refine_tree_xyz(
    ...     mesh, pts, octree_levels=[4, 2], method="radial", finalize=True
    ... )

    Plot the mesh

    >>> fig = plt.figure(figsize=(6, 6))
    >>> ax = fig.add_subplot(111)
    >>> mesh.plotGrid(ax=ax)
    >>> ax.set_xbound(mesh.x0[0], mesh.x0[0] + np.sum(mesh.hx))
    >>> ax.set_ybound(mesh.x0[1], mesh.x0[1] + np.sum(mesh.hy))
    >>> ax.set_title("QuadTree Mesh")
    >>> plt.show()
    """
    if octree_levels_padding is not None:
        if len(octree_levels_padding) != len(octree_levels):
            raise ValueError(
                "'octree_levels_padding' must be the length %i" % len(octree_levels)
            )

    else:
        octree_levels_padding = np.zeros_like(octree_levels)
    octree_levels = np.asarray(octree_levels)
    octree_levels_padding = np.asarray(octree_levels_padding)

    # levels = mesh.max_level - np.arange(len(octree_levels))
    # non_zeros = octree_levels != 0
    # levels = levels[non_zeros]

    # Trigger different refine methods
    if method.lower() == "radial":
        # padding = octree_levels[non_zeros]
        # mesh.refine_points(xyz, levels, padding, finalize=finalize,)
        warnings.warn(
            "The radial option is deprecated as of `0.9.0` please update your code to "
            "use the `TreeMesh.refine_points` functionality. It will be removed in a "
            "future version of discretize.",
            DeprecationWarning,
        )

        # Compute the outer limits of each octree level
        rMax = np.cumsum(
            mesh.h[0].min() * octree_levels * 2 ** np.arange(len(octree_levels))
        )
        rs = np.ones(xyz.shape[0])
        level = np.ones(xyz.shape[0], dtype=np.int32)
        for ii, _nC in enumerate(octree_levels):
            # skip "zero" sized balls
            if rMax[ii] > 0:
                mesh.refine_ball(
                    xyz, rs * rMax[ii], level * (mesh.max_level - ii), finalize=False
                )
        if finalize:
            mesh.finalize()

    elif method.lower() == "surface":
        warnings.warn(
            "The surface option is deprecated as of `0.9.0` please update your code to "
            "use the `TreeMesh.refine_surface` functionality. It will be removed in a "
            "future version of discretize.",
            DeprecationWarning,
        )
        # padding = np.zeros((len(octree_levels), mesh.dim))
        # padding[:, -1] = np.maximum(octree_levels - 1, 0)
        # padding[:, :-1] = octree_levels_padding[:, None]
        # padding = padding[non_zeros]
        # mesh.refine_surface(xyz, levels, padding, finalize=finalize, pad_down=True, pad_up=False)

        # Compute centroid
        centroid = np.mean(xyz, axis=0)

        if mesh.dim == 2:
            rOut = np.abs(centroid[0] - xyz).max()
            hz = mesh.h[1].min()
        else:
            # Largest outer point distance
            rOut = np.linalg.norm(
                np.r_[
                    np.abs(centroid[0] - xyz[:, 0]).max(),
                    np.abs(centroid[1] - xyz[:, 1]).max(),
                ]
            )
            hz = mesh.h[2].min()

        # Compute maximum depth of refinement
        zmax = np.cumsum(hz * octree_levels * 2 ** np.arange(len(octree_levels)))

        # Compute maximum horizontal padding offset
        padWidth = np.cumsum(
            mesh.h[0].min()
            * octree_levels_padding
            * 2 ** np.arange(len(octree_levels_padding))
        )

        # Increment the vertical offset
        zOffset = 0
        xyPad = -1
        depth = zmax[-1]
        # Cycle through the Tree levels backward
        for ii in range(len(octree_levels) - 1, -1, -1):
            dx = mesh.h[0].min() * 2**ii

            if mesh.dim == 3:
                dy = mesh.h[1].min() * 2**ii
                dz = mesh.h[2].min() * 2**ii
            else:
                dz = mesh.h[1].min() * 2**ii

            # Increase the horizontal extent of the surface
            if xyPad != padWidth[ii]:
                xyPad = padWidth[ii]

                # Calculate expansion for padding XY cells
                expansion_factor = (rOut + xyPad) / rOut
                xLoc = (xyz - centroid) * expansion_factor + centroid

                if mesh.dim == 3:
                    # Create a new triangulated surface
                    tri2D = Delaunay(xLoc[:, :2])
                    F = interpolate.LinearNDInterpolator(tri2D, xLoc[:, 2])
                else:
                    F = interpolate.interp1d(
                        xLoc[:, 0], xLoc[:, 1], fill_value="extrapolate"
                    )

            limx = np.r_[xLoc[:, 0].max(), xLoc[:, 0].min()]
            nCx = int(np.ceil((limx[0] - limx[1]) / dx))

            if mesh.dim == 3:
                limy = np.r_[xLoc[:, 1].max(), xLoc[:, 1].min()]
                nCy = int(np.ceil((limy[0] - limy[1]) / dy))

                # Create a grid at the octree level in xy
                CCx, CCy = np.meshgrid(
                    np.linspace(limx[1], limx[0], nCx),
                    np.linspace(limy[1], limy[0], nCy),
                )

                xy = np.c_[CCx.reshape(-1), CCy.reshape(-1)]

                # Only keep points within triangulation
                indexTri = tri2D.find_simplex(xy)

            else:
                xy = np.linspace(limx[1], limx[0], nCx)
                indexTri = np.ones_like(xy, dtype="bool")

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
                        np.c_[newLoc[indIn, :dim], newLoc[indIn, -1] - zOffset],
                        np.ones(nnz) * mesh.max_level - ii,
                        finalize=False,
                    )

                zOffset += dz

            depth -= dz * octree_levels[ii]

        if finalize:
            mesh.finalize()

    elif method.lower() == "box":
        warnings.warn(
            "The box option is deprecated as of `0.9.0` please update your code to "
            "use the `TreeMesh.refine_bounding_box` functionality. It will be removed in a "
            "future version of discretize.",
            DeprecationWarning,
        )
        # padding = np.zeros((len(octree_levels), mesh.dim))
        # padding[:, -1] = np.maximum(octree_levels - 1, 0)
        # padding[:, :-1] = octree_levels_padding[:, None]
        # padding = padding[non_zeros]
        # mesh.refine_bounding_box(xyz, levels, padding, finalize=finalize)

        # Define the data extent [bottom SW, top NE]
        bsw = np.min(xyz, axis=0)
        tne = np.max(xyz, axis=0)

        hs = np.asarray([h.min() for h in mesh.h])
        hx = hs[0]
        hz = hs[-1]

        # Pre-calculate outer extent of each level
        # x_pad
        padWidth = np.cumsum(
            hx * octree_levels_padding * 2 ** np.arange(len(octree_levels))
        )
        if mesh.dim == 3:
            # y_pad
            hy = hs[1]
            padWidth = np.c_[
                padWidth,
                np.cumsum(
                    hy * octree_levels_padding * 2 ** np.arange(len(octree_levels))
                ),
            ]
        # Pre-calculate max depth of each level
        padWidth = np.c_[
            padWidth,
            np.cumsum(
                hz
                * np.maximum(octree_levels - 1, 0)
                * 2 ** np.arange(len(octree_levels))
            ),
        ]

        levels = []
        BSW = []
        TNE = []
        for ii, octZ in enumerate(octree_levels):
            if octZ > 0:
                levels.append(mesh.max_level - ii)
                BSW.append(bsw - padWidth[ii])
                TNE.append(tne + padWidth[ii])

        mesh.refine_box(BSW, TNE, levels, finalize=finalize)

    else:
        raise NotImplementedError(
            "Only method= 'radial', 'surface'" " or 'box' have been implemented"
        )

    return mesh


def active_from_xyz(mesh, xyz, grid_reference="CC", method="linear"):
    """Return boolean array indicating which cells are below surface.

    For a set of locations defining a surface, **active_from_xyz** outputs a
    boolean array indicating which mesh cells like below the surface points.
    This method uses SciPy's interpolation routine to interpolate between
    location points defining the surface. Nearest neighbour interpolation
    is used for cells outside the convex hull of the surface points.

    Parameters
    ----------
    mesh : discretize.TensorMesh or discretize.TreeMesh or discretize.CylindricalMesh
        Mesh object. If *mesh* is a cylindrical mesh, it must be symmetric
    xyz : (N, dim) numpy.ndarray
        Points defining the surface topography.
    grid_reference : {'CC', 'N'}
        Define where the cell is defined relative to surface. Choose between {'CC','N'}

        - If 'CC' is used, cells are active if their centers are below the surface.
        - If 'N' is used, cells are active if they lie entirely below the surface.

    method : {'linear', 'nearest'}
        Interpolation method for locations between the xyz points.

    Returns
    -------
    (n_cells) numpy.ndarray of bool
        1D mask array of *bool* for the active cells below xyz.

    Examples
    --------
    Here we define the active cells below a parabola. We demonstrate the differences
    that appear when using the 'CC' and 'N' options for *reference_grid*.

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from discretize import TensorMesh
    >>> from discretize.utils import active_from_xyz

    Determine active cells for a given mesh and topography

    >>> mesh = TensorMesh([5, 5])
    >>> topo_func = lambda x: -3*(x-0.2)*(x-0.8)+.5
    >>> topo_points = np.linspace(0, 1)
    >>> topo_vals = topo_func(topo_points)
    >>> active_cc = active_from_xyz(mesh, np.c_[topo_points, topo_vals], grid_reference='CC')
    >>> active_n = active_from_xyz(mesh, np.c_[topo_points, topo_vals], grid_reference='N')

    Plot visual representation

    .. collapse:: Expand to show scripting for plot

        >>> ax = plt.subplot(121)
        >>> mesh.plot_image(active_cc, ax=ax)
        >>> mesh.plot_grid(centers=True, ax=ax)
        >>> ax.plot(np.linspace(0,1), topo_func(np.linspace(0,1)), color='C3')
        >>> ax.set_title("CC")
        >>> ax = plt.subplot(122)
        >>> mesh.plot_image(active_n, ax=ax)
        >>> mesh.plot_grid(nodes=True, ax=ax)
        >>> ax.plot(np.linspace(0,1), topo_func(np.linspace(0,1)), color='C3')
        >>> ax.set_title("N")
        >>> plt.show()
    """
    try:
        if not mesh.is_symmetric:
            raise NotImplementedError(
                "Unsymmetric CylindricalMesh is not yet supported"
            )
    except AttributeError:
        pass

    if grid_reference not in ["N", "CC"]:
        raise ValueError(
            "Value of grid_reference must be 'N' (nodal) or 'CC' (cell center)"
        )

    dim = mesh.dim - 1

    if mesh.dim == 3:
        if xyz.shape[1] != 3:
            raise ValueError("xyz locations of shape (*, 3) required for 3D mesh")
        if method == "linear":
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

    if grid_reference == "CC":
        # this should work for all 4 mesh types...
        locations = mesh.cell_centers

        if mesh.dim == 1:
            active = np.zeros(mesh.nC, dtype="bool")
            active[np.searchsorted(mesh.cell_centers_x, xyz).max() :] = True
            return active

    elif grid_reference == "N":
        try:
            # try for Cyl, Tensor, and Tree operations
            if mesh.dim == 3:
                locations = np.vstack(
                    [
                        mesh.cell_centers
                        + (np.c_[-1, 1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                        mesh.cell_centers
                        + (np.c_[-1, -1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                        mesh.cell_centers
                        + (np.c_[1, 1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                        mesh.cell_centers
                        + (np.c_[1, -1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                    ]
                )

            elif mesh.dim == 2:
                locations = np.vstack(
                    [
                        mesh.cell_centers
                        + (np.c_[-1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                        mesh.cell_centers
                        + (np.c_[1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                    ]
                )

            else:
                active = np.zeros(mesh.nC, dtype="bool")
                active[np.searchsorted(mesh.nodes_x, xyz).max() :] = True

                return active
        except AttributeError:
            # Try for Curvilinear Mesh
            gridN = mesh.gridN.reshape((*mesh.vnN, mesh.dim), order="F")
            if mesh.dim == 3:
                locations = np.vstack(
                    [
                        gridN[:-1, 1:, 1:].reshape((-1, mesh.dim), order="F"),
                        gridN[:-1, :-1, 1:].reshape((-1, mesh.dim), order="F"),
                        gridN[1:, 1:, 1:].reshape((-1, mesh.dim), order="F"),
                        gridN[1:, :-1, 1:].reshape((-1, mesh.dim), order="F"),
                    ]
                )
            elif mesh.dim == 2:
                locations = np.vstack(
                    [
                        gridN[:-1, 1:].reshape((-1, mesh.dim), order="F"),
                        gridN[1:, 1:].reshape((-1, mesh.dim), order="F"),
                    ]
                )

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
        (locations[:, dim] < z_xyz).reshape((mesh.nC, -1), order="F"), axis=1
    )

    return active.ravel()


def example_simplex_mesh(rect_shape):
    """Create a simple tetrahedral mesh on a unit cube in 2D or 3D.

    Returns the nodes and connectivity of a triangulated domain on the [0, 1] cube.
    This is not necessarily a good triangulation, just a complete one. This is mostly
    used for testing purposes. In 2D, this discretizes each rectangle into two triangles.
    In 3D, each cube is broken into 6 tetrahedrons.

    Parameters
    ----------
    rect_shape : (dim) array_like of int
        For each dimension, create n+1 nodes along that axis.

    Returns
    -------
    points : (n_points, dim) numpy.ndarray
        array of created nodes
    simplics : (n_cells, dim + 1) numpy.ndarray
        connectivity of nodes for each cell.

    Examples
    --------
    >>> from discretize import SimplexMesh
    >>> from discretize.utils import example_simplex_mesh
    >>> from matplotlib import pyplot as plt
    >>> nodes, simplices = example_simplex_mesh((5, 6))
    >>> mesh = SimplexMesh(nodes, simplices)
    >>> mesh.plot_grid()
    >>> plt.show()
    """
    if len(rect_shape) == 2:
        n1, n2 = rect_shape
        xs, ys = np.mgrid[0 : 1 : (n1 + 1) * 1j, 0 : 1 : (n2 + 1) * 1j]
        points = np.c_[xs.reshape(-1), ys.reshape(-1)]

        node_inds = np.arange((n1 + 1) * (n2 + 1)).reshape((n1 + 1, n2 + 1))

        left_triangs = np.c_[
            node_inds[:-1, :-1].reshape(-1),  # i00
            node_inds[1:, :-1].reshape(-1),  # i10
            node_inds[:-1, 1:].reshape(-1),  # i01
        ]
        right_triangs = np.c_[
            node_inds[1:, 1:].reshape(-1),  # i11
            node_inds[1:, :-1].reshape(-1),  # i10
            node_inds[:-1, 1:].reshape(-1),  # i01
        ]

        simplices = np.r_[left_triangs, right_triangs]
    if len(rect_shape) == 3:
        n1, n2, n3 = rect_shape
        xs, ys, zs = np.mgrid[
            0 : 1 : (n1 + 1) * 1j, 0 : 1 : (n2 + 1) * 1j, 0 : 1 : (n3 + 1) * 1j
        ]
        points = np.c_[xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)]

        node_inds = np.arange((n1 + 1) * (n2 + 1) * (n3 + 1)).reshape(
            (n1 + 1, n2 + 1, n3 + 1)
        )

        a_triangs = np.c_[
            node_inds[1:, :-1, :-1].reshape(-1),  # i100
            node_inds[:-1, :-1, 1:].reshape(-1),  # i001
            node_inds[:-1, 1:, :-1].reshape(-1),  # i010
            node_inds[:-1, :-1, :-1].reshape(-1),  # i000
        ]
        b_triangs = np.c_[
            node_inds[:-1, 1:, 1:].reshape(-1),  # i011
            node_inds[1:, :-1, :-1].reshape(-1),  # i100
            node_inds[:-1, :-1, 1:].reshape(-1),  # i001
            node_inds[:-1, 1:, :-1].reshape(-1),  # i010
        ]
        c_triangs = np.c_[
            node_inds[1:, 1:, :-1].reshape(-1),  # i110
            node_inds[:-1, 1:, 1:].reshape(-1),  # i011
            node_inds[1:, :-1, :-1].reshape(-1),  # i100
            node_inds[:-1, 1:, :-1].reshape(-1),  # i010
        ]
        d_triangs = np.c_[
            node_inds[1:, :-1, 1:].reshape(-1),  # i101
            node_inds[:-1, 1:, 1:].reshape(-1),  # i011
            node_inds[1:, :-1, :-1].reshape(-1),  # i100
            node_inds[:-1, :-1, 1:].reshape(-1),  # i001
        ]
        e_triangs = np.c_[
            node_inds[1:, :-1, 1:].reshape(-1),  # i101
            node_inds[1:, 1:, :-1].reshape(-1),  # i110
            node_inds[:-1, 1:, 1:].reshape(-1),  # i011
            node_inds[1:, 1:, 1:].reshape(-1),  # i111
        ]
        f_triangs = np.c_[
            node_inds[1:, :-1, 1:].reshape(-1),  # i101
            node_inds[1:, 1:, :-1].reshape(-1),  # i110
            node_inds[:-1, 1:, 1:].reshape(-1),  # i011
            node_inds[1:, :-1, :-1].reshape(-1),  # i100
        ]

        simplices = np.r_[
            a_triangs, b_triangs, c_triangs, d_triangs, e_triangs, f_triangs
        ]

    return points, simplices


meshTensor = deprecate_function(
    unpack_widths, "meshTensor", removal_version="1.0.0", future_warn=True
)
closestPoints = deprecate_function(
    closest_points_index, "closestPoints", removal_version="1.0.0", future_warn=True
)
ExtractCoreMesh = deprecate_function(
    extract_core_mesh, "ExtractCoreMesh", removal_version="1.0.0", future_warn=True
)
closest_points = deprecate_function(
    closest_points_index, "closest_points", removal_version="1.0.0", future_warn=True
)
