"""Utilities for creating averaging operators."""
import numpy as np
import scipy.sparse as sp
from discretize.utils.matrix_utils import mkvc, sub2ind
from discretize.utils.code_utils import deprecate_function

try:
    from discretize._extensions import interputils_cython as pyx

    _interp_point_1D = pyx._interp_point_1D
    _interpmat1D = pyx._interpmat1D
    _interpmat2D = pyx._interpmat2D
    _interpmat3D = pyx._interpmat3D
    _vol_interp = pyx._tensor_volume_averaging
    _interpCython = True
except ImportError as err:
    print(err)
    import os

    # Check if being called from non-standard location (i.e. a git repository)
    # is tree_ext.cpp here? will not be in the folder if installed to site-packages...
    file_test = (
        os.path.dirname(os.path.abspath(__file__))
        + "/_extensions/interputils_cython.pyx"
    )
    if os.path.isfile(file_test):
        # Then we are being run from a repository
        print(
            """
            Unable to import interputils_cython.

            It would appear that discretize is being imported from its repository.
            If this is intentional, you need to run:

            python setup.py build_ext --inplace

            to build the cython code.
            """
        )
    _interpCython = False


def interpolation_matrix(locs, x, y=None, z=None):
    """
    Generate interpolation matrix which maps a tensor quantity to a set of locations.

    This function generates a sparse matrix for interpolating tensor quantities to a set
    of specified locations. It uses nD linear interpolation. The user may generate the
    interpolation matrix for tensor quantities that live on 1D, 2D or 3D tensors. This
    functionality is frequently used to interpolate quantites from cell centers or nodes
    to specified locations.

    In higher dimensions the ordering of the output has the 1st dimension changing the
    quickest.

    Parameters
    ----------
    locs : (n, dim) numpy.ndarray
        The locations for the interpolated values. Here *n* is
        the number of locations and *dim* is the dimension (1, 2 or 3)
    x : (nx) numpy.ndarray
        Vector defining the locations of the tensor along the x-axis
    y : (ny) numpy.ndarray, optional
        Vector defining the locations of the tensor along the y-axis. Required if
        ``dim`` is 2.
    z : (nz) numpy.ndarray, optional
        Vector defining the locations of the tensor along the z-axis. Required if
        ``dim`` is 3.

    Returns
    -------
    (n, nx * ny * nz) scipy.sparse.csr_matrix
        A sparse matrix which interpolates the tensor quantity on cell centers or nodes
        to the set of specified locations.

    Examples
    --------
    Here is a 1D example where a function evaluated on a regularly spaced grid
    is interpolated to a set of random locations. To compare the accuracy, the
    function is evaluated at the set of random locations.

    >>> from discretize.utils import interpolation_matrix
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> np.random.seed(14)

    Create an interpolation matrix

    >>> locs = np.random.rand(50)*0.8+0.1
    >>> x = np.linspace(0, 1, 7)
    >>> dense = np.linspace(0, 1, 200)
    >>> fun = lambda x: np.cos(2*np.pi*x)
    >>> Q = interpolation_matrix(locs, x)

    Plot original function and interpolation

    .. collapse:: Expand to show scripting for plot

        >>> fig1 = plt.figure(figsize=(5, 3))
        >>> ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
        >>> ax.plot(dense, fun(dense), 'k:', lw=3)
        >>> ax.plot(x, fun(x), 'ks', markersize=8)
        >>> ax.plot(locs, Q*fun(x), 'go', markersize=4)
        >>> ax.plot(locs, fun(locs), 'rs', markersize=4)
        >>> ax.legend(
        ...     [
        ...         'True Function',
        ...         'True (discrete loc.)',
        ...         'Interpolated (computed)',
        ...         'True (interp. loc.)'
        ...     ],
        ...     loc='upper center'
        ... )
        >>> plt.show()

    Here, demonstrate a similar example on a 2D mesh using a 2D Gaussian distribution.
    We interpolate the Gaussian from the nodes to cell centers and examine the relative
    error.

    >>> hx = np.ones(10)
    >>> hy = np.ones(10)
    >>> mesh = TensorMesh([hx, hy], x0='CC')
    >>> def fun(x, y):
    ...     return np.exp(-(x**2 + y**2)/2**2)

    Define the the value at the mesh nodes,

    >>> nodes = mesh.nodes
    >>> val_nodes = fun(nodes[:, 0], nodes[:, 1])

    >>> centers = mesh.cell_centers
    >>> A = interpolation_matrix(
    ...     centers, mesh.nodes_x, mesh.nodes_y
    ... )
    >>> val_interp = A.dot(val_nodes)

    Plot the interpolated values, along with the true values at cell centers,

    .. collapse:: Expand to show scripting for plot

        >>> val_centers = fun(centers[:, 0], centers[:, 1])
        >>> fig = plt.figure(figsize=(11,3.3))
        >>> clim = (0., 1.)
        >>> ax1 = fig.add_subplot(131)
        >>> ax2 = fig.add_subplot(132)
        >>> ax3 = fig.add_subplot(133)
        >>> mesh.plot_image(val_centers, ax=ax1, clim=clim)
        >>> mesh.plot_image(val_interp, ax=ax2, clim=clim)
        >>> mesh.plot_image(val_centers-val_interp, ax=ax3, clim=clim)
        >>> ax1.set_title('Analytic at Centers')
        >>> ax2.set_title('Interpolated from Nodes')
        >>> ax3.set_title('Relative Error')
        >>> plt.show()
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

    I = np.repeat(range(npts), 2 ** len(shape))
    J = sub2ind(shape, inds)
    Q = sp.csr_matrix((vals, (I, J)), shape=(npts, np.prod(shape)))
    return Q


def volume_average(mesh_in, mesh_out, values=None, output=None):
    """Volume averaging interpolation between meshes.

    This volume averaging function looks for overlapping cells in each mesh,
    and weights the output values by the partial volume ratio of the overlapping
    input cells. The volume average operation should result in an output such that
    ``np.sum(mesh_in.cell_volumes*values)`` = ``np.sum(mesh_out.cell_volumes*output)``,
    when the input and output meshes have the exact same extent. When the output mesh
    extent goes beyond the input mesh, it is assumed to have constant values in that
    direction. When the output mesh extent is smaller than the input mesh, only the
    overlapping extent of the input mesh contributes to the output.

    This function operates in three different modes. If only *mesh_in* and
    *mesh_out* are given, the returned value is a ``scipy.sparse.csr_matrix``
    that represents this operation (so it could potentially be applied repeatedly).
    If *values* is given, the volume averaging is performed right away (without
    internally forming the matrix) and the returned value is the result of this.
    If *output* is given as well, it will be filled with the values of the
    operation and then returned (assuming it has the correct ``dtype``).

    Parameters
    ----------
    mesh_in : ~discretize.TensorMesh or ~discretize.TreeMesh
        Input mesh (the mesh you are interpolating from)
    mesh_out : ~discretize.TensorMesh or ~discretize.TreeMesh
        Output mesh (the mesh you are interpolating to)
    values : (mesh_in.n_cells) numpy.ndarray, optional
        Array with values defined at the cells of ``mesh_in``
    output : (mesh_out.n_cells) numpy.ndarray of float, optional
        Output array to be overwritten

    Returns
    -------
    (mesh_out.n_cells, mesh_in.n_cells) scipy.sparse.csr_matrix or (mesh_out.n_cells) numpy.ndarray
        If *values* = *None* , the returned value is a matrix representing this
        operation, otherwise it is a :class:`numpy.ndarray` of the result of the
        operation.

    Examples
    --------
    Create two meshes with the same extent, but different divisions (the meshes
    do not have to be the same extent).

    >>> import numpy as np
    >>> from discretize import TensorMesh
    >>> h1 = np.ones(32)
    >>> h2 = np.ones(16)*2
    >>> mesh_in = TensorMesh([h1, h1])
    >>> mesh_out = TensorMesh([h2, h2])

    Create a random model defined on the input mesh, and use volume averaging to
    interpolate it to the output mesh.

    >>> from discretize.utils import volume_average
    >>> model1 = np.random.rand(mesh_in.nC)
    >>> model2 = volume_average(mesh_in, mesh_out, model1)

    Because these two meshes' cells are perfectly aligned, but the output mesh
    has 1 cell for each 4 of the input cells, this operation should effectively
    look like averaging each of those cells values

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(6, 3))
    >>> ax1 = plt.subplot(121)
    >>> mesh_in.plot_image(model1, ax=ax1)
    >>> ax2 = plt.subplot(122)
    >>> mesh_out.plot_image(model2, ax=ax2)
    >>> plt.show()

    """
    try:
        in_type = mesh_in._meshType
        out_type = mesh_out._meshType
    except AttributeError:
        raise TypeError("Both input and output mesh must be valid discetize meshes")

    valid_meshs = ["TENSOR", "TREE"]
    if in_type not in valid_meshs or out_type not in valid_meshs:
        raise NotImplementedError(
            f"Volume averaging is only implemented for TensorMesh and TreeMesh, "
            f"not {type(mesh_in).__name__} and/or {type(mesh_out).__name__}"
        )

    if mesh_in.dim != mesh_out.dim:
        raise ValueError("Both meshes must have the same dimension")

    if values is not None and len(values) != mesh_in.nC:
        raise ValueError(
            "Input array does not have the same length as the number of cells in input mesh"
        )
    if output is not None and len(output) != mesh_out.nC:
        raise ValueError(
            "Output array does not have the same length as the number of cells in output mesh"
        )

    if values is not None:
        values = np.asarray(values, dtype=np.float64)
    if output is not None:
        output = np.asarray(output, dtype=np.float64)

    if in_type == "TENSOR":
        if out_type == "TENSOR":
            return _vol_interp(mesh_in, mesh_out, values, output)
        elif out_type == "TREE":
            return mesh_out._vol_avg_from_tens(mesh_in, values, output)
    elif in_type == "TREE":
        if out_type == "TENSOR":
            return mesh_in._vol_avg_to_tens(mesh_out, values, output)
        elif out_type == "TREE":
            return mesh_out._vol_avg_from_tree(mesh_in, values, output)
    else:
        raise TypeError("Unsupported mesh types")


interpmat = deprecate_function(
    interpolation_matrix, "interpmat", removal_version="1.0.0", future_warn=True
)
