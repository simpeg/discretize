"""Useful functions for working with vectors and matrices."""
import numpy as np
import scipy.sparse as sp
from discretize.utils.code_utils import is_scalar, deprecate_function
import warnings


def mkvc(x, n_dims=1, **kwargs):
    """Coerce a vector to the specified dimensionality.

    This function converts a :class:`numpy.ndarray` to a vector. In general,
    the output vector has a dimension of 1. However, the dimensionality
    can be specified if the user intends to carry out a dot product with
    a higher order array.

    Parameters
    ----------
    x : array_like
        An array that will be reorganized and output as a vector. The input array
        will be flattened on input in Fortran order.
    n_dims : int
        The dimension of the output vector. :data:`numpy.newaxis` are appended to the
        output array until it has this many axes.

    Returns
    -------
    numpy.ndarray
        The output vector, with at least ``n_dims`` axes.

    Examples
    --------
    Here, we reorganize a simple 2D array as a vector and demonstrate the
    impact of the *n_dim* argument.

    >>> from discretize.utils import mkvc
    >>> import numpy as np

    >>> a = np.random.rand(3, 2)
    >>> a
    array([[0.33534155, 0.25334363],
           [0.07147884, 0.81080958],
           [0.85892774, 0.74357806]])

    >>> v = mkvc(a)
    >>> v
    array([0.33534155, 0.07147884, 0.85892774, 0.25334363, 0.81080958,
           0.74357806])

    In Higher dimensions:

    >>> for ii in range(1, 4):
    ...     v = mkvc(a, ii)
    ...     print('Shape of output with n_dim =', ii, ': ', v.shape)
    Shape of output with n_dim = 1 :  (6,)
    Shape of output with n_dim = 2 :  (6, 1)
    Shape of output with n_dim = 3 :  (6, 1, 1)
    """
    if "numDims" in kwargs:
        warnings.warn(
            "The numDims keyword argument has been deprecated, please use n_dims. "
            "This will be removed in discretize 1.0.0",
            FutureWarning,
        )
        n_dims = kwargs["numDims"]
    if type(x) == np.matrix:
        x = np.array(x)

    if hasattr(x, "tovec"):
        x = x.tovec()

    if isinstance(x, Zero):
        return x

    if not isinstance(x, np.ndarray):
        raise TypeError("Vector must be a numpy array")

    if n_dims == 1:
        return x.flatten(order="F")
    elif n_dims == 2:
        return x.flatten(order="F")[:, np.newaxis]
    elif n_dims == 3:
        return x.flatten(order="F")[:, np.newaxis, np.newaxis]


def sdiag(v):
    """Generate sparse diagonal matrix from a vector.

    This function creates a sparse diagonal matrix whose diagonal elements
    are defined by the input vector *v*. For a vector of length *n*,
    the output matrix has shape (n,n).

    Parameters
    ----------
    v : (n) numpy.ndarray or discretize.utils.Zero
        The vector defining the diagonal elements of the sparse matrix being constructed

    Returns
    -------
    (n, n) scipy.sparse.csr_matrix or discretize.utils.Zero
        The sparse diagonal matrix.

    Examples
    --------
    Use a 1D array of values to construct a sparse diagonal matrix.

    >>> from discretize.utils import sdiag
    >>> import numpy as np
    >>> v = np.array([6., 3., 1., 8., 0., 5.])
    >>> M = sdiag(v)
    """
    if isinstance(v, Zero):
        return Zero()

    return sp.spdiags(mkvc(v), 0, v.size, v.size, format="csr")


def sdinv(M):
    """Return inverse of a sparse diagonal matrix.

    This function extracts the diagonal elements of the input matrix *M*
    and creates a sparse diagonal matrix from the reciprocal these elements.
    If the input matrix *M* is diagonal, the output is the inverse of *M*.

    Parameters
    ----------
    M : (n, n) scipy.sparse.csr_matrix
        A sparse diagonal matrix

    Returns
    -------
    (n, n) scipy.sparse.csr_matrix
        The inverse of the sparse diagonal matrix.

    Examples
    --------
    >>> from discretize.utils import sdiag, sdinv
    >>> import numpy as np

    >>> v = np.array([6., 3., 1., 8., 0., 5.])
    >>> M = sdiag(v)
    >>> Minv = sdinv(M)

    """
    return sdiag(1.0 / M.diagonal())


def speye(n):
    """Generate sparse identity matrix.

    Parameters
    ----------
    n : int
        The dimensions of the sparse identity matrix.

    Returns
    -------
    (n, n) scipy.sparse.csr_matrix
        The sparse identity matrix.
    """
    return sp.identity(n, format="csr")


def kron3(A, B, C):
    r"""Compute kronecker products between 3 sparse matricies.

    Where :math:`\otimes` denotes the Kronecker product and *A, B* and *C* are
    sparse matrices, this function outputs :math:`(A \otimes B) \otimes C`.

    Parameters
    ----------
    A, B, C : scipy.sparse.spmatrix
        Sparse matrices.

    Returns
    -------
    scipy.sparse.csr_matrix
        Kroneker between the 3 sparse matrices.
    """
    return sp.kron(sp.kron(A, B), C, format="csr")


def spzeros(n1, n2):
    """Generate sparse matrix of zeros of shape=(n1, n2).

    Parameters
    ----------
    n1 : int
        Number of rows.
    n2 : int
        Number of columns.

    Returns
    -------
    (n1, n2) scipy.sparse.dia_matrix
        A sparse matrix of zeros.
    """
    return sp.dia_matrix((n1, n2))


def ddx(n):
    r"""Create 1D difference (derivative) operator from nodes to centers.

    For n cells, the 1D difference (derivative) operator from nodes to
    centers is sparse, has shape (n, n+1) and takes the form:

    .. math::
        \begin{bmatrix}
        -1 & 1 & & & \\
        & -1 & 1 & & \\
        & & \ddots & \ddots & \\
        & & & -1 & 1
        \end{bmatrix}

    Parameters
    ----------
    n : int
        Number of cells

    Returns
    -------
    (n, n + 1) scipy.sparse.csr_matrix
        The 1D difference operator from nodes to centers.
    """
    return sp.spdiags((np.ones((n + 1, 1)) * [-1, 1]).T, [0, 1], n, n + 1, format="csr")


def av(n):
    r"""Create 1D averaging operator from nodes to cell-centers.

    For n cells, the 1D averaging operator from nodes to centerss
    is sparse, has shape (n, n+1) and takes the form:

    .. math::
        \begin{bmatrix}
        1/2 & 1/2 & & & \\
        & 1/2 & 1/2 & & \\
        & & \ddots & \ddots & \\
        & & & 1/2 & 1/2
        \end{bmatrix}

    Parameters
    ----------
    n : int
        Number of cells

    Returns
    -------
    (n, n + 1) scipy.sparse.csr_matrix
        The 1D averaging operator from nodes to centers.
    """
    return sp.spdiags(
        (0.5 * np.ones((n + 1, 1)) * [1, 1]).T, [0, 1], n, n + 1, format="csr"
    )


def av_extrap(n):
    r"""Create 1D averaging operator from cell-centers to nodes.

    For n cells, the 1D averaging operator from cell centers to nodes
    is sparse and has shape (n+1, n). Values at the outmost nodes are
    extrapolated from the nearest cell center value. Thus the operator
    takes the form:

    .. math::
        \begin{bmatrix}
        1 & & & & \\
        1/2 & 1/2 & & & \\
        & 1/2 & 1/2 & & & \\
        & & \ddots & \ddots & \\
        & & & 1/2 & 1/2 \\
        & & & & 1
        \end{bmatrix}

    Parameters
    ----------
    n : int
        Number of cells

    Returns
    -------
    (n+1, n) scipy.sparse.csr_matrix
        The 1D averaging operator from cell-centers to nodes.
    """
    Av = sp.spdiags(
        (0.5 * np.ones((n, 1)) * [1, 1]).T, [-1, 0], n + 1, n, format="csr"
    ) + sp.csr_matrix(([0.5, 0.5], ([0, n], [0, n - 1])), shape=(n + 1, n))
    return Av


def ndgrid(*args, vector=True, order="F"):
    """Generate gridded locations for 1D, 2D, or 3D tensors.

    For 1D, 2D, or 3D tensors, this function takes the unique positions defining
    a tensor along each of its axis and returns the gridded locations.
    For 2D and 3D meshes, the user may treat the unique *x*, *y* (and *z*)
    positions a successive positional arguments or as a single argument using
    a list [*x*, *y*, (*z*)].

    For outputs, let *dim* be the number of dimension (1, 2 or 3) and let *n* be
    the total number of gridded locations. The gridded *x*, *y* (and *z*) locations
    can be return as a single numpy array of shape [n, ndim]. The user can also
    return the gridded *x*, *y* (and *z*) locations as a list of length *ndim*.
    The list contains entries contain the *x*, *y* (and *z*) locations as tensors.
    See examples.

    Parameters
    ----------
    *args : (n, dim) numpy.ndarray or (dim) list of (n) numpy.ndarray
        Positions along each axis of the tensor. The user can define these as
        successive positional arguments *x*, *y*, (and *z*) or as a single argument
        using a list [*x*, *y*, (*z*)].
    vector : bool, optional
        If *True*, the output is a numpy array of dimension [n, ndim]. If *False*,
        the gridded x, y (and z) locations are returned as separate ndarrays in a list.
        Default is *True*.
    order : {'F', 'C', 'A'}
        Define ordering using one of the following options:
        'C' is C-like ordering, 'F' is Fortran-like ordering, 'A' is Fortran
        ordering if memory is contigious and C-like otherwise. Default = 'F'.
        See :func:`numpy.reshape` for more on this argument.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        If *vector* = *True* the gridded *x*, *y*, (and *z*) locations are
        returned as a numpy array of shape [n, ndim]. If *vector* = *False*,
        the gridded *x*, *y*, (and *z*) are returned as a list of vectors.

    Examples
    --------
    >>> from discretize.utils import ndgrid
    >>> import numpy as np

    >>> x = np.array([1, 2, 3])
    >>> y = np.array([2, 4])

    >>> ndgrid([x, y])
    array([[1, 2],
           [2, 2],
           [3, 2],
           [1, 4],
           [2, 4],
           [3, 4]])

    >>> ndgrid(x, y, order='C')
    array([[1, 2],
           [1, 4],
           [2, 2],
           [2, 4],
           [3, 2],
           [3, 4]])

    >>> ndgrid(x, y, vector=False)
    [array([[1, 1],
           [2, 2],
           [3, 3]]), array([[2, 4],
           [2, 4],
           [2, 4]])]
    """
    # Read the keyword arguments, and only accept a vector=True/False
    if not isinstance(vector, bool):
        raise TypeError("'vector' keyword must be a bool")

    # you can either pass a list [x1, x2, x3] or each seperately
    if type(args[0]) == list:
        xin = args[0]
    else:
        xin = args

    # Each vector needs to be a numpy array
    try:
        if len(xin) == 1:
            return np.array(xin[0])
        meshed = np.meshgrid(*xin, indexing="ij")
    except Exception:
        raise TypeError("All arguments must be array like")

    if vector:
        return np.column_stack([x.reshape(-1, order=order) for x in meshed])
    return meshed


def make_boundary_bool(shape, bdir="xyz", **kwargs):
    r"""Return boundary indices of a tensor grid.

    For a tensor grid whose shape is given (1D, 2D or 3D), this function
    returns a boolean index array identifying the x, y and/or z
    boundary locations.

    Parameters
    ----------
    shape : (dim) tuple of int
        Defines the shape of the tensor (1D, 2D or 3D).
    bdir : str containing characters 'x', 'y' and/or 'z'
        Specify the boundaries whose indices you want returned; e.g. for a 3D
        tensor, you may set *dir* = 'xz' to return the indices of the x and
        z boundary locations.

    Returns
    -------
    numpy.ndarray of bool
        Indices of boundary locations of the tensor for specified boundaries. The
        returned order matches the order the items occur in the flattened ``ndgrid``

    Examples
    --------
    Here we construct a 3x3 tensor and find the indices of the boundary locations.

    >>> from discretize.utils.matrix_utils import ndgrid, make_boundary_bool
    >>> import numpy as np

    Define a 3x3 tensor grid

    >>> x = np.array([1, 2, 3])
    >>> y = np.array([2, 4, 6])
    >>> tensor_grid = ndgrid(x, y)

    Find indices of boundary locations.

    >>> shape = (len(x), len(y))
    >>> bool_ind = make_boundary_bool(shape)
    >>> tensor_grid[bool_ind]
    array([[1, 2],
           [2, 2],
           [3, 2],
           [1, 4],
           [3, 4],
           [1, 6],
           [2, 6],
           [3, 6]])

    Find indices of locations of only the x boundaries,

    >>> bool_ind_x = make_boundary_bool(shape, 'x')
    >>> tensor_grid[bool_ind_x]
    array([[1, 2],
           [3, 2],
           [1, 4],
           [3, 4],
           [1, 6],
           [3, 6]])
    """
    old_dir = kwargs.pop("dir", None)
    if old_dir is not None:
        warnings.warn(
            "The `dir` keyword argument has been renamed to `bdir` to avoid shadowing the "
            "builtin variable `dir`. This will be removed in discretize 1.0.0",
            FutureWarning,
        )
        bdir = old_dir
    is_b = np.zeros(shape, dtype=bool, order="F")
    if "x" in bdir:
        is_b[[0, -1]] = True
    if len(shape) > 1:
        if "y" in bdir:
            is_b[:, [0, -1]] = True
    if len(shape) > 2:
        if "z" in bdir:
            is_b[:, :, [0, -1]] = True
    return is_b.reshape(-1, order="F")


def ind2sub(shape, inds):
    r"""Return subscripts of tensor grid elements from indices.

    This function is a wrapper for :func:`numpy.unravel_index` with a hard-coded Fortran
    order.

    Consider the :math:`n^{th}` element of a tensor grid with *N* elements.
    The position of this element in the tensor grid can also be defined by
    subscripts (i,j,k). For an array containing the indices for a set of
    tensor elements, this function returns the corresponding subscripts.

    Parameters
    ----------
    shape : (dim) tuple of int
        Defines the shape of the tensor (1D, 2D or 3D).
    inds : array_like of int
        The indices of the tensor elements whose subscripts you want returned.

    Returns
    -------
    (dim) tuple of numpy.ndarray
        Corresponding subscipts for the indices provided. The output is a
        tuple containing 1D integer arrays for the i, j and k subscripts, respectively.
        The output array will match the shape of the **inds** input.

    See Also
    --------
    numpy.unravel_index
    """
    return np.unravel_index(inds, shape, order="F")


def sub2ind(shape, subs):
    r"""Return indices of tensor grid elements from subscripts.

    This function is a wrapper for :func:`numpy.ravel_multi_index` with a hard-coded
    Fortran order, and a column order for the ``multi_index``

    Consider elements of a tensors grid whose positions are given by the
    subscripts (i,j,k). This function will return the corresponding indices
    of these elements. Each row of the input array *subs* defines the
    ijk for a particular tensor element.

    Parameters
    ----------
    shape : (dim) tuple of int
        Defines the shape of the tensor (1D, 2D or 3D).
    subs : (N, dim) array_like of int
        The subscripts of the tensor grid elements. Each rows defines the position
        of a particular tensor element. The shape of of the array is (N, ndim).

    Returns
    -------
    numpy.ndarray of int
        The indices of the tensor grid elements defined by *subs*.

    See Also
    --------
    numpy.ravel_multi_index

    Examples
    --------
    He we recreate the examples from :func:`numpy.ravel_multi_index` to illustrate the
    differences. The indices corresponding to each dimension are now columns in the
    array (instead of rows), and it assumed to use a Fortran order.

    >>> import numpy as np
    >>> from discretize.utils import sub2ind
    >>> arr = np.array([[3, 4], [6, 5], [6, 1]])
    >>> sub2ind((7, 6), arr)
    array([31, 41, 13], dtype=int64)
    """
    if len(shape) == 1:
        return subs
    subs = np.atleast_2d(subs)
    if subs.shape[1] != len(shape):
        raise ValueError(
            "Indexing must be done as a column vectors. e.g. [[3,6],[6,2],...]"
        )
    inds = np.ravel_multi_index(subs.T, shape, order="F")
    return mkvc(inds)


def get_subarray(A, ind):
    """Extract a subarray.

    For a :class:`numpy.ndarray`, the function **get_subarray** extracts a subset of
    the array. The portion of the original array being extracted is defined
    by providing the indices along each axis.

    Parameters
    ----------
    A : numpy.ndarray
        The original numpy array. Must be 1, 2 or 3 dimensions.
    ind : (dim) list of numpy.ndarray
        A list of numpy arrays containing the indices being extracted along each
        dimension. The length of the list must equal the dimensions of the input array.

    Returns
    -------
    numpy.ndarray
        The subarray extracted from the original array

    Examples
    --------
    Here we construct a random 3x3 numpy array and use **get_subarray** to extract
    the first column.

    >>> from discretize.utils import get_subarray
    >>> import numpy as np
    >>> A = np.random.rand(3, 3)
    >>> A
    array([[1.07969034e-04, 9.78613931e-01, 6.62123429e-01],
           [8.80722877e-01, 7.61035691e-01, 7.42546796e-01],
           [9.09488911e-01, 7.80626334e-01, 8.67663825e-01]])

    Define the indexing along the columns and rows and create the indexing list

    >>> ind_x = np.array([0, 1, 2])
    >>> ind_y = np.array([0, 2])
    >>> ind = [ind_x, ind_y]

    Extract the first, and third column of A

    >>> get_subarray(A, ind)
    array([[1.07969034e-04, 6.62123429e-01],
           [8.80722877e-01, 7.42546796e-01],
           [9.09488911e-01, 8.67663825e-01]])
    """
    if not isinstance(ind, list):
        raise TypeError("ind must be a list of vectors")
    if len(A.shape) != len(ind):
        raise ValueError("ind must have the same length as the dimension of A")

    if len(A.shape) == 2:
        return A[ind[0], :][:, ind[1]]
    elif len(A.shape) == 3:
        return A[ind[0], :, :][:, ind[1], :][:, :, ind[2]]
    else:
        raise Exception("get_subarray does not support dimension asked.")


def inverse_3x3_block_diagonal(
    a11, a12, a13, a21, a22, a23, a31, a32, a33, return_matrix=True, **kwargs
):
    r"""Invert a set of 3x3 matricies from vectors containing their elements.

    Parameters
    ----------
    a11, a12, ..., a33 : (n_blocks) numpy.ndarray
        Vectors which contain the
        corresponding element for all 3x3 matricies
    return_matrix : bool, optional
        - **True**: Returns the sparse block 3x3 matrix *M* (default).
        - **False:** Returns the vectors containing the elements of each matrix' inverse.

    Returns
    -------
    (3 * n_blocks, 3 * n_blocks) scipy.sparse.coo_matrix or list of (n_blocks)
        numpy.ndarray. If *return_matrix = False*, the function will return vectors
        *b11, b12, b13, b21, b22, b23, b31, b32, b33*. If *return_matrix = True*, the
        function will return the block matrix *M*.

    Notes
    -----
    The elements of a 3x3 matrix *A* are given by:

    .. math::
        A = \begin{bmatrix}
        a_{11} & a_{12} & a_{13} \\
        a_{21} & a_{22} & a_{23} \\
        a_{31} & a_{32} & a_{33}
        \end{bmatrix}

    For a set of 3x3 matricies, the elements may be stored in a set of 9 distinct vectors
    :math:`\mathbf{a_{11}}`, :math:`\mathbf{a_{12}}`, ..., :math:`\mathbf{a_{33}}`.
    For each matrix, **inverse_3x3_block_diagonal** ouputs the vectors containing the
    elements of each matrix' inverse; i.e.
    :math:`\mathbf{b_{11}}`, :math:`\mathbf{b_{12}}`, ..., :math:`\mathbf{b_{33}}`
    where:

    .. math::
        A^{-1} = B = \begin{bmatrix}
        b_{11} & b_{12} & b_{13} \\
        b_{21} & b_{22} & b_{23} \\
        b_{31} & b_{32} & b_{33}
        \end{bmatrix}

    For special applications, we may want to output the elements of the inverses
    of the matricies as a 3x3 block matrix of the form:

    .. math::
        M = \begin{bmatrix}
        D_{11} & D_{12} & D_{13} \\
        D_{21} & D_{22} & D_{23} \\
        D_{31} & D_{32} & D_{33}
        \end{bmatrix}

    where :math:`D_{ij}` are diagonal matrices whose non-zero elements
    are defined by vector :math:`\\mathbf{b_{ij}}`. Where *n* is the
    number of matricies, the block matrix is sparse with dimensions
    (3n, 3n).

    Examples
    --------
    Here, we define four 3x3 matricies and reorganize their elements into
    9 vectors a11, a12, ..., a33. We then examine the outputs of the
    function **inverse_3x3_block_diagonal** when the argument
    *return_matrix* is set to both *True* and *False*.

    >>> from discretize.utils import inverse_3x3_block_diagonal
    >>> import numpy as np
    >>> import scipy as sp
    >>> import matplotlib.pyplot as plt

    Define four 3x3 matricies, and organize their elements into nine vectors

    >>> A1 = np.random.uniform(1, 10, (3, 3))
    >>> A2 = np.random.uniform(1, 10, (3, 3))
    >>> A3 = np.random.uniform(1, 10, (3, 3))
    >>> A4 = np.random.uniform(1, 10, (3, 3))
    >>> [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]] = np.stack(
    ...     [A1, A2, A3, A4], axis=-1
    ... )

    Return the elements of their inverse and validate

    >>> b11, b12, b13, b21, b22, b23, b31, b32, b33 = inverse_3x3_block_diagonal(
    ...     a11, a12, a13, a21, a22, a23, a31, a32, a33, return_matrix=False
    ... )
    >>> Bs = np.stack([[b11, b12, b13],[b21, b22, b23],[b31, b32, b33]])
    >>> B1, B2, B3, B4 = Bs.transpose((2, 0, 1))

    >>> np.linalg.inv(A1)
    array([[ 0.20941584,  0.18477151, -0.22637147],
           [-0.06420656, -0.34949639,  0.29216461],
           [-0.14226339,  0.11160555,  0.0907583 ]])
    >>> B1
    array([[ 0.20941584,  0.18477151, -0.22637147],
           [-0.06420656, -0.34949639,  0.29216461],
           [-0.14226339,  0.11160555,  0.0907583 ]])

    We can also return this as a sparse matrix with block diagonal inverse

    >>> M = inverse_3x3_block_diagonal(
    ...     a11, a12, a13, a21, a22, a23, a31, a32, a33
    ... )
    >>> plt.spy(M)
    >>> plt.show()
    """
    if "returnMatrix" in kwargs:
        warnings.warn(
            "The returnMatrix keyword argument has been deprecated, please use return_matrix. "
            "This will be removed in discretize 1.0.0",
            FutureWarning,
        )
        return_matrix = kwargs["returnMatrix"]

    a11 = mkvc(a11)
    a12 = mkvc(a12)
    a13 = mkvc(a13)
    a21 = mkvc(a21)
    a22 = mkvc(a22)
    a23 = mkvc(a23)
    a31 = mkvc(a31)
    a32 = mkvc(a32)
    a33 = mkvc(a33)

    detA = (
        a31 * a12 * a23
        - a31 * a13 * a22
        - a21 * a12 * a33
        + a21 * a13 * a32
        + a11 * a22 * a33
        - a11 * a23 * a32
    )

    b11 = +(a22 * a33 - a23 * a32) / detA
    b12 = -(a12 * a33 - a13 * a32) / detA
    b13 = +(a12 * a23 - a13 * a22) / detA

    b21 = +(a31 * a23 - a21 * a33) / detA
    b22 = -(a31 * a13 - a11 * a33) / detA
    b23 = +(a21 * a13 - a11 * a23) / detA

    b31 = -(a31 * a22 - a21 * a32) / detA
    b32 = +(a31 * a12 - a11 * a32) / detA
    b33 = -(a21 * a12 - a11 * a22) / detA

    if not return_matrix:
        return b11, b12, b13, b21, b22, b23, b31, b32, b33

    return sp.vstack(
        (
            sp.hstack((sdiag(b11), sdiag(b12), sdiag(b13))),
            sp.hstack((sdiag(b21), sdiag(b22), sdiag(b23))),
            sp.hstack((sdiag(b31), sdiag(b32), sdiag(b33))),
        )
    )


def inverse_2x2_block_diagonal(a11, a12, a21, a22, return_matrix=True, **kwargs):
    r"""
    Invert a set of 2x2 matricies from vectors containing their elements.

    Parameters
    ----------
    a11, a12, a21, a22 : (n_blocks) numpy.ndarray
        All arguments a11, a12, a21, a22 are vectors which contain the
        corresponding element for all 2x2 matricies
    return_matrix : bool, optional
        - **True:** Returns the sparse block 2x2 matrix *M*.
        - **False:** Returns the vectors containing the elements of each matrix' inverse.

    Returns
    -------
    (2 * n_blocks, 2 * n_blocks) scipy.sparse.coo_matrix or list of (n_blocks) numpy.ndarray
        If *return_matrix = False*, the function will return vectors
        *b11, b12, b21, b22*.
        If *return_matrix = True*, the function will return the
        block matrix *M*

    Notes
    -----
    The elements of a 2x2 matrix *A* are given by:

    .. math::
        A = \begin{bmatrix}
        a_{11} & a_{12} \\
        a_{21} & a_{22}
        \end{bmatrix}

    For a set of 2x2 matricies, the elements may be stored in a set of 4 distinct vectors
    :math:`\mathbf{a_{11}}`, :math:`\mathbf{a_{12}}`, :math:`\mathbf{a_{21}}` and
    :math:`\mathbf{a_{22}}`.
    For each matrix, **inverse_2x2_block_diagonal** ouputs the vectors containing the
    elements of each matrix' inverse; i.e.
    :math:`\mathbf{b_{11}}`, :math:`\mathbf{b_{12}}`, :math:`\mathbf{b_{21}}` and
    :math:`\mathbf{b_{22}}` where:

    .. math::
        A^{-1} = B = \begin{bmatrix}
        b_{11} & b_{12} \\
        b_{21} & b_{22}
        \end{bmatrix}

    For special applications, we may want to output the elements of the inverses
    of the matricies as a 2x2 block matrix of the form:

    .. math::
        M = \begin{bmatrix}
        D_{11} & D_{12} \\
        D_{21} & D_{22}
        \end{bmatrix}

    where :math:`D_{ij}` are diagonal matrices whose non-zero elements
    are defined by vector :math:`\mathbf{b_{ij}}`. Where *n* is the
    number of matricies, the block matrix is sparse with dimensions
    (2n, 2n).

    Examples
    --------
    Here, we define four 2x2 matricies and reorganize their elements into
    4 vectors a11, a12, a21 and a22. We then examine the outputs of the
    function **inverse_2x2_block_diagonal** when the argument
    *return_matrix* is set to both *True* and *False*.

    >>> from discretize.utils import inverse_2x2_block_diagonal
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    Define four 3x3 matricies, and organize their elements into four vectors

    >>> A1 = np.random.uniform(1, 10, (2, 2))
    >>> A2 = np.random.uniform(1, 10, (2, 2))
    >>> A3 = np.random.uniform(1, 10, (2, 2))
    >>> A4 = np.random.uniform(1, 10, (2, 2))
    >>> [[a11, a12], [a21, a22]] = np.stack([A1, A2, A3, A4], axis=-1)

    Return the elements of their inverse and validate

    >>> b11, b12, b21, b22 = inverse_2x2_block_diagonal(
    ...     a11, a12, a21, a22, return_matrix=False
    ... )
    >>> Bs = np.stack([[b11, b12],[b21, b22]])
    >>> B1, B2, B3, B4 = Bs.transpose((2, 0, 1))

    >>> np.linalg.inv(A1)
    array([[ 0.34507439, -0.4831833 ],
           [-0.24286626,  0.57531461]])
    >>> B1
    array([[ 0.34507439, -0.4831833 ],
           [-0.24286626,  0.57531461]])

    Plot the sparse block matrix containing elements of the inverses

    >>> M = inverse_2x2_block_diagonal(
    ...     a11, a12, a21, a22
    ... )
    >>> plt.spy(M)
    >>> plt.show()
    """
    if "returnMatrix" in kwargs:
        warnings.warn(
            "The returnMatrix keyword argument has been deprecated, please use return_matrix. "
            "This will be removed in discretize 1.0.0",
            FutureWarning,
        )
        return_matrix = kwargs["returnMatrix"]

    a11 = mkvc(a11)
    a12 = mkvc(a12)
    a21 = mkvc(a21)
    a22 = mkvc(a22)

    # compute inverse of the determinant.
    detAinv = 1.0 / (a11 * a22 - a21 * a12)

    b11 = +detAinv * a22
    b12 = -detAinv * a12
    b21 = -detAinv * a21
    b22 = +detAinv * a11

    if not return_matrix:
        return b11, b12, b21, b22

    return sp.vstack(
        (sp.hstack((sdiag(b11), sdiag(b12))), sp.hstack((sdiag(b21), sdiag(b22))))
    )


def invert_blocks(A):
    """Invert a set of 2x2 or 3x3 matricies.

    This is a shortcut function that will only invert 2x2 and 3x3 matrices.
    The function is broadcast over the last two dimensions of A.

    Parameters
    ----------
    A : (..., N, N) numpy.ndarray
        the block of matrices to invert, N must be either 2 or 3.

    Returns
    -------
    (..., N, N) numpy.ndarray
        the block of inverted matrices

    See Also
    --------
    numpy.linalg.inv : Similar to this function, but is not specialized to 2x2 or 3x3
    inverse_2x2_block_diagonal : use when each element of the blocks is separated
    inverse_3x3_block_diagonal : use when each element of the blocks is separated

    Examples
    --------
    >>> from discretize.utils import invert_blocks
    >>> import numpy as np
    >>> x = np.ones((1000, 3, 3))
    >>> x[..., 1, 1] = 0
    >>> x[..., 1, 2] = 0
    >>> x[..., 2, 1] = 0
    >>> As = np.einsum('...ij,...jk', x, x.transpose(0, 2, 1))
    >>> Ainvs = invert_blocks(As)
    >>> As[0] @ Ainvs[0]
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    if A.shape[-1] != A.shape[-2]:
        raise ValueError(f"Last two dimensions are not equal, got {A.shape}")

    if A.shape[-1] == 2:
        a11 = A[..., 0, 0]
        a12 = A[..., 0, 1]
        a21 = A[..., 1, 0]
        a22 = A[..., 1, 1]

        detA = a11 * a22 - a21 * a12
        B = np.empty_like(A)
        B[..., 0, 0] = a22 / detA
        B[..., 0, 1] = -a12 / detA
        B[..., 1, 0] = -a21 / detA
        B[..., 1, 1] = a11 / detA

    elif A.shape[-1] == 3:
        a11 = A[..., 0, 0]
        a12 = A[..., 0, 1]
        a13 = A[..., 0, 2]
        a21 = A[..., 1, 0]
        a22 = A[..., 1, 1]
        a23 = A[..., 1, 2]
        a31 = A[..., 2, 0]
        a32 = A[..., 2, 1]
        a33 = A[..., 2, 2]

        B = np.empty_like(A)
        B[..., 0, 0] = a22 * a33 - a23 * a32
        B[..., 0, 1] = a13 * a32 - a12 * a33
        B[..., 0, 2] = a12 * a23 - a13 * a22

        B[..., 1, 0] = a31 * a23 - a21 * a33
        B[..., 1, 1] = a11 * a33 - a31 * a13
        B[..., 1, 2] = a21 * a13 - a11 * a23

        B[..., 2, 0] = a21 * a32 - a31 * a22
        B[..., 2, 1] = a31 * a12 - a11 * a32
        B[..., 2, 2] = a11 * a22 - a21 * a12

        detA = a11 * B[..., 0, 0] + a21 * B[..., 0, 1] + a31 * B[..., 0, 2]
        B /= detA[..., None, None]
    else:
        raise NotImplementedError("Only supports 2x2 and 3x3 blocks")
    return B


class TensorType(object):
    r"""Class for determining property tensor type.

    For a given *mesh*, the **TensorType** class examines the :class:`numpy.ndarray`
    *tensor* to determine whether *tensor* defines a scalar, isotropic,
    diagonal anisotropic or full tensor anisotropic constitutive relationship
    for each cell on the mesh. The general theory behind this functionality
    is explained below.

    Parameters
    ----------
    mesh : discretize.base.BaseTensorMesh
        An instance of any of the mesh classes support in discretize; i.e. *TensorMesh*,
        *CylindricalMesh*, *TreeMesh* or *CurvilinearMesh*.
    tensor : numpy.ndarray or a float
        The shape of the input argument *tensor* must fall into one of these
        classifications:

        - *Scalar:* A float is entered.
        - *Isotropic:* A 1D numpy.ndarray with a property value for every cell.
        - *Anisotropic:* A (*nCell*, *dim*) numpy.ndarray of shape where each row
          defines the diagonal-anisotropic property parameters for each cell.
          *nParam* = 2 for 2D meshes and *nParam* = 3 for 3D meshes.
        - *Tensor:* A (*nCell*, *nParam*) numpy.ndarray where each row
          defines the full anisotropic property parameters for each cell.
          *nParam* = 3 for 2D meshes and *nParam* = 6 for 3D meshes.

    Notes
    -----
    The relationship between a quantity and its response to external
    stimuli (e.g. Ohm's law) can be defined by a scalar quantity:

    .. math::
        \vec{j} = \sigma \vec{e}

    Or in the case of anisotropy, the relationship is defined generally by
    a symmetric tensor:

    .. math::
        \vec{j} = \Sigma \vec{e} \;\;\; where \;\;\;
        \Sigma = \begin{bmatrix}
        \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
        \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\
        \sigma_{xz} & \sigma_{yz} & \sigma_{zz}
        \end{bmatrix}

    In 3D, the tensor is defined by 6 independent element (3 independent elements in
    2D). When using the input argument *tensor* to define the consitutive relationship
    for every cell in the *mesh*, there are 4 classifications recognized by discretize:

    - **Scalar:** :math:`\vec{j} = \sigma \vec{e}`, where :math:`\sigma` a constant.
      Thus the input argument *tensor* is a float.
    - **Isotropic:** :math:`\vec{j} = \sigma \vec{e}`, where :math:`\sigma` varies
      spatially. Thus the input argument *tensor* is a 1D array that provides a
      :math:`\sigma` value for every cell in the mesh.
    - **Anisotropic:** :math:`\vec{j} = \Sigma \vec{e}`, where the off-diagonal elements
      are zero. That is, :math:`\Sigma` is diagonal. In this case, the input argument
      *tensor* defining the physical properties in each cell is a :class:`numpy.ndarray`
      of shape (*nCells*, *dim*).
    - **Tensor:** :math:`\vec{j} = \Sigma \vec{e}`, where off-diagonal elements are
      non-zero and :math:`\Sigma` is a full tensor. In this case, the input argument
      *tensor* defining the physical properties in each cell is a :class:`numpy.ndarray`
      of shape (*nCells*, *nParam*). In 2D, *nParam* = 3 and in 3D, *nParam* = 6.
    """

    def __init__(self, mesh, tensor):
        if tensor is None:  # default is ones
            self._tt = -1
            self._tts = "none"
        elif is_scalar(tensor):
            self._tt = 0
            self._tts = "scalar"
        elif tensor.size == mesh.nC:
            self._tt = 1
            self._tts = "isotropic"
        elif (mesh.dim == 2 and tensor.size == mesh.nC * 2) or (
            mesh.dim == 3 and tensor.size == mesh.nC * 3
        ):
            self._tt = 2
            self._tts = "anisotropic"
        elif (mesh.dim == 2 and tensor.size == mesh.nC * 3) or (
            mesh.dim == 3 and tensor.size == mesh.nC * 6
        ):
            self._tt = 3
            self._tts = "tensor"
        else:
            raise Exception("Unexpected shape of tensor: {}".format(tensor.shape))

    def __str__(self):
        """Represent tensor type as a string."""
        return "TensorType[{0:d}]: {1!s}".format(self._tt, self._tts)

    def __eq__(self, v):
        """Compare tensor type equal to a value."""
        return self._tt == v

    def __le__(self, v):
        """Compare tensor type less than or equal to a value."""
        return self._tt <= v

    def __ge__(self, v):
        """Compare tensor type greater than or equal to a value."""
        return self._tt >= v

    def __lt__(self, v):
        """Compare tensor type less than a value."""
        return self._tt < v

    def __gt__(self, v):
        """Compare tensor type greater than a value."""
        return self._tt > v


def make_property_tensor(mesh, tensor):
    r"""Construct the physical property tensor.

    For a given *mesh*, the input parameter *tensor* is a :class:`numpy.ndarray`
    defining the constitutive relationship (e.g. Ohm's law) between two
    discrete vector quantities :math:`\boldsymbol{j}` and
    :math:`\boldsymbol{e}` living at cell centers. The function
    **make_property_tensor** constructs the property tensor
    :math:`\boldsymbol{M}` for the entire mesh such that:

    >>> j = M @ e

    where the Cartesian components of the discrete vector for
    are organized according to:

    >>> e = np.r_[ex, ey, ez]
    >>> j = np.r_[jx, jy, jz]

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
       A mesh
    tensor : numpy.ndarray or a float
        - *Scalar:* A float is entered.
        - *Isotropic:* A 1D numpy.ndarray with a property value for every cell.
        - *Anisotropic:* A (*nCell*, *dim*) numpy.ndarray where each row
          defines the diagonal-anisotropic property parameters for each cell.
          *nParam* = 2 for 2D meshes and *nParam* = 3 for 3D meshes.
        - *Tensor:* A (*nCell*, *nParam*) numpy.ndarray where each row defines
          the full anisotropic property parameters for each cell. *nParam* = 3 for 2D
          meshes and *nParam* = 6 for 3D meshes.

    Returns
    -------
    (dim * n_cells, dim * n_cells) scipy.sparse.coo_matrix
        The property tensor.

    Notes
    -----
    The relationship between a quantity and its response to external
    stimuli (e.g. Ohm's law) in each cell can be defined by a scalar
    function :math:`\sigma` in the isotropic case, or by a tensor
    :math:`\Sigma` in the anisotropic case, i.e.:

    .. math::
        \vec{j} = \sigma \vec{e} \;\;\;\;\;\; \textrm{or} \;\;\;\;\;\; \vec{j} = \Sigma \vec{e}

    where

    .. math::
        \Sigma = \begin{bmatrix}
        \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
        \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\
        \sigma_{xz} & \sigma_{yz} & \sigma_{zz}
        \end{bmatrix}

    Examples
    --------
    For the 4 classifications allowable (scalar, isotropic, anistropic and tensor),
    we construct and compare the property tensor on a small 2D mesh. For this
    example, note the following:

        - The dimensions for all property tensors are the same
        - All property tensors, except in the case of full anisotropy are diagonal
          sparse matrices
        - For the scalar case, the non-zero elements are equal
        - For the isotropic case, the non-zero elements repreat in order for the x, y
          (and z) components
        - For the anisotropic case (diagonal anisotropy), the non-zero elements do not
          repeat
        - For the tensor caes (full anisotropy), there are off-diagonal components

    >>> from discretize.utils import make_property_tensor
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib as mpl

    Define a 2D tensor mesh

    >>> h = [1., 1., 1.]
    >>> mesh = TensorMesh([h, h], origin='00')

    Define a physical property for all cases (2D)

    >>> sigma_scalar = 4.
    >>> sigma_isotropic = np.random.randint(1, 10, mesh.nC)
    >>> sigma_anisotropic = np.random.randint(1, 10, (mesh.nC, 2))
    >>> sigma_tensor = np.random.randint(1, 10, (mesh.nC, 3))

    Construct the property tensor in each case

    >>> M_scalar = make_property_tensor(mesh, sigma_scalar)
    >>> M_isotropic = make_property_tensor(mesh, sigma_isotropic)
    >>> M_anisotropic = make_property_tensor(mesh, sigma_anisotropic)
    >>> M_tensor = make_property_tensor(mesh, sigma_tensor)

    Plot the property tensors.

    .. collapse:: Expand to show scripting for plot

        >>> M_list = [M_scalar, M_isotropic, M_anisotropic, M_tensor]
        >>> case_list = ['Scalar', 'Isotropic', 'Anisotropic', 'Full Tensor']
        >>> ax1 = 4*[None]
        >>> fig = plt.figure(figsize=(15, 4))
        >>> for ii in range(0, 4):
        ...     ax1[ii] = fig.add_axes([0.05+0.22*ii, 0.05, 0.18, 0.9])
        ...     ax1[ii].imshow(
        ...         M_list[ii].todense(), interpolation='none', cmap='binary', vmax=10.
        ...     )
        ...     ax1[ii].set_title(case_list[ii], fontsize=24)
        >>> ax2 = fig.add_axes([0.92, 0.15, 0.01, 0.7])
        >>> norm = mpl.colors.Normalize(vmin=0., vmax=10.)
        >>> cbar = mpl.colorbar.ColorbarBase(
        ...     ax2, norm=norm, orientation="vertical", cmap=mpl.cm.binary
        ... )
        >>> plt.show()
    """
    if tensor is None:  # default is ones
        tensor = np.ones(mesh.nC)

    if is_scalar(tensor):
        tensor = tensor * np.ones(mesh.nC)

    propType = TensorType(mesh, tensor)
    if propType == 1:  # Isotropic!
        Sigma = sp.kron(sp.identity(mesh.dim), sdiag(mkvc(tensor)))
    elif propType == 2:  # Diagonal tensor
        Sigma = sdiag(mkvc(tensor))
    elif mesh.dim == 2 and tensor.size == mesh.nC * 3:  # Fully anisotropic, 2D
        tensor = tensor.reshape((mesh.nC, 3), order="F")
        row1 = sp.hstack((sdiag(tensor[:, 0]), sdiag(tensor[:, 2])))
        row2 = sp.hstack((sdiag(tensor[:, 2]), sdiag(tensor[:, 1])))
        Sigma = sp.vstack((row1, row2))
    elif mesh.dim == 3 and tensor.size == mesh.nC * 6:  # Fully anisotropic, 3D
        tensor = tensor.reshape((mesh.nC, 6), order="F")
        row1 = sp.hstack(
            (sdiag(tensor[:, 0]), sdiag(tensor[:, 3]), sdiag(tensor[:, 4]))
        )
        row2 = sp.hstack(
            (sdiag(tensor[:, 3]), sdiag(tensor[:, 1]), sdiag(tensor[:, 5]))
        )
        row3 = sp.hstack(
            (sdiag(tensor[:, 4]), sdiag(tensor[:, 5]), sdiag(tensor[:, 2]))
        )
        Sigma = sp.vstack((row1, row2, row3))
    else:
        raise Exception("Unexpected shape of tensor")

    return Sigma


def inverse_property_tensor(mesh, tensor, return_matrix=False, **kwargs):
    r"""Construct the inverse of the physical property tensor.

    For a given *mesh*, the input parameter *tensor* is a :class:`numpy.ndarray`
    defining the constitutive relationship (e.g. Ohm's law) between two
    discrete vector quantities :math:`\boldsymbol{j}` and
    :math:`\boldsymbol{e}` living at cell centers. Where :math:`\boldsymbol{M}`
    is the physical property tensor, **inverse_property_tensor**
    explicitly constructs the inverse of the physical
    property tensor :math:`\boldsymbol{M^{-1}}` for all cells such that:

    >>> e = Mi @ j

    where the Cartesian components of the discrete vectors are
    organized according to:

    >>> j = np.r_[jx, jy, jz]
    >>> e = np.r_[ex, ey, ez]

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
       A mesh
    tensor : numpy.ndarray or float
        - *Scalar:* A float is entered.
        - *Isotropic:* A 1D numpy.ndarray with a property value for every cell.
        - *Anisotropic:* A (*nCell*, *dim*) numpy.ndarray where each row
          defines the diagonal-anisotropic property parameters for each cell.
          *nParam* = 2 for 2D meshes and *nParam* = 3 for 3D meshes.
        - *Tensor:* A (*nCell*, *nParam*) numpy.ndarray where each row defines
          the full anisotropic property parameters for each cell. *nParam* = 3 for 2D
          meshes and *nParam* = 6 for 3D meshes.

    return_matrix : bool, optional
        - *True:* the function returns the inverse of the property tensor.
        - *False:* the function returns the non-zero elements of the inverse of the
          property tensor in a numpy.ndarray in the same order as the input argument
          *tensor*.

    Returns
    -------
    numpy.ndarray or scipy.sparse.coo_matrix
        - If *return_matrix* = *False*, the function outputs the parameters defining the
          inverse of the property tensor in a numpy.ndarray with the same dimensions as
          the input argument *tensor*
        - If *return_natrix* = *True*, the function outputs the inverse of the property
          tensor as a *scipy.sparse.coo_matrix*.

    Notes
    -----
    The relationship between a quantity and its response to external
    stimuli (e.g. Ohm's law) in each cell can be defined by a scalar
    function :math:`\sigma` in the isotropic case, or by a tensor
    :math:`\Sigma` in the anisotropic case, i.e.:

    .. math::
        \vec{j} = \sigma \vec{e} \;\;\;\;\;\; \textrm{or} \;\;\;\;\;\;
        \vec{j} = \Sigma \vec{e}

    where

    .. math::
        \Sigma = \begin{bmatrix}
        \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
        \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\
        \sigma_{xz} & \sigma_{yz} & \sigma_{zz}
        \end{bmatrix}

    Examples
    --------
    For the 4 classifications allowable (scalar, isotropic, anistropic and tensor),
    we construct the property tensor on a small 2D mesh. We then construct the
    inverse of the property tensor and compare.

    >>> from discretize.utils import make_property_tensor, inverse_property_tensor
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib as mpl

    Define a 2D tensor mesh

    >>> h = [1., 1., 1.]
    >>> mesh = TensorMesh([h, h], origin='00')

    Define a physical property for all cases (2D)

    >>> sigma_scalar = 4.
    >>> sigma_isotropic = np.random.randint(1, 10, mesh.nC)
    >>> sigma_anisotropic = np.random.randint(1, 10, (mesh.nC, 2))
    >>> sigma_tensor = np.random.randint(1, 10, (mesh.nC, 3))

    Construct the property tensor in each case

    >>> M_scalar = make_property_tensor(mesh, sigma_scalar)
    >>> M_isotropic = make_property_tensor(mesh, sigma_isotropic)
    >>> M_anisotropic = make_property_tensor(mesh, sigma_anisotropic)
    >>> M_tensor = make_property_tensor(mesh, sigma_tensor)

    Construct the inverse property tensor in each case

    >>> Minv_scalar = inverse_property_tensor(mesh, sigma_scalar, return_matrix=True)
    >>> Minv_isotropic = inverse_property_tensor(mesh, sigma_isotropic, return_matrix=True)
    >>> Minv_anisotropic = inverse_property_tensor(mesh, sigma_anisotropic, return_matrix=True)
    >>> Minv_tensor = inverse_property_tensor(mesh, sigma_tensor, return_matrix=True)

    Plot the property tensors.

    .. collapse:: Expand to show scripting for plot

        >>> M_list = [M_scalar, M_isotropic, M_anisotropic, M_tensor]
        >>> Minv_list = [Minv_scalar, Minv_isotropic, Minv_anisotropic, Minv_tensor]
        >>> case_list = ['Scalar', 'Isotropic', 'Anisotropic', 'Full Tensor']
        >>> fig1 = plt.figure(figsize=(15, 4))
        >>> ax1 = 4*[None]
        >>> for ii in range(0, 4):
        ...     ax1[ii] = fig1.add_axes([0.05+0.22*ii, 0.05, 0.18, 0.9])
        ...     ax1[ii].imshow(
        ...         M_list[ii].todense(), interpolation='none', cmap='binary', vmax=10.
        ...     )
        ...     ax1[ii].set_title('$M$ (' + case_list[ii] + ')', fontsize=24)
        >>> cax1 = fig1.add_axes([0.92, 0.15, 0.01, 0.7])
        >>> norm1 = mpl.colors.Normalize(vmin=0., vmax=10.)
        >>> cbar1 = mpl.colorbar.ColorbarBase(
        ...     cax1, norm=norm1, orientation="vertical", cmap=mpl.cm.binary
        ... )
        >>> plt.show()

    Plot the inverse property tensors.

    .. collapse:: Expand to show scripting for plot

        >>> fig2 = plt.figure(figsize=(15, 4))
        >>> ax2 = 4*[None]
        >>> for ii in range(0, 4):
        ...     ax2[ii] = fig2.add_axes([0.05+0.22*ii, 0.05, 0.18, 0.9])
        ...     ax2[ii].imshow(
        ...         Minv_list[ii].todense(), interpolation='none', cmap='binary', vmax=1.
        ...     )
        ...     ax2[ii].set_title('$M^{-1}$ (' + case_list[ii] + ')', fontsize=24)
        >>> cax2 = fig2.add_axes([0.92, 0.15, 0.01, 0.7])
        >>> norm2 = mpl.colors.Normalize(vmin=0., vmax=1.)
        >>> cbar2 = mpl.colorbar.ColorbarBase(
        ...     cax2, norm=norm2, orientation="vertical", cmap=mpl.cm.binary
        ... )
        >>> plt.show()
    """
    if "returnMatrix" in kwargs:
        warnings.warn(
            "The returnMatrix keyword argument has been deprecated, please use return_matrix. "
            "This will be removed in discretize 1.0.0",
            FutureWarning,
        )
        return_matrix = kwargs["returnMatrix"]

    propType = TensorType(mesh, tensor)

    if is_scalar(tensor):
        T = 1.0 / tensor
    elif propType < 3:  # Isotropic or Diagonal
        T = 1.0 / mkvc(tensor)  # ensure it is a vector.
    elif mesh.dim == 2 and tensor.size == mesh.nC * 3:  # Fully anisotropic, 2D
        tensor = tensor.reshape((mesh.nC, 3), order="F")
        B = inverse_2x2_block_diagonal(
            tensor[:, 0], tensor[:, 2], tensor[:, 2], tensor[:, 1], return_matrix=False
        )
        b11, b12, b21, b22 = B
        T = np.r_[b11, b22, b12]
    elif mesh.dim == 3 and tensor.size == mesh.nC * 6:  # Fully anisotropic, 3D
        tensor = tensor.reshape((mesh.nC, 6), order="F")
        B = inverse_3x3_block_diagonal(
            tensor[:, 0],
            tensor[:, 3],
            tensor[:, 4],
            tensor[:, 3],
            tensor[:, 1],
            tensor[:, 5],
            tensor[:, 4],
            tensor[:, 5],
            tensor[:, 2],
            return_matrix=False,
        )
        b11, b12, b13, b21, b22, b23, b31, b32, b33 = B
        T = np.r_[b11, b22, b33, b12, b13, b23]
    else:
        raise Exception("Unexpected shape of tensor")

    if return_matrix:
        return make_property_tensor(mesh, T)

    return T


class Zero(object):
    """Carries out arithmetic operations between 0 and arbitrary quantities.

    This class was designed to manage basic arithmetic operations between
    0 and :class:`numpy.ndarray` of any shape. It is a short circuiting evaluation that
    will return the expected values.

    Examples
    --------
    >>> import numpy as np
    >>> from discretize.utils import Zero
    >>> Z = Zero()
    >>> Z
    Zero
    >>> x = np.arange(5)
    >>> x + Z
    array([0, 1, 2, 3, 4])
    >>> Z - x
    array([ 0, -1, -2, -3, -4])
    >>> Z * x
    Zero
    >>> Z @ x
    Zero
    >>> Z[0]
    Zero
    """

    __numpy_ufunc__ = True
    __array_ufunc__ = None

    def __repr__(self):
        """Represent zeros a string."""
        return "Zero"

    def __add__(self, v):
        """Add a value to zero."""
        return v

    def __radd__(self, v):
        """Add zero to a value."""
        return v

    def __iadd__(self, v):
        """Add zero to a value inplace."""
        return v

    def __sub__(self, v):
        """Subtract a value from zero."""
        return -v

    def __rsub__(self, v):
        """Subtract zero from a value."""
        return v

    def __isub__(self, v):
        """Subtract zero from a value inplace."""
        return v

    def __mul__(self, v):
        """Multiply zero by a value."""
        return self

    def __rmul__(self, v):
        """Multiply a value by zero."""
        return self

    def __matmul__(self, v):
        """Multiply zero by a matrix."""
        return self

    def __rmatmul__(self, v):
        """Multiply a matrix by zero."""
        return self

    def __div__(self, v):
        """Divide zero by a value."""
        return self

    def __truediv__(self, v):
        """Divide zero by a value."""
        return self

    def __rdiv__(self, v):
        """Try to divide a value by zero."""
        raise ZeroDivisionError("Cannot divide by zero.")

    def __rtruediv__(self, v):
        """Try to divide a value by zero."""
        raise ZeroDivisionError("Cannot divide by zero.")

    def __rfloordiv__(self, v):
        """Try to divide a value by zero."""
        raise ZeroDivisionError("Cannot divide by zero.")

    def __pos__(self):
        """Return zero."""
        return self

    def __neg__(self):
        """Negate zero."""
        return self

    def __lt__(self, v):
        """Compare less than zero."""
        return 0 < v

    def __le__(self, v):
        """Compare less than or equal to zero."""
        return 0 <= v

    def __eq__(self, v):
        """Compare equal to zero."""
        return v == 0

    def __ne__(self, v):
        """Compare not equal to zero."""
        return not (0 == v)

    def __ge__(self, v):
        """Compare greater than or equal to zero."""
        return 0 >= v

    def __gt__(self, v):
        """Compare greater than zero."""
        return 0 > v

    def transpose(self):
        """Return the transpose of the *Zero* class, i.e. itself."""
        return self

    def __getitem__(self, key):
        """Get an element of the *Zero* class, i.e. itself."""
        return self

    @property
    def ndim(self):
        """Return the dimension of *Zero* class, i.e. *None*."""
        return None

    @property
    def shape(self):
        """Return the shape *Zero* class, i.e. *None*."""
        return _inftup(None)

    @property
    def T(self):
        """Return the *Zero* class as an operator."""
        return self


class Identity(object):
    """Carries out arithmetic operations involving the identity.

    This class was designed to manage basic arithmetic operations between the identity
    matrix and :class:`numpy.ndarray` of any shape. It is a short circuiting evaluation
    that will return the expected values.

    Parameters
    ----------
    positive : bool, optional
        Whether it is a positive (or negative) Identity matrix

    Examples
    --------
    >>> import numpy as np
    >>> from discretize.utils import Identity, Zero
    >>> Z = Zero()
    >>> I = Identity()
    >>> x = np.arange(5)
    >>> x + I
    array([1, 2, 3, 4, 5])
    >>> I - x
    array([ 1, 0, -1, -2, -3])
    >>> I * x
    array([0, 1, 2, 3, 4])
    >>> I @ x
    array([0, 1, 2, 3, 4])
    >>> I @ Z
    Zero
    """

    __numpy_ufunc__ = True
    __array_ufunc__ = None

    _positive = True

    def __init__(self, positive=True):
        self._positive = positive

    def __repr__(self):
        """Represent 1 (or -1 if not positive)."""
        if self._positive:
            return "I"
        else:
            return "-I"

    def __pos__(self):
        """Return positive 1 (or -1 if not positive)."""
        return self

    def __neg__(self):
        """Negate 1 (or -1 if not positive)."""
        return Identity(not self._positive)

    def __add__(self, v):
        """Add 1 (or -1 if not positive) to a value."""
        if sp.issparse(v):
            return v + speye(v.shape[0]) if self._positive else v - speye(v.shape[0])
        return v + 1 if self._positive else v - 1

    def __radd__(self, v):
        """Add 1 (or -1 if not positive) to a value."""
        return self.__add__(v)

    def __sub__(self, v):
        """Subtract a value from 1 (or -1 if not positive)."""
        return self + -v

    def __rsub__(self, v):
        """Subtract 1 (or -1 if not positive) from a value."""
        return -self + v

    def __mul__(self, v):
        """Multiply 1 (or -1 if not positive) by a value."""
        return v if self._positive else -v

    def __rmul__(self, v):
        """Multiply 1 (or -1 if not positive) by a value."""
        return v if self._positive else -v

    def __matmul__(self, v):
        """Multiply 1 (or -1 if not positive) by a matrix."""
        return v if self._positive else -v

    def __rmatmul__(self, v):
        """Multiply a matrix by 1 (or -1 if not positive)."""
        return v if self._positive else -v

    def __div__(self, v):
        """Divide 1 (or -1 if not positive) by a value."""
        if sp.issparse(v):
            raise NotImplementedError("Sparse arrays not divisibile.")
        return 1 / v if self._positive else -1 / v

    def __truediv__(self, v):
        """Divide 1 (or -1 if not positive) by a value."""
        if sp.issparse(v):
            raise NotImplementedError("Sparse arrays not divisibile.")
        return 1.0 / v if self._positive else -1.0 / v

    def __rdiv__(self, v):
        """Divide a value by 1 (or -1 if not positive)."""
        return v if self._positive else -v

    def __rtruediv__(self, v):
        """Divide a value by 1 (or -1 if not positive)."""
        return v if self._positive else -v

    def __floordiv__(self, v):
        """Flooring division of 1 (or -1 if not positive) by a value."""
        return 1 // v if self._positive else -1 // v

    def __rfloordiv__(self, v):
        """Flooring division of a value by 1 (or -1 if not positive)."""
        return v // 1 if self._positive else v // -1

    def __lt__(self, v):
        """Compare less than 1 (or -1 if not positive)."""
        return 1 < v if self._positive else -1 < v

    def __le__(self, v):
        """Compare less than or equal to 1 (or -1 if not positive)."""
        return 1 <= v if self._positive else -1 <= v

    def __eq__(self, v):
        """Compare equal to 1 (or -1 if not positive)."""
        return v == 1 if self._positive else v == -1

    def __ne__(self, v):
        """Compare not equal to 1 (or -1 if not positive)."""
        return (not (1 == v)) if self._positive else (not (-1 == v))

    def __ge__(self, v):
        """Compare greater than or equal to 1 (or -1 if not positive)."""
        return 1 >= v if self._positive else -1 >= v

    def __gt__(self, v):
        """Compare greater than 1 (or -1 if not positive)."""
        return 1 > v if self._positive else -1 > v

    @property
    def ndim(self):
        """Return the dimension of *Identity* class, i.e. *None*."""
        return None

    @property
    def shape(self):
        """Return the shape of *Identity* class, i.e. *None*."""
        return _inftup(None)

    @property
    def T(self):
        """Return the *Identity* class as an operator."""
        return self

    def transpose(self):
        """Return the transpose of the *Identity* class, i.e. itself."""
        return self


class _inftup(tuple):
    """An infinitely long tuple of a value repeated infinitely."""

    def __init__(self, val=None):
        self._val = val

    def __getitem__(self, key):
        if isinstance(key, slice):
            return _inftup(self._val)
        return self._val

    def __len__(self):
        return 0

    def __repr__(self):
        return f"({self._val}, {self._val}, ...)"


################################################
#             DEPRECATED FUNCTIONS
################################################

sdInv = deprecate_function(sdinv, "sdInv", removal_version="1.0.0", future_warn=True)

getSubArray = deprecate_function(
    get_subarray, "getSubArray", removal_version="1.0.0", future_warn=True
)

inv3X3BlockDiagonal = deprecate_function(
    inverse_3x3_block_diagonal,
    "inv3X3BlockDiagonal",
    removal_version="1.0.0",
    future_warn=True,
)

inv2X2BlockDiagonal = deprecate_function(
    inverse_2x2_block_diagonal,
    "inv2X2BlockDiagonal",
    removal_version="1.0.0",
    future_warn=True,
)

makePropertyTensor = deprecate_function(
    make_property_tensor,
    "makePropertyTensor",
    removal_version="1.0.0",
    future_warn=True,
)

invPropertyTensor = deprecate_function(
    inverse_property_tensor,
    "invPropertyTensor",
    removal_version="1.0.0",
    future_warn=True,
)
