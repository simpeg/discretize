import numpy as np
import scipy.sparse as sp
from discretize.utils.code_utils import is_scalar, deprecate_function
import warnings


def mkvc(x, n_dims=1, **kwargs):
    """Creates a vector with specified dimensionality.

    This function converts a numpy.ndarray to a vector. In general,
    the output vector has a dimension of 1. However, the dimensionality
    can be specified if the user intends to carry out a dot product with
    a higher order array.

    Parameters
    ----------
    x : numpy.ndarray
        An nD array that will be reorganized and output as a vector
    n_dims : int
        The dimension of the output vector.

    Returns
    -------
    numpy.ndarray
        The output vector

    Examples
    --------

    Here, we reorganize a simple 2D array as a vector and demonstrate the
    impact of the *n_dim* argument.

    >>> from discretize.utils import mkvc
    >>> import numpy as np
    >>> 
    >>> a = np.random.rand(3, 2)
    >>> print('Original array:')
    >>> print(a)
    >>> 
    >>> v = mkvc(a)
    >>> print('Vector with n_dim = 1:')
    >>> print(v)
    >>> 
    >>> print('Higher dimensions:')
    >>> for ii in range(1, 4):
    >>>     temp = mkvc(a, ii)
    >>>     print('Shape of output with n_dim =', ii, ': ', np.shape(temp))


    """
    if "numDims" in kwargs:
        warnings.warn(
            "The numDims keyword argument has been deprecated, please use n_dims. "
            "This will be removed in discretize 1.0.0",
            DeprecationWarning,
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
    v : numpy.ndarray
        The vector defining the diagonal elements of the sparse matrix being constructed

    Returns
    -------
    scipy.sparse.csr_matrix
        The sparse diagonal matrix.

    Examples
    --------

    Use a 1D array of values to construct a sparse diagonal matrix.

    >>> from discretize.utils import sdiag
    >>> import numpy as np
    >>> 
    >>> v = np.array([6., 3., 1., 8., 0., 5.])
    >>> M = sdiag(v)

    """
    if isinstance(v, Zero):
        return Zero()

    return sp.spdiags(mkvc(v), 0, v.size, v.size, format="csr")


def sdinv(M):
    """Return inverse of a sparse diagonal matrix

    This function extracts the diagonal elements of the input matrix *M*
    and creates a sparse diagonal matrix from the reciprocal these elements.
    If the input matrix *M* is diagonal, the output is the inverse of *M*.

    Parameters
    ----------
    M : scipy.sparse.csr_matrix
        A sparse diagonal matrix

    Returns
    -------
    scipy.sparse.csr_matrix
        The inverse of the sparse diagonal matrix.

    Examples
    --------

    >>> from discretize.utils import sdiag, sdinv
    >>> import numpy as np
    >>> 
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
    scipy.sparse.csr_matrix
        The sparse identity matrix.

    """
    return sp.identity(n, format="csr")


def kron3(A, B, C):
    r"""Compute kronecker products between 3 sparse matricies.

    Where :math:`\otimes` denotes the Kronecker product and *A, B* and *C* are
    sparse matrices, this function outputs :math:`(A \otimes B) \otimes C`.
    
    Parameters
    ----------
    A, B, C : scipy.sparse.xxx_matrix
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
    scipy.sparse.dia_matrix
        A sparse matrix of zeros.

    """
    return sp.dia_matrix((n1, n2))


def ddx(n):
    """Create 1D difference (derivative) operator from nodes to centers.

    For n cells, the 1D difference (derivative) operator from nodes to
    centers is sparse, has shape (n, n+1) and takes the form:

    .. math::
        \\begin{bmatrix}
        -1 & \\; 1 & & & \n
        & -1 & \\; 1 & & \n
        & & \\ddots & \\ddots & \n
        & & & -1 & \\; 1 \\;
        \\end{bmatrix}

    Parameters
    ----------
    n : int
        Number of cells

    Returns
    -------
    scipy.sparse.csr_matrix
        The 1D difference operator from nodes to centers.

    """
    return sp.spdiags((np.ones((n + 1, 1)) * [-1, 1]).T, [0, 1], n, n + 1, format="csr")


def av(n):
    """Create 1D averaging operator from nodes to cell-centers.

    For n cells, the 1D averaging operator from nodes to centerss
    is sparse, has shape (n, n+1) and takes the form:

    .. math::
        \\begin{bmatrix}
        1/2 & 1/2 & & & \n
        & 1/2 & 1/2 & & \n
        & & \\ddots & \\ddots & \n
        & & & 1/2 & 1/2 \\;
        \\end{bmatrix}

    Parameters
    ----------
    n : int
        Number of cells

    Returns
    -------
    scipy.sparse.csr_matrix
        The 1D averaging operator from nodes to centers.

    """
    return sp.spdiags(
        (0.5 * np.ones((n + 1, 1)) * [1, 1]).T, [0, 1], n, n + 1, format="csr"
    )


def av_extrap(n):
    """Create 1D averaging operator from cell-centers to nodes.
    
    For n cells, the 1D averaging operator from cell centers to nodes
    is sparse and has shape (n+1, n). Values at the outmost nodes are
    extrapolated from the nearest cell center value. Thus the operator
    takes the form:

    .. math::
        \\begin{bmatrix}
        1 & & & & \n
        1/2 & 1/2 & & & \n
        & 1/2 & 1/2 & & & \n
        & & \\ddots & \\ddots & \n
        & & & 1/2 & 1/2 \n
        & & & & 1 \\;
        \\end{bmatrix}

    Parameters
    ----------
    n : int
        Number of cells

    Returns
    -------
    scipy.sparse.csr_matrix
        The 1D averaging operator from cell-centers to nodes.

    """
    Av = sp.spdiags(
        (0.5 * np.ones((n, 1)) * [1, 1]).T, [-1, 0], n + 1, n, format="csr"
    ) + sp.csr_matrix(([0.5, 0.5], ([0, n], [0, n - 1])), shape=(n + 1, n))
    return Av


def ndgrid(*args, **kwargs):
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
    args : numpy.ndarray or a list of numpy.ndarray
        Positions along each axis of the tensor. The user can define these as
        successive positional arguments *x*, *y*, (and *z*) or as a single argument
        using a list [*x*, *y*, (*z*)].
    vector : bool, optional kwargs
        If *True*, the output is a numpy array of dimension [n, ndim]. If *False*,
        the gridded x, y (and z) locations are returned as separate ndarrays in a list.
        Default is *True*.
    order : str, optional kwargs
        Define ordering using one of the following options {'C', 'F', 'A'}.
        'C' is C-like ordering. 'F' is Fortran-like ordering. 'A' is Fortran
        ordering if memory is contigious and C-like otherwise. Default = 'F'.
        See numpy.reshape for more on this argument.
    

    Returns
    -------
    numpy.ndarray or list of numpy.array
        If *vector=True* the gridded *x*, *y*, (and *z*) locations are
        returned as a numpy array of shape [n, ndim]. If *vector=False*,
        the gridded *x*, *y*, (and *z*) are returned as a list of vectors.
    

    Examples
    --------

    >>> from discretize.utils import ndgrid
    >>> import numpy as np
    >>> 
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([2, 4])
    >>> 
    >>> xy = ndgrid([x, y])
    >>> print('Gridded x,y locations with F ordering:')
    >>> print(xy)
    >>> 
    >>> xy = ndgrid([x, y], order='C')
    >>> print('Gridded x,y locations with C ordering:')
    >>> print(xy)
    >>> 
    >>> xy = ndgrid(x, y, vector=False)
    >>> print('X location tensor:')
    >>> print(xy[0])
    >>> print('Y location tensor:')
    >>> print(xy[1])


    """

    # Read the keyword arguments, and only accept a vector=True/False
    vector = kwargs.pop("vector", True)
    order = kwargs.pop("order", "F")
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


def make_boundary_bool(shape, dir="xyz"):
    r"""Return boundary indices of a tensor grid.

    For a tensor grid whose shape is given (1D, 2D or 3D), this function
    returns a boolean index array identifying the x, y and/or z
    boundary locations.

    Parameters
    ----------
    shape : tuple of int
        Defines the shape of the tensor (1D, 2D or 3D).
    dir : str containing characters 'x', 'y' and/or 'z'
        Specify the boundaries whose indices you want returned; e.g. for a 3D
        tensor, you may set *dir* = 'xz' to return the indices of the x and
        z boundary locations.

    Returns
    -------
    numpy.array of bool
        Indices of boundary locations of the tensor for specified boundaries.

    Examples
    --------

    Here we construct a 3x3 tensor and find the indices of the boundary locations.

    >>> from discretize.utils.matrix_utils import ndgrid, make_boundary_bool
    >>> import numpy as np
    >>> 
    >>> # Define 3x3 tensor grid
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([2, 4, 6])
    >>> tensor_grid = ndgrid([x, y])
    >>> print('TENSOR GRID:')
    >>> print(tensor_grid)
    >>> 
    >>> # Find indices of boundary locations.
    >>> shape = (len(x), len(y))
    >>> bool_ind = make_boundary_bool(shape)
    >>> print('\nBOUNDARY LOCATIONS:')
    >>> print(bool_ind)
    >>> 
    >>> # Find indices of locations on x boundaries
    >>> bool_ind_x = make_boundary_bool(shape, 'x')
    >>> print('X-BOUNDARY LOCATIONS:')
    >>> print(bool_ind_x)

    """
    is_b = np.zeros(shape, dtype=bool, order="F")
    if "x" in dir:
        is_b[[0, -1]] = True
    if len(shape) > 1:
        if "y" in dir:
            is_b[:, [0, -1]] = True
    if len(shape) > 2:
        if "z" in dir:
            is_b[:, :, [0, -1]] = True
    return is_b.reshape(-1, order="F")


def ind2sub(shape, inds):
    r"""Return subscripts of tensor grid elements from indices.

    Consider the :math:`n^{th}` element of a tensor grid with *N* elements.
    The position of this element in the tensor grid can also be defined by
    subscripts (i,j,k). For an array containing the indices for a set of
    tensor elements, this function returns the corresponding subscripts.

    Parameters
    ----------
    shape : tuple of int
        Defines the shape of the tensor (1D, 2D or 3D).
    inds : list or numpy.array of int
        The indices of the tensor elements whose subscripts you want returned.

    Returns
    -------
    tuple of ndarray
        Corresponding subscipts for the indices provided. The output is a
        tuple containing 1D integer arrays for the i, j and k subscripts, respectively.
        The size of the output tuple is equal to the dimension of the tensor grid
        (1D, 2D or 3D).

    Examples
    --------

    Here we construct a 3x3 tensor and return the subscripts for
    a set of indices.

    >>> from discretize.utils.matrix_utils import ndgrid, ind2sub
    >>> import numpy as np
    >>> 
    >>> # Define 3x3 tensor grid
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([2, 4, 6])
    >>> tensor_grid = ndgrid([x, y])
    >>> print('3x3 TENSOR GRID:')
    >>> print(tensor_grid)
    >>> 
    >>> # Define indices of elements whose subscipts you want
    >>> shape = (len(x), len(y))
    >>> ind = np.array([0, 4, 7], dtype=int)
    >>> print('\nINDICES:')
    >>> print(ind)
    >>> 
    >>> # Return the subscipts
    >>> I, J = ind2sub(shape, ind)
    >>> print('\nSUBSCRIPTS (i,j):')
    >>> for kk in range(0, len(ind)):
    >>>     print('({}, {})'.format(I[kk], J[kk]))


    """
    if type(inds) is not np.ndarray:
        inds = np.array(inds)
    if len(inds.shape) != 1:
        raise ValueError("Indexing must be done as a 1D row vector, e.g. [3,6,6,...]")
    return np.unravel_index(inds, shape, order="F")


def sub2ind(shape, subs):
    r"""Return indices of tensor grid elements from subscripts.

    Consider elements of a tensors grid whose positions are given by the
    subscripts (i,j,k). This function will return the corresponding indices
    of these elements. Each row of the input array *subs* defines the
    ijk for a particular tensor element.

    Parameters
    ----------
    shape : tuple of int
        Defines the shape of the tensor (1D, 2D or 3D).
    subs : ndarray of int
        The subscripts of the tensor grid elements. Each rows defines the position
        of a particular tensor element. The shape of of the array is (N, ndim). 

    Returns
    -------
    ndarray of int
        The indices of the tensor grid elements defined by *subs*.

    Examples
    --------

    Here we construct a 3x3 tensor and return the indices for
    a set of subscripts.

    >>> from discretize.utils.matrix_utils import ndgrid, ind2sub
    >>> import numpy as np
    >>> 
    >>> # Define 3x3 tensor grid
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([2, 4, 6])
    >>> tensor_grid = ndgrid([x, y])
    >>> print('3x3 TENSOR GRID:')
    >>> print(tensor_grid)
    >>> 
    >>> # Define subscripts (i,j)
    >>> I = np.array([0, 1, 1], dtype=int)
    >>> J = np.array([0, 1, 2], dtype=int)
    >>> print('\nSUBSCRIPTS (i,j):')
    >>> for kk in range(0, len(I)):
    >>>     print('({}, {})'.format(I[kk], J[kk]))
    >>> 
    >>> # Find the corresponding indices
    >>> shape = (len(x), len(y))
    >>> ind = sub2ind(shape, np.c_[I, J])
    >>> print('\nINDICES:')
    >>> print(ind)

    """
    if len(shape) == 1:
        return subs
    if type(subs) is not np.ndarray:
        subs = np.array(subs)
    if len(subs.shape) == 1:
        subs = subs[np.newaxis, :]
    if subs.shape[1] != len(shape):
        raise ValueError(
            "Indexing must be done as a column vectors. e.g. [[3,6],[6,2],...]"
        )
    inds = np.ravel_multi_index(subs.T, shape, order="F")
    return mkvc(inds)


def get_subarray(A, ind):
    """subarray"""
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
    """Invert a set of 3x3 matricies from vectors containing their elements.

    The elements of a 3x3 matrix *A* are given by:

    .. math::
        A = \\begin{bmatrix}
        a_{11} & a_{12} & a_{13} \n
        a_{21} & a_{22} & a_{23} \n
        a_{31} & a_{32} & a_{33}
        \\end{bmatrix}

    For a set of 3x3 matricies, the elements may be stored in a set of 9 distinct vectors
    :math:`\\mathbf{a_{11}}`, :math:`\\mathbf{a_{12}}`, ..., :math:`\\mathbf{a_{33}}`.
    For each matrix, **inverse_3x3_block_diagonal** ouputs the vectors containing the
    elements of each matrix' inverse; i.e.
    :math:`\\mathbf{b_{11}}`, :math:`\\mathbf{b_{12}}`, ..., :math:`\\mathbf{b_{33}}`
    where:

    .. math::
        A^{-1} = B = \\begin{bmatrix}
        b_{11} & b_{12} & b_{13} \n
        b_{21} & b_{22} & b_{23} \n
        b_{31} & b_{32} & b_{33}
        \\end{bmatrix}

    For special applications, we may want to output the elements of the inverses
    of the matricies as a 3x3 block matrix of the form:

    .. math::
        M = \\begin{bmatrix}
        D_{11} & D_{12} & D_{13} \n
        D_{21} & D_{22} & D_{23} \n
        D_{31} & D_{32} & D_{33} 
        \\end{bmatrix}

    where :math:`D_{ij}` are diagonal matrices whose non-zero elements
    are defined by vector :math:`\\mathbf{b_{ij}}`. Where *n* is the
    number of matricies, the block matrix is sparse with dimensions
    (3n, 3n).


    Parameters
    ----------
    aij : numpy.ndarray
        All arguments a11, a12, ..., a33 are vectors which contain the
        corresponding element for all 3x3 matricies
    return_matrix : bool, optional

        - **True**: Returns the sparse block 3x3 matrix *M* (default).
        - **False:** Returns the vectors containing the elements of each matrix' inverse.


    Returns
    -------
    sparse.coo.coo_matrix or list of numpy.ndarray
        If *return_matrix = False*, the function will return vectors
        *b11, b12, b13, b21, b22, b23, b31, b32, b33*.
        If *return_matrix = True*, the function will return the
        block matrix *M*.

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
    >>> 
    >>> # Define four 3x3 matricies
    >>> np.random.seed(45)
    >>> A1 = np.random.uniform(1, 10, (3, 3))
    >>> A2 = np.random.uniform(1, 10, (3, 3))
    >>> A3 = np.random.uniform(1, 10, (3, 3))
    >>> A4 = np.random.uniform(1, 10, (3, 3))
    >>> 
    >>> # Organize elements in vectors a11 to a33
    >>> v = []
    >>> for ii in range(0, 3):
    >>>     for jj in range(0, 3):
    >>>         v.append(
    >>>             np.c_[A1[ii, jj], A2[ii, jj], A3[ii, jj], A4[ii, jj]]
    >>>         )
    >>> 
    >>> a11, a12, a13, a21, a22, a23, a31, a32, a33 = v
    >>> 
    >>> # Return the elements of their inverse and validate
    >>> b11, b12, b13, b21, b22, b23, b31, b32, b33 = inverse_3x3_block_diagonal(
    >>>     a11, a12, a13, a21, a22, a23, a31, a32, a33, return_matrix=False
    >>> )
    >>> 
    >>> B = []
    >>> for ii in range(0, 4):
    >>>     B.append(
    >>>         np.r_[
    >>>             np.c_[b11[ii], b12[ii], b13[ii]],
    >>>             np.c_[b21[ii], b22[ii], b23[ii]],
    >>>             np.c_[b31[ii], b32[ii], b33[ii]],
    >>>         ]
    >>>     )
    >>>     
    >>> B1, B2, B3, B4 = B
    >>> 
    >>> print('Inverse of A1 using numpy.linalg.inv')
    >>> print(np.linalg.inv(A1))
    >>> print('B1 constructed from vectors b11 to b33')
    >>> print(B1)
    >>> 
    >>> # Plot the sparse block matrix containing elements of the inverses
    >>> M = inverse_3x3_block_diagonal(
    >>>     a11, a12, a13, a21, a22, a23, a31, a32, a33
    >>> )
    >>> 
    >>> plt.spy(M)


    """
    if "returnMatrix" in kwargs:
        warnings.warn(
            "The returnMatrix keyword argument has been deprecated, please use return_matrix. "
            "This will be removed in discretize 1.0.0",
            DeprecationWarning,
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
    """
    Invert a set of 2x2 matricies from vectors containing their elements.

    The elements of a 2x2 matrix *A* are given by:

    .. math::
        A = \\begin{bmatrix}
        a_{11} & a_{12} \n
        a_{21} & a_{22}
        \\end{bmatrix}

    For a set of 2x2 matricies, the elements may be stored in a set of 4 distinct vectors
    :math:`\\mathbf{a_{11}}`, :math:`\\mathbf{a_{12}}`, :math:`\\mathbf{a_{21}}` and
    :math:`\\mathbf{a_{22}}`.
    For each matrix, **inverse_2x2_block_diagonal** ouputs the vectors containing the
    elements of each matrix' inverse; i.e.
    :math:`\\mathbf{b_{11}}`, :math:`\\mathbf{b_{12}}`, :math:`\\mathbf{b_{21}}` and
    :math:`\\mathbf{b_{22}}` where:

    .. math::
        A^{-1} = B = \\begin{bmatrix}
        b_{11} & b_{12} \n
        b_{21} & b_{22}
        \\end{bmatrix}

    For special applications, we may want to output the elements of the inverses
    of the matricies as a 2x2 block matrix of the form:

    .. math::
        M = \\begin{bmatrix}
        D_{11} & D_{12} \n
        D_{21} & D_{22} 
        \\end{bmatrix}

    where :math:`D_{ij}` are diagonal matrices whose non-zero elements
    are defined by vector :math:`\\mathbf{b_{ij}}`. Where *n* is the
    number of matricies, the block matrix is sparse with dimensions
    (2n, 2n).


    Parameters
    ----------
    aij : numpy.ndarray
        All arguments a11, a12, a21, a22 are vectors which contain the
        corresponding element for all 2x2 matricies
    return_matrix : bool, optional

        - **True:** Returns the sparse block 2x2 matrix *M*.
        - **False:** Returns the vectors containing the elements of each matrix' inverse.


    Returns
    -------
    sparse.coo.coo_matrix or list of numpy.ndarray
        If *return_matrix = False*, the function will return vectors
        *b11, b12, b21, b22*.
        If *return_matrix = True*, the function will return the
        block matrix *M*

    Examples
    --------

    Here, we define four 2x2 matricies and reorganize their elements into 
    4 vectors a11, a12, a21 and a22. We then examine the outputs of the
    function **inverse_2x2_block_diagonal** when the argument
    *return_matrix* is set to both *True* and *False*.

    >>> from discretize.utils import inverse_2x2_block_diagonal
    >>> import numpy as np
    >>> import scipy as sp
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> # Define four 3x3 matricies
    >>> A1 = np.random.uniform(1, 10, (2, 2))
    >>> A2 = np.random.uniform(1, 10, (2, 2))
    >>> A3 = np.random.uniform(1, 10, (2, 2))
    >>> A4 = np.random.uniform(1, 10, (2, 2))
    >>> 
    >>> # Organize elements in vectors
    >>> v = []
    >>> for ii in range(0, 2):
    >>>     for jj in range(0, 2):
    >>>         v.append(
    >>>             np.c_[A1[ii, jj], A2[ii, jj], A3[ii, jj], A4[ii, jj]]
    >>>         )
    >>> 
    >>> a11, a12, a21, a22 = v
    >>> 
    >>> # Return the elements of their inverse and validate
    >>> b11, b12, b21, b22 = inverse_2x2_block_diagonal(
    >>>     a11, a12, a21, a22, return_matrix=False
    >>> )
    >>> 
    >>> B = []
    >>> for ii in range(0, 4):
    >>>     B.append(
    >>>         np.r_[
    >>>             np.c_[b11[ii], b12[ii]],
    >>>             np.c_[b21[ii], b22[ii]]
    >>>         ]
    >>>     )
    >>>     
    >>> B1, B2, B3, B4 = B
    >>> 
    >>> print('Inverse of A1 using numpy.linalg.inv')
    >>> print(np.linalg.inv(A1))
    >>> print('B1 constructed from vectors b11 to b33')
    >>> print(B1)
    >>> 
    >>> # Plot the sparse block matrix containing elements of the inverses
    >>> M = inverse_2x2_block_diagonal(
    >>>     a11, a12, a21, a22
    >>> )
    >>> 
    >>> plt.spy(M)
    """
    if "returnMatrix" in kwargs:
        warnings.warn(
            "The returnMatrix keyword argument has been deprecated, please use return_matrix. "
            "This will be removed in discretize 1.0.0",
            DeprecationWarning,
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


class TensorType(object):
    r"""Class for determining tensor type.

    For a given *mesh*, *tensor* is a numpy.ndarray defining the constitutive
    relationship between two vector quantities for each cell in the mesh.
    The **TensorType** class examines the dimensions of *tensor* to
    determine if it defines a scalar, isotropic, diagonal anisotropic or tensor
    anisotropic constitutive relationship.

    The relationship between a quantity and its response to external
    stimuli (e.g. Ohm's law) can be defined by a scalar quantity:

    .. math::
        \vec{j} = \sigma \vec{e}

    Or in the case of anisotropy, the relationship is defined generally by
    a tensor:
    
    .. math::
        \vec{j} = \Sigma \vec{e} \;\;\; where \;\;\;
        \Sigma = \begin{bmatrix}
        \sigma_{1} & \sigma_{4} & \sigma_{5} \\
        \sigma_{4} & \sigma_{2} & \sigma_{6} \\
        \sigma_{5} & \sigma_{6} & \sigma_{3}
        \end{bmatrix}

    where the tensor is symmetric and defined by 6 independent element
    (3 independent elements in 2D). For a given *mesh*, the parameters
    defined in the numpy.ndarray *tensor* belong to one of the following
    classifications in discretize:

        - **Scalar:** :math:`\vec{j} = \sigma \vec{e}`, where :math:`\sigma` a constant. Thus the input argument *tensor* is a float. 
        
        - **Isotropic:** :math:`\vec{j} = \sigma \vec{e}`, where :math:`\sigma (r)` varies spatially. Thus the input argument *tensor* is a 1D array that provides a :math:`\sigma` value for every cell in the mesh.

        - **Anisotropic:** :math:`\vec{j} = \Sigma \vec{e}`, where the off-diagonal elements are zero. That is, :math:`\Sigma` is diagonal. In this case, the input argument *tensor* defining the physical properties in each cell is a numpy.ndarray of shape (*nCells*, *dim*).
        
        - **Tensor:** :math:`\vec{j} = \Sigma \vec{e}`, where off-diagonal elements are non-zero and :math:`\Sigma` is a full tensor. In this case, the input argument *tensor* defining the physical properties in each cell is a numpy.ndarray of shape (*nCells*, *nParam*). In 2D, *nParam* = 3 and in 3D, *nParam* = 6.

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
        return "TensorType[{0:d}]: {1!s}".format(self._tt, self._tts)

    def __eq__(self, v):
        return self._tt == v

    def __le__(self, v):
        return self._tt <= v

    def __ge__(self, v):
        return self._tt >= v

    def __lt__(self, v):
        return self._tt < v

    def __gt__(self, v):
        return self._tt > v


def make_property_tensor(mesh, tensor):
    r"""Construct the physical property tensor.

    The relationship between a quantity and its response to external
    stimuli (e.g. Ohm's law) in each cell can be defined by a scalar
    function :math:`\sigma` in the isotropic case, or by a tensor
    :math:`\Sigma` in the anisotropic case, i.e.:

    .. math::
        \vec{j} = \sigma \vec{e} \;\;\;\;\;\; \textrm{or} \;\;\;\;\;\; \vec{j} = \Sigma \vec{e}

    where
    
    .. math::
        \Sigma = \begin{bmatrix}
        \sigma_{1} & \sigma_{4} & \sigma_{5} \\
        \sigma_{4} & \sigma_{2} & \sigma_{6} \\
        \sigma_{5} & \sigma_{6} & \sigma_{3}
        \end{bmatrix}

    For a given *mesh*, the input parameter *tensor* is a numpy.ndarray
    defining the constitutive relationship (e.g. Ohm's law) between two
    discrete vector quantities :math:`\boldsymbol{j}` and
    :math:`\boldsymbol{e}` living at cell centers. The function
    **make_property_tensor** constructs the property tensor
    :math:`\boldsymbol{M}` for the entire mesh such that:

    .. math::
        \boldsymbol{j} = \boldsymbol{M \, e}

    where the Cartesian components of the discrete vector for
    are organized according to:

    .. math::
        \begin{bmatrix} \boldsymbol{j_x} \\ \boldsymbol{j_y} \\ \boldsymbol{j_z} \end{bmatrix} \;\;\; \textrm{and} \;\;\;
        \begin{bmatrix} \boldsymbol{e_x} \\ \boldsymbol{e_y} \\ \boldsymbol{e_z} \end{bmatrix}
    

    Parameters
    ----------
    mesh : discretize.BaseMesh
       A mesh
    tensor : numpy.ndarray or a float

        - *Scalar:* A float is entered.
        - *Isotropic:* A 1D numpy.ndarray with a property value for every cell.
        - *Anisotropic:* A numpy.ndarray of shape (*nCell*, *dim*) where each row defines the diagonal-anisotropic property parameters for each cell. *nParam* = 2 for 2D meshes and *nParam* = 3 for 3D meshes.
        - *Tensor:* A numpy.ndarray of shape (*nCell*, *nParam*) where each row defines the full anisotropic property parameters for each cell. *nParam* = 3 for 2D meshes and *nParam* = 6 for 3D meshes.

    Returns
    -------
    sparse.coo.coo_matrix
        The property tensor.
    
    
    Examples
    --------

    For the 4 classifications allowable (scalar, isotropic, anistropic and tensor),
    we construct and compare the property tensor on a small 2D mesh. For this
    example, note the following:

        - The dimensions for all property tensors are the same
        - All property tensors, except in the case of full anisotropy are diagonal sparse matrices
        - For the scalar case, the non-zero elements are equal
        - For the isotropic case, the non-zero elements repreat in order for the x, y (and z) components
        - For the anisotropic case (diagonal anisotropy), the non-zero elements do not repeat
        - For the tensor caes (full anisotropy), there are off-diagonal components

    >>> from discretize.utils import make_property_tensor
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib as mpl
    >>> 
    >>> np.random.seed(31)
    >>> 
    >>> # Define a 2D tensor mesh
    >>> h = [1., 1., 1.]
    >>> mesh = TensorMesh([h, h], origin='00')
    >>> 
    >>> # Define a physical property for all cases (2D)
    >>> sigma_scalar = 4.
    >>> sigma_isotropic = np.random.randint(1, 10, mesh.nC)
    >>> sigma_anisotropic = np.random.randint(1, 10, (mesh.nC, 2))
    >>> sigma_tensor = np.random.randint(1, 10, (mesh.nC, 3))
    >>> 
    >>> # Construct the property tensor in each case
    >>> M_scalar = make_property_tensor(mesh, sigma_scalar)
    >>> M_isotropic = make_property_tensor(mesh, sigma_isotropic)
    >>> M_anisotropic = make_property_tensor(mesh, sigma_anisotropic)
    >>> M_tensor = make_property_tensor(mesh, sigma_tensor)
    >>> 
    >>> M_list = [M_scalar, M_isotropic, M_anisotropic, M_tensor]
    >>> case_list = ['Scalar', 'Isotropic', 'Anisotropic', 'Full Tensor']
    >>> ax1 = 4*[None]
    >>> 
    >>> # Plot the property tensors.
    >>> fig = plt.figure(figsize=(15, 4))
    >>> for ii in range(0, 4):
    >>>     ax1[ii] = fig.add_axes([0.05+0.22*ii, 0.05, 0.18, 0.9])
    >>>     ax1[ii].imshow(
    >>>         M_list[ii].todense(), interpolation='none', cmap='binary', vmax=10.
    >>>     )
    >>>     ax1[ii].set_title(case_list[ii], fontsize=20)
    >>> 
    >>> ax2 = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    >>> norm = mpl.colors.Normalize(vmin=0., vmax=10.)
    >>> cbar = mpl.colorbar.ColorbarBase(
    >>>     ax2, norm=norm, orientation="vertical", cmap=mpl.cm.binary
    >>> )


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

    The relationship between a quantity and its response to external
    stimuli (e.g. Ohm's law) in each cell can be defined by the reciprocal
    of a scalar :math:`\sigma` in the isotropic case, or by the inverse
    of a tensor :math:`\Sigma` in the anisotropic case, i.e.:

    .. math::
        \vec{e} = \sigma^{-1} \vec{j} \;\;\;\;\;\; \textrm{or} \;\;\;\;\;\; \vec{e} = \Sigma^{-1} \vec{j}

    where
    
    .. math::
        \Sigma = \begin{bmatrix}
        \sigma_{1} & \sigma_{4} & \sigma_{5} \\
        \sigma_{4} & \sigma_{2} & \sigma_{6} \\
        \sigma_{5} & \sigma_{6} & \sigma_{3}
        \end{bmatrix}

    For a given *mesh*, the input parameter *tensor* is a numpy.ndarray
    defining the constitutive relationship (e.g. Ohm's law) between two
    discrete vector quantities :math:`\boldsymbol{j}` and
    :math:`\boldsymbol{e}` living at cell centers. Where :math:`\boldsymbol{M}`
    is the physical property tensor, **inverse_property_tensor**
    explicitly constructs the inverse of the physical
    property tensor :math:`\boldsymbol{M^{-1}}` for all cells such that:

    .. math::
        \boldsymbol{e} = \boldsymbol{M^{-1} \, j}
    
    where the Cartesian components of the discrete vectors are
    organized according to:

    .. math::
        \begin{bmatrix} \boldsymbol{j_x} \\ \boldsymbol{j_y} \\ \boldsymbol{j_z} \end{bmatrix} \;\;\; \textrm{and} \;\;\;
        \begin{bmatrix} \boldsymbol{e_x} \\ \boldsymbol{e_y} \\ \boldsymbol{e_z} \end{bmatrix}
    

    Parameters
    ----------
    mesh : discretize.BaseMesh
       A mesh
    tensor : numpy.ndarray or a float

        - *Scalar:* A float is entered.
        - *Isotropic:* A 1D numpy.ndarray with a property value for every cell.
        - *Anisotropic:* A numpy.ndarray of shape (*nCell*, *dim*) where each row defines the diagonal-anisotropic property parameters for each cell. *nParam* = 2 for 2D meshes and *nParam* = 3 for 3D meshes.
        - *Tensor:* A numpy.ndarray of shape (*nCell*, *nParam*) where each row defines the full anisotropic property parameters for each cell. *nParam* = 3 for 2D meshes and *nParam* = 6 for 3D meshes.

    return_matrix : bool (default=False)
        If *True*, the function returns the inverse of the property tensor. If *False*, the function returns the non-zero elements of the inverse of the property tensor in a numpy.ndarray in the same order as the input argument *tensor*.

    Returns
    -------
    numpy.ndarray or sparse.coo.coo_matrix
    
        - If *return_matrix* = *False*, the function outputs the parameters defining the inverse of the property tensor in a numpy.ndarray with the same dimensions as the input argument *tensor*
        - *True:* the function outputs the inverse of the property tensor as a *sparse.coo.coo_matrix*.


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
    >>> 
    >>> np.random.seed(31)
    >>> 
    >>> # Define a 2D tensor mesh
    >>> h = [1., 1., 1.]
    >>> mesh = TensorMesh([h, h], origin='00')
    >>> 
    >>> # Define a physical property for all cases (2D)
    >>> sigma_scalar = 4.
    >>> sigma_isotropic = np.random.randint(1, 10, mesh.nC)
    >>> sigma_anisotropic = np.random.randint(1, 10, (mesh.nC, 2))
    >>> sigma_tensor = np.random.randint(1, 10, (mesh.nC, 3))
    >>> 
    >>> # Construct the property tensor in each case
    >>> M_scalar = make_property_tensor(mesh, sigma_scalar)
    >>> M_isotropic = make_property_tensor(mesh, sigma_isotropic)
    >>> M_anisotropic = make_property_tensor(mesh, sigma_anisotropic)
    >>> M_tensor = make_property_tensor(mesh, sigma_tensor)
    >>> 
    >>> # Construct the inverse property tensor in each case
    >>> Minv_scalar = inverse_property_tensor(mesh, sigma_scalar, return_matrix=True)
    >>> Minv_isotropic = inverse_property_tensor(mesh, sigma_isotropic, return_matrix=True)
    >>> Minv_anisotropic = inverse_property_tensor(mesh, sigma_anisotropic, return_matrix=True)
    >>> Minv_tensor = inverse_property_tensor(mesh, sigma_tensor, return_matrix=True)
    >>> 
    >>> # Lists for plotting
    >>> M_list = [M_scalar, M_isotropic, M_anisotropic, M_tensor]
    >>> Minv_list = [Minv_scalar, Minv_isotropic, Minv_anisotropic, Minv_tensor]
    >>> case_list = ['Scalar', 'Isotropic', 'Anisotropic', 'Full Tensor']
    >>> 
    >>> # Plot the property tensors.
    >>> fig1 = plt.figure(figsize=(15, 4))
    >>> ax1 = 4*[None]
    >>> for ii in range(0, 4):
    >>>     ax1[ii] = fig1.add_axes([0.05+0.22*ii, 0.05, 0.18, 0.9])
    >>>     ax1[ii].imshow(
    >>>         M_list[ii].todense(), interpolation='none', cmap='binary', vmax=10.
    >>>     )
    >>>     ax1[ii].set_title('$M$ (' + case_list[ii] + ')', fontsize=20)
    >>> 
    >>> cax1 = fig1.add_axes([0.92, 0.15, 0.01, 0.7])
    >>> norm1 = mpl.colors.Normalize(vmin=0., vmax=10.)
    >>> cbar1 = mpl.colorbar.ColorbarBase(
    >>>     cax1, norm=norm1, orientation="vertical", cmap=mpl.cm.binary
    >>> )
    >>> 
    >>> # Plot the inverse property tensors.
    >>> fig2 = plt.figure(figsize=(15, 4))
    >>> ax2 = 4*[None]
    >>> for ii in range(0, 4):
    >>>     ax2[ii] = fig2.add_axes([0.05+0.22*ii, 0.05, 0.18, 0.9])
    >>>     ax2[ii].imshow(
    >>>         Minv_list[ii].todense(), interpolation='none', cmap='binary', vmax=1.
    >>>     )
    >>>     ax2[ii].set_title('$M^{-1}$ (' + case_list[ii] + ')', fontsize=20)
    >>> 
    >>> cax2 = fig2.add_axes([0.92, 0.15, 0.01, 0.7])
    >>> norm2 = mpl.colors.Normalize(vmin=0., vmax=1.)
    >>> cbar2 = mpl.colorbar.ColorbarBase(
    >>>     cax2, norm=norm2, orientation="vertical", cmap=mpl.cm.binary
    >>> )

    """

    if "returnMatrix" in kwargs:
        warnings.warn(
            "The returnMatrix keyword argument has been deprecated, please use return_matrix. "
            "This will be removed in discretize 1.0.0",
            DeprecationWarning,
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

    __numpy_ufunc__ = True
    __array_ufunc__ = None

    def __add__(self, v):
        return v

    def __radd__(self, v):
        return v

    def __iadd__(self, v):
        return v

    def __sub__(self, v):
        return -v

    def __rsub__(self, v):
        return v

    def __isub__(self, v):
        return v

    def __mul__(self, v):
        return self

    def __rmul__(self, v):
        return self

    def __matmul__(self, v):
        return self

    def __rmatmul__(self, v):
        return self

    def __div__(self, v):
        return self

    def __truediv__(self, v):
        return self

    def __rdiv__(self, v):
        raise ZeroDivisionError("Cannot divide by zero.")

    def __rtruediv__(self, v):
        raise ZeroDivisionError("Cannot divide by zero.")

    def __rfloordiv__(self, v):
        raise ZeroDivisionError("Cannot divide by zero.")

    def __pos__(self):
        return self

    def __neg__(self):
        return self

    def __lt__(self, v):
        return 0 < v

    def __le__(self, v):
        return 0 <= v

    def __eq__(self, v):
        return v == 0

    def __ne__(self, v):
        return not (0 == v)

    def __ge__(self, v):
        return 0 >= v

    def __gt__(self, v):
        return 0 > v

    def transpose(self):
        return self

    def __getitem__(self, key):
        return self

    @property
    def ndim(self):
        return None

    @property
    def shape(self):
        return _inftup(None)

    @property
    def T(self):
        return self


class Identity(object):

    __numpy_ufunc__ = True
    __array_ufunc__ = None

    _positive = True

    def __init__(self, positive=True):
        self._positive = positive

    def __pos__(self):
        return self

    def __neg__(self):
        return Identity(not self._positive)

    def __add__(self, v):
        if sp.issparse(v):
            return v + speye(v.shape[0]) if self._positive else v - speye(v.shape[0])
        return v + 1 if self._positive else v - 1

    def __radd__(self, v):
        return self.__add__(v)

    def __sub__(self, v):
        return self + -v

    def __rsub__(self, v):
        return -self + v

    def __mul__(self, v):
        return v if self._positive else -v

    def __rmul__(self, v):
        return v if self._positive else -v

    def __matmul__(self, v):
        return v if self._positive else -v

    def __rmatmul__(self, v):
        return v if self._positive else -v

    def __div__(self, v):
        if sp.issparse(v):
            raise NotImplementedError("Sparse arrays not divisibile.")
        return 1 / v if self._positive else -1 / v

    def __truediv__(self, v):
        if sp.issparse(v):
            raise NotImplementedError("Sparse arrays not divisibile.")
        return 1.0 / v if self._positive else -1.0 / v

    def __rdiv__(self, v):
        return v if self._positive else -v

    def __rtruediv__(self, v):
        return v if self._positive else -v

    def __floordiv__(self, v):
        return 1 // v if self._positive else -1 // v

    def __rfloordiv__(self, v):
        return 1 // v if self._positivie else -1 // v

    def __lt__(self, v):
        return 1 < v if self._positive else -1 < v

    def __le__(self, v):
        return 1 <= v if self._positive else -1 <= v

    def __eq__(self, v):
        return v == 1 if self._positive else v == -1

    def __ne__(self, v):
        return (not (1 == v)) if self._positive else (not (-1 == v))

    def __ge__(self, v):
        return 1 >= v if self._positive else -1 >= v

    def __gt__(self, v):
        return 1 > v if self._positive else -1 > v

    @property
    def ndim(self):
        return None

    @property
    def shape(self):
        return _inftup(None)

    @property
    def T(self):
        return self

    def transpose(self):
        return self


class _inftup(tuple):
    """An infinitely long tuple of a value repeated infinitely"""

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


sdInv = deprecate_function(sdinv, "sdInv", removal_version="1.0.0")
getSubArray = deprecate_function(get_subarray, "getSubArray", removal_version="1.0.0")
inv3X3BlockDiagonal = deprecate_function(
    inverse_3x3_block_diagonal, "inv3X3BlockDiagonal", removal_version="1.0.0"
)
inv2X2BlockDiagonal = deprecate_function(
    inverse_2x2_block_diagonal, "inv2X2BlockDiagonal", removal_version="1.0.0"
)
makePropertyTensor = deprecate_function(
    make_property_tensor, "makePropertyTensor", removal_version="1.0.0"
)
invPropertyTensor = deprecate_function(
    inverse_property_tensor, "invPropertyTensor", removal_version="1.0.0"
)
