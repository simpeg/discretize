import numpy as np
import scipy.sparse as sp
from discretize.utils.code_utils import is_scalar, deprecate_function
import warnings


def mkvc(x, n_dims=1, **kwargs):
    """Creates a vector with the number of dimension specified

    e.g.::

        a = np.array([1, 2, 3])

        mkvc(a, 1).shape
            > (3, )

        mkvc(a, 2).shape
            > (3, 1)

        mkvc(a, 3).shape
            > (3, 1, 1)

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


def sdiag(h):
    """Sparse diagonal matrix"""
    if isinstance(h, Zero):
        return Zero()

    return sp.spdiags(mkvc(h), 0, h.size, h.size, format="csr")


def sdinv(M):
    "Inverse of a sparse diagonal matrix"
    return sdiag(1.0 / M.diagonal())


def speye(n):
    """Sparse identity"""
    return sp.identity(n, format="csr")


def kron3(A, B, C):
    """Three kron prods"""
    return sp.kron(sp.kron(A, B), C, format="csr")


def spzeros(n1, n2):
    """a sparse matrix of zeros"""
    return sp.dia_matrix((n1, n2))


def ddx(n):
    """Define 1D derivatives, inner, this means we go from n+1 to n"""
    return sp.spdiags((np.ones((n + 1, 1)) * [-1, 1]).T, [0, 1], n, n + 1, format="csr")


def av(n):
    """Define 1D averaging operator from nodes to cell-centers."""
    return sp.spdiags(
        (0.5 * np.ones((n + 1, 1)) * [1, 1]).T, [0, 1], n, n + 1, format="csr"
    )


def av_extrap(n):
    """Define 1D averaging operator from cell-centers to nodes."""
    Av = sp.spdiags(
        (0.5 * np.ones((n, 1)) * [1, 1]).T, [-1, 0], n + 1, n, format="csr"
    ) + sp.csr_matrix(([0.5, 0.5], ([0, n], [0, n - 1])), shape=(n + 1, n))
    return Av


def ndgrid(*args, **kwargs):
    """
    Form tensorial grid for 1, 2, or 3 dimensions.

    Returns as column vectors by default.

    To return as matrix input:

        ndgrid(..., vector=False)

    The inputs can be a list or separate arguments.

    e.g.::

        a = np.array([1, 2, 3])
        b = np.array([1, 2])

        XY = ndgrid(a, b)
            > [[1 1]
               [2 1]
               [3 1]
               [1 2]
               [2 2]
               [3 2]]

        X, Y = ndgrid(a, b, vector=False)
            > X = [[1 1]
                   [2 2]
                   [3 3]]
            > Y = [[1 2]
                   [1 2]
                   [1 2]]

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


def ind2sub(shape, inds):
    """From the given shape, returns the subscripts of the given index"""
    if type(inds) is not np.ndarray:
        inds = np.array(inds)
    if len(inds.shape) != 1:
        raise ValueError("Indexing must be done as a 1D row vector, e.g. [3,6,6,...]")
    return np.unravel_index(inds, shape, order="F")


def sub2ind(shape, subs):
    """From the given shape, returns the index of the given subscript"""
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
    """B = inverse_3x3_block_diagonal(a11, a12, a13, a21, a22, a23, a31, a32, a33)

    inverts a stack of 3x3 matrices

    Input:
     A   - a11, a12, a13, a21, a22, a23, a31, a32, a33

    Output:
     B   - inverse
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
    """B = inverse_2x2_block_diagonal(a11, a12, a21, a22)

    Inverts a stack of 2x2 matrices by using the inversion formula

    inv(A) = (1/det(A)) * cof(A)^T

    Input:
    A   - a11, a12, a21, a22

    Output:
    B   - inverse
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


class TensorType(object):
    def __init__(self, M, tensor):
        if tensor is None:  # default is ones
            self._tt = -1
            self._tts = "none"
        elif is_scalar(tensor):
            self._tt = 0
            self._tts = "scalar"
        elif tensor.size == M.nC:
            self._tt = 1
            self._tts = "isotropic"
        elif (M.dim == 2 and tensor.size == M.nC * 2) or (
            M.dim == 3 and tensor.size == M.nC * 3
        ):
            self._tt = 2
            self._tts = "anisotropic"
        elif (M.dim == 2 and tensor.size == M.nC * 3) or (
            M.dim == 3 and tensor.size == M.nC * 6
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


def make_property_tensor(M, tensor):
    if tensor is None:  # default is ones
        tensor = np.ones(M.nC)

    if is_scalar(tensor):
        tensor = tensor * np.ones(M.nC)

    propType = TensorType(M, tensor)
    if propType == 1:  # Isotropic!
        Sigma = sp.kron(sp.identity(M.dim), sdiag(mkvc(tensor)))
    elif propType == 2:  # Diagonal tensor
        Sigma = sdiag(mkvc(tensor))
    elif M.dim == 2 and tensor.size == M.nC * 3:  # Fully anisotropic, 2D
        tensor = tensor.reshape((M.nC, 3), order="F")
        row1 = sp.hstack((sdiag(tensor[:, 0]), sdiag(tensor[:, 2])))
        row2 = sp.hstack((sdiag(tensor[:, 2]), sdiag(tensor[:, 1])))
        Sigma = sp.vstack((row1, row2))
    elif M.dim == 3 and tensor.size == M.nC * 6:  # Fully anisotropic, 3D
        tensor = tensor.reshape((M.nC, 6), order="F")
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


def inverse_property_tensor(M, tensor, return_matrix=False, **kwargs):
    if "returnMatrix" in kwargs:
        warnings.warn(
            "The returnMatrix keyword argument has been deprecated, please use return_matrix. "
            "This will be removed in discretize 1.0.0",
            FutureWarning,
        )
        return_matrix = kwargs["returnMatrix"]

    propType = TensorType(M, tensor)

    if is_scalar(tensor):
        T = 1.0 / tensor
    elif propType < 3:  # Isotropic or Diagonal
        T = 1.0 / mkvc(tensor)  # ensure it is a vector.
    elif M.dim == 2 and tensor.size == M.nC * 3:  # Fully anisotropic, 2D
        tensor = tensor.reshape((M.nC, 3), order="F")
        B = inverse_2x2_block_diagonal(
            tensor[:, 0], tensor[:, 2], tensor[:, 2], tensor[:, 1], return_matrix=False
        )
        b11, b12, b21, b22 = B
        T = np.r_[b11, b22, b12]
    elif M.dim == 3 and tensor.size == M.nC * 6:  # Fully anisotropic, 3D
        tensor = tensor.reshape((M.nC, 6), order="F")
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
        return make_property_tensor(M, T)

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

    def  __getitem__(self, key):
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
