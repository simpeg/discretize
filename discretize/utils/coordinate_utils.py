"""Simple utilities for coordinate transformations."""
import numpy as np
from discretize.utils.matrix_utils import mkvc
from discretize.utils.code_utils import as_array_n_by_dim, deprecate_function


def cylindrical_to_cartesian(grid, vec=None):
    r"""Transform from cylindrical to cartesian coordinates.

    Transform a grid or a vector from cylindrical coordinates :math:`(r, \theta, z)` to
    Cartesian coordinates :math:`(x, y, z)`. :math:`\theta` is given in radians.

    Parameters
    ----------
    grid : (n, 3) array_like
        Location points defined in cylindrical coordinates :math:`(r, \theta, z)`.
    vec : (n, 3) array_like, optional
        Vector defined in cylindrical coordinates :math:`(r, \theta, z)` at the
        locations grid. Will also except a flattend array in column major order with the
        same number of elements.

    Returns
    -------
    (n, 3) numpy.ndarray
        If `vec` is ``None``, this returns the transformed `grid` array, otherwise
        this is the transformed `vec` array.

    Examples
    --------
    Here, we convert a series of vectors in 3D space from cylindrical coordinates
    to Cartesian coordinates.

    >>> from discretize.utils import cylindrical_to_cartesian
    >>> import numpy as np

    Construct original set of vectors in cylindrical coordinates

    >>> r = np.ones(9)
    >>> phi = np.linspace(0, 2*np.pi, 9)
    >>> z = np.linspace(-4., 4., 9)
    >>> u = np.c_[r, phi, z]
    >>> u
    array([[ 1.        ,  0.        , -4.        ],
           [ 1.        ,  0.78539816, -3.        ],
           [ 1.        ,  1.57079633, -2.        ],
           [ 1.        ,  2.35619449, -1.        ],
           [ 1.        ,  3.14159265,  0.        ],
           [ 1.        ,  3.92699082,  1.        ],
           [ 1.        ,  4.71238898,  2.        ],
           [ 1.        ,  5.49778714,  3.        ],
           [ 1.        ,  6.28318531,  4.        ]])

    Create equivalent set of vectors in Cartesian coordinates

    >>> v = cylindrical_to_cartesian(u)
    >>> v
    array([[ 1.00000000e+00,  0.00000000e+00, -4.00000000e+00],
           [ 7.07106781e-01,  7.07106781e-01, -3.00000000e+00],
           [ 6.12323400e-17,  1.00000000e+00, -2.00000000e+00],
           [-7.07106781e-01,  7.07106781e-01, -1.00000000e+00],
           [-1.00000000e+00,  1.22464680e-16,  0.00000000e+00],
           [-7.07106781e-01, -7.07106781e-01,  1.00000000e+00],
           [-1.83697020e-16, -1.00000000e+00,  2.00000000e+00],
           [ 7.07106781e-01, -7.07106781e-01,  3.00000000e+00],
           [ 1.00000000e+00, -2.44929360e-16,  4.00000000e+00]])
    """
    grid = np.atleast_2d(grid)

    if vec is None:
        return np.hstack(
            [
                mkvc(grid[:, 0] * np.cos(grid[:, 1]), 2),
                mkvc(grid[:, 0] * np.sin(grid[:, 1]), 2),
                mkvc(grid[:, 2], 2),
            ]
        )
    vec = np.asanyarray(vec)
    if len(vec.shape) == 1 or vec.shape[1] == 1:
        vec = vec.reshape(grid.shape, order="F")

    x = vec[:, 0] * np.cos(grid[:, 1]) - vec[:, 1] * np.sin(grid[:, 1])
    y = vec[:, 0] * np.sin(grid[:, 1]) + vec[:, 1] * np.cos(grid[:, 1])

    newvec = [x, y]
    if grid.shape[1] == 3:
        z = vec[:, 2]
        newvec += [z]

    return np.vstack(newvec).T


def cyl2cart(grid, vec=None):
    """Transform from cylindrical to cartesian coordinates.

    An alias for `cylindrical_to_cartesian``.

    See Also
    --------
    cylindrical_to_cartesian
    """
    return cylindrical_to_cartesian(grid, vec)


def cartesian_to_cylindrical(grid, vec=None):
    r"""Transform from cartesian to cylindrical coordinates.

    Transform a grid or a vector from Cartesian coordinates :math:`(x, y, z)` to
    cylindrical coordinates :math:`(r, \theta, z)`.

    Parameters
    ----------
    grid : (n, 3) array_like
        Location points defined in Cartesian coordinates :math:`(x, y z)`.
    vec : (n, 3) array_like, optional
        Vector defined in Cartesian coordinates. This also accepts a flattened array
        with the same total elements in column major order.

    Returns
    -------
    (n, 3) numpy.ndarray
        If `vec` is ``None``, this returns the transformed `grid` array, otherwise
        this is the transformed `vec` array.

    Examples
    --------
    Here, we convert a series of vectors in 3D space from Cartesian coordinates
    to cylindrical coordinates.

    >>> from discretize.utils import cartesian_to_cylindrical
    >>> import numpy as np

    Create set of vectors in Cartesian coordinates

    >>> r = np.ones(9)
    >>> phi = np.linspace(0, 2*np.pi, 9)
    >>> z = np.linspace(-4., 4., 9)
    >>> x = r*np.cos(phi)
    >>> y = r*np.sin(phi)
    >>> u = np.c_[x, y, z]
    >>> u
    array([[ 1.00000000e+00,  0.00000000e+00, -4.00000000e+00],
           [ 7.07106781e-01,  7.07106781e-01, -3.00000000e+00],
           [ 6.12323400e-17,  1.00000000e+00, -2.00000000e+00],
           [-7.07106781e-01,  7.07106781e-01, -1.00000000e+00],
           [-1.00000000e+00,  1.22464680e-16,  0.00000000e+00],
           [-7.07106781e-01, -7.07106781e-01,  1.00000000e+00],
           [-1.83697020e-16, -1.00000000e+00,  2.00000000e+00],
           [ 7.07106781e-01, -7.07106781e-01,  3.00000000e+00],
           [ 1.00000000e+00, -2.44929360e-16,  4.00000000e+00]])

    Compute equivalent set of vectors in cylindrical coordinates

    >>> v = cartesian_to_cylindrical(u)
    >>> v
    array([[ 1.00000000e+00,  0.00000000e+00, -4.00000000e+00],
           [ 1.00000000e+00,  7.85398163e-01, -3.00000000e+00],
           [ 1.00000000e+00,  1.57079633e+00, -2.00000000e+00],
           [ 1.00000000e+00,  2.35619449e+00, -1.00000000e+00],
           [ 1.00000000e+00,  3.14159265e+00,  0.00000000e+00],
           [ 1.00000000e+00, -2.35619449e+00,  1.00000000e+00],
           [ 1.00000000e+00, -1.57079633e+00,  2.00000000e+00],
           [ 1.00000000e+00, -7.85398163e-01,  3.00000000e+00],
           [ 1.00000000e+00, -2.44929360e-16,  4.00000000e+00]])
    """
    grid = as_array_n_by_dim(grid, 3)
    theta = np.arctan2(grid[:, 1], grid[:, 0])
    if vec is None:
        return np.c_[np.linalg.norm(grid[:, :2], axis=-1), theta, grid[:, 2]]
    vec = as_array_n_by_dim(vec, 3)

    return np.hstack(
        [
            mkvc(np.cos(theta) * vec[:, 0] + np.sin(theta) * vec[:, 1], 2),
            mkvc(-np.sin(theta) * vec[:, 0] + np.cos(theta) * vec[:, 1], 2),
            mkvc(vec[:, 2], 2),
        ]
    )


def cart2cyl(grid, vec=None):
    """Transform from cartesian to cylindrical coordinates.

    An alias for cartesian_to_cylindrical

    See Also
    --------
    cartesian_to_cylindrical
    """
    return cartesian_to_cylindrical(grid, vec)


def rotation_matrix_from_normals(v0, v1, tol=1e-20):
    r"""Generate a 3x3 rotation matrix defining the rotation from vector v0 to v1.

    This function uses Rodrigues' rotation formula to generate the rotation
    matrix :math:`\mathbf{A}` going from vector :math:`\mathbf{v_0}` to
    vector :math:`\mathbf{v_1}`. Thus:

    .. math::
        \mathbf{Av_0} = \mathbf{v_1}

    For detailed desciption of the algorithm, see
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Parameters
    ----------
    v0 : (3) numpy.ndarray
        Starting orientation direction
    v1 : (3) numpy.ndarray
        Finishing orientation direction
    tol : float, optional
        Numerical tolerance. If the length of the rotation axis is below this value,
        it is assumed to be no rotation, and an identity matrix is returned.

    Returns
    -------
    (3, 3) numpy.ndarray
        The rotation matrix from v0 to v1.
    """
    # ensure both v0, v1 are vectors of length 1
    if len(v0) != 3:
        raise ValueError("Length of n0 should be 3")
    if len(v1) != 3:
        raise ValueError("Length of n1 should be 3")

    # ensure both are true normals
    n0 = v0 * 1.0 / np.linalg.norm(v0)
    n1 = v1 * 1.0 / np.linalg.norm(v1)

    n0dotn1 = n0.dot(n1)

    # define the rotation axis, which is the cross product of the two vectors
    rotAx = np.cross(n0, n1)

    if np.linalg.norm(rotAx) < tol:
        return np.eye(3, dtype=float)

    rotAx *= 1.0 / np.linalg.norm(rotAx)

    cosT = n0dotn1 / (np.linalg.norm(n0) * np.linalg.norm(n1))
    sinT = np.sqrt(1.0 - n0dotn1**2)

    ux = np.array(
        [
            [0.0, -rotAx[2], rotAx[1]],
            [rotAx[2], 0.0, -rotAx[0]],
            [-rotAx[1], rotAx[0], 0.0],
        ],
        dtype=float,
    )

    return np.eye(3, dtype=float) + sinT * ux + (1.0 - cosT) * (ux.dot(ux))


def rotate_points_from_normals(xyz, v0, v1, x0=np.r_[0.0, 0.0, 0.0]):
    r"""Rotate a set of xyz locations about a specified point.

    Rotate a grid of Cartesian points about a location x0 according to the
    rotation defined from vector v0 to v1.

    Let :math:`\mathbf{x}` represent an input xyz location, let :math:`\mathbf{x_0}` be
    the origin of rotation, and let :math:`\mathbf{R}` denote the rotation matrix from
    vector v0 to v1. Where :math:`\mathbf{x'}` is the new xyz location, this function
    outputs the following operation for all input locations:

    .. math::
        \mathbf{x'} = \mathbf{R (x - x_0)} + \mathbf{x_0}

    Parameters
    ----------
    xyz : (n, 3) numpy.ndarray
        locations to rotate
    v0 : (3) numpy.ndarray
        Starting orientation direction
    v1 : (3) numpy.ndarray
        Finishing orientation direction
    x0 : (3) numpy.ndarray, optional
        The origin of rotation.

    Returns
    -------
    (n, 3) numpy.ndarray
        The rotated xyz locations.
    """
    # Compute rotation matrix between v0 and v1
    R = rotation_matrix_from_normals(v0, v1)

    if xyz.shape[1] != 3:
        raise ValueError("Grid of xyz points should be n x 3")
    if len(x0) != 3:
        raise ValueError("x0 should have length 3")

    # Define origin
    X0 = np.ones([xyz.shape[0], 1]) * mkvc(x0)

    return (xyz - X0).dot(R.T) + X0  # equivalent to (R*(xyz - X0)).T + X0


rotationMatrixFromNormals = deprecate_function(
    rotation_matrix_from_normals,
    "rotationMatrixFromNormals",
    removal_version="1.0.0",
    future_warn=True,
)
rotatePointsFromNormals = deprecate_function(
    rotate_points_from_normals,
    "rotatePointsFromNormals",
    removal_version="1.0.0",
    future_warn=True,
)
