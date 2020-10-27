import numpy as np
from discretize.utils.matrix_utils import mkvc
from discretize.utils.code_utils import deprecate_function


def cylindrical_to_cartesian(grid, vec=None):
    """
    Take a grid defined in cylindrical coordinates :math:`(r, \theta, z)` and
    transform it to cartesian coordinates.
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
    """An alias for cylindrical_to_cartesian"""
    return cylindrical_to_cartesian(grid, vec)


def cartesian_to_cylindrical(grid, vec=None):
    """
    Take a grid defined in cartesian coordinates and transform it to cyl
    coordinates
    """
    if vec is None:
        vec = grid

    vec = np.atleast_2d(vec)
    grid = np.atleast_2d(grid)

    theta = np.arctan2(grid[:, 1], grid[:, 0])

    return np.hstack(
        [
            mkvc(np.cos(theta) * vec[:, 0] + np.sin(theta) * vec[:, 1], 2),
            mkvc(-np.sin(theta) * vec[:, 0] + np.cos(theta) * vec[:, 1], 2),
            mkvc(vec[:, 2], 2),
        ]
    )


def cart2cyl(grid, vec=None):
    """An alias for cartesian_to_cylindrical"""
    return cylindrical_to_cartesian(grid, vec)


def rotation_matrix_from_normals(v0, v1, tol=1e-20):
    """
    Performs the minimum number of rotations to define a rotation from the
    direction indicated by the vector n0 to the direction indicated by n1.
    The axis of rotation is n0 x n1
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    :param numpy.array v0: vector of length 3
    :param numpy.array v1: vector of length 3
    :param tol = 1e-20: tolerance. If the norm of the cross product between the two vectors is below this, no rotation is performed
    :rtype: numpy.array, 3x3
    :return: rotation matrix which rotates the frame so that n0 is aligned with n1
    """

    # ensure both n0, n1 are vectors of length 1
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
    sinT = np.sqrt(1.0 - n0dotn1 ** 2)

    ux = np.array(
        [
            [0.0, -rotAx[2], rotAx[1]],
            [rotAx[2], 0.0, -rotAx[0]],
            [-rotAx[1], rotAx[0], 0.0],
        ],
        dtype=float,
    )

    return np.eye(3, dtype=float) + sinT * ux + (1.0 - cosT) * (ux.dot(ux))


def rotate_points_from_normals(XYZ, n0, n1, x0=np.r_[0.0, 0.0, 0.0]):
    """
    rotates a grid so that the vector n0 is aligned with the vector n1

    :param numpy.array n0: vector of length 3, should have norm 1
    :param numpy.array n1: vector of length 3, should have norm 1
    :param numpy.array x0: vector of length 3, point about which we perform the rotation
    :rtype: numpy.array, 3x3
    :return: rotation matrix which rotates the frame so that n0 is aligned with n1
    """

    R = rotation_matrix_from_normals(n0, n1)

    if XYZ.shape[1] != 3:
        raise ValueError("Grid XYZ should be 3 wide")
    if len(x0) != 3:
        raise ValueError("x0 should have length 3")

    X0 = np.ones([XYZ.shape[0], 1]) * mkvc(x0)

    return (XYZ - X0).dot(R.T) + X0  # equivalent to (R*(XYZ - X0)).T + X0


rotationMatrixFromNormals = deprecate_function(
    rotation_matrix_from_normals, "rotationMatrixFromNormals", removal_version="1.0.0"
)
rotatePointsFromNormals = deprecate_function(
    rotate_points_from_normals, "rotatePointsFromNormals", removal_version="1.0.0"
)
