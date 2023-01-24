"""Functions for working with curvilinear meshes."""
import numpy as np
from discretize.utils.matrix_utils import mkvc, ndgrid, sub2ind
from discretize.utils.code_utils import deprecate_function
import warnings


def example_curvilinear_grid(nC, exType):
    """Create the gridded node locations for a curvilinear mesh.

    Parameters
    ----------
    nC : list of int
        list of number of cells in each dimension. Must be length 2 or 3
    exType : {"rect", "rotate", "sphere"}
        String specifying the style of example curvilinear mesh.

    Returns
    -------
    list of numpy.ndarray
        List containing the gridded x, y (and z) node locations for the
        curvilinear mesh.
    """
    if not isinstance(nC, list):
        raise TypeError("nC must be a list containing the number of nodes")
    if len(nC) != 2 and len(nC) != 3:
        raise ValueError("nC must either two or three dimensions")
    exType = exType.lower()

    possibleTypes = ["rect", "rotate", "sphere"]
    if exType not in possibleTypes:
        raise TypeError("Not a possible example type.")

    if exType == "rect":
        return list(
            ndgrid([np.cumsum(np.r_[0, np.ones(nx) / nx]) for nx in nC], vector=False)
        )
    elif exType == "sphere":
        nodes = list(
            ndgrid(
                [np.cumsum(np.r_[0, np.ones(nx) / nx]) - 0.5 for nx in nC], vector=False
            )
        )
        nodes = np.stack(nodes, axis=-1)
        nodes = 2 * nodes
        # L_inf distance to center
        r0 = np.linalg.norm(nodes, ord=np.inf, axis=-1)
        # L2 distance to center
        r2 = np.linalg.norm(nodes, axis=-1)
        r0[r0 == 0.0] = 1.0
        r2[r2 == 0.0] = 1.0
        scale = r0 / r2
        nodes = nodes * scale[..., None]
        nodes = np.transpose(nodes, (-1, *np.arange(len(nC))))
        nodes = [node for node in nodes]  # turn it into a list
        return nodes
    elif exType == "rotate":
        if len(nC) == 2:
            X, Y = ndgrid(
                [np.cumsum(np.r_[0, np.ones(nx) / nx]) for nx in nC], vector=False
            )
            amt = 0.5 - np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
            amt[amt < 0] = 0
            return [X + (-(Y - 0.5)) * amt, Y + (+(X - 0.5)) * amt]
        elif len(nC) == 3:
            X, Y, Z = ndgrid(
                [np.cumsum(np.r_[0, np.ones(nx) / nx]) for nx in nC], vector=False
            )
            amt = 0.5 - np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)
            amt[amt < 0] = 0
            return [
                X + (-(Y - 0.5)) * amt,
                Y + (-(Z - 0.5)) * amt,
                Z + (-(X - 0.5)) * amt,
            ]


def volume_tetrahedron(xyz, A, B, C, D):
    r"""Return the tetrahedron volumes for a specified set of verticies.

    Let *xyz* be an (n, 3) array denoting a set of vertex locations.
    Any 4 vertex locations *a, b, c* and *d* can be used to define a tetrahedron.
    For the set of tetrahedra whose verticies are indexed in vectors
    *A, B, C* and *D*, this function returns the corresponding volumes.
    See algorithm: https://en.wikipedia.org/wiki/Tetrahedron#Volume

    .. math::
       vol = {1 \over 6} \big | ( \mathbf{a - d} ) \cdot
       ( ( \mathbf{b - d} ) \times ( \mathbf{c - d} ) ) \big |

    Parameters
    ----------
    xyz : (n_pts, 3) numpy.ndarray
        x,y, and z locations for all verticies
    A : (n_tetra) numpy.ndarray of int
        Vector containing the indicies for the **a** vertex locations
    B : (n_tetra) numpy.ndarray of int
        Vector containing the indicies for the **b** vertex locations
    C : (n_tetra) numpy.ndarray of int
        Vector containing the indicies for the **c** vertex locations
    D : (n_tetra) numpy.ndarray of int
        Vector containing the indicies for the **d** vertex locations

    Returns
    -------
    (n_tetra) numpy.ndarray
        Volumes of the tetrahedra whose vertices are indexed by
        *A, B, C* and *D*.

    Examples
    --------
    Here we define a small 3D tensor mesh. 4 nodes are chosen to
    be the verticies of a tetrahedron. We compute the volume of this
    tetrahedron. Note that xyz locations for the verticies can be
    scattered and do not require regular spacing.

    >>> from discretize.utils import volume_tetrahedron
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib as mpl
    >>> mpl.rcParams.update({"font.size": 14})

    Define corners of a uniform cube

    >>> h = [1, 1]
    >>> mesh = TensorMesh([h, h, h])
    >>> xyz = mesh.nodes

    Specify the indicies of the corner points

    >>> A = np.array([0])
    >>> B = np.array([6])
    >>> C = np.array([8])
    >>> D = np.array([24])

    Compute volume for all tetrahedra and the extract first one

    >>> vol = volume_tetrahedron(xyz, A, B, C, D)
    >>> vol = vol[0]
    >>> vol
    array([1.33333333])

    Plotting small mesh and tetrahedron

    .. collapse:: Expand to show scripting for plot

        >>> fig = plt.figure(figsize=(7, 7))
        >>> ax = fig.gca(projection='3d')
        >>> mesh.plot_grid(ax=ax)
        >>> k = [0, 6, 8, 0, 24, 6, 24, 8]
        >>> xyz_tetra = xyz[k, :]
        >>> ax.plot(xyz_tetra[:, 0], xyz_tetra[:, 1], xyz_tetra[:, 2], 'r')
        >>> ax.text(-0.25, 0., 3., 'Volume of the tetrahedron: {:g} $m^3$'.format(vol))
        >>> plt.show()
    """
    AD = xyz[A, :] - xyz[D, :]
    BD = xyz[B, :] - xyz[D, :]
    CD = xyz[C, :] - xyz[D, :]

    V = (
        (BD[:, 0] * CD[:, 1] - BD[:, 1] * CD[:, 0]) * AD[:, 2]
        - (BD[:, 0] * CD[:, 2] - BD[:, 2] * CD[:, 0]) * AD[:, 1]
        + (BD[:, 1] * CD[:, 2] - BD[:, 2] * CD[:, 1]) * AD[:, 0]
    )
    return np.abs(V / 6)


def index_cube(nodes, grid_shape, n=None):
    """Return the index of nodes on a tensor (or curvilinear) mesh.

    For 2D tensor meshes, each cell is defined by nodes
    *A, B, C* and *D*. And for 3D tensor meshes, each cell
    is defined by nodes *A* through *H* (see below). *index_cube*
    outputs the indices for the specified node(s) for all
    cells in the mesh.

    TWO DIMENSIONS::

      node(i,j+1)      node(i+i,j+1)
           B -------------- C
           |                |
           |    cell(i,j)   |
           |        I       |
           |                |
           A -------------- D
       node(i,j)        node(i+1,j)

    THREE DIMENSIONS::

        node(i,j+1,k+1)    node(i+1,j+1,k+1)
                F ---------------- G
               /|                / |
              / |               /  |
             /  |              /   |
     node(i,j,k+1)     node(i+1,j,k+1)
           E --------------- H     |
           |    B -----------|---- C
           |   / cell(i,j,k) |   /
           |  /        I     |  /
           | /               | /
           A --------------- D
      node(i,j,k)     node(i+1,j,k)

    Parameters
    ----------
    nodes : str
        String specifying which nodes to return. For 2D meshes,
        *nodes* must be a string containing combinations of the characters 'A', 'B',
        'C', or 'D'. For 3D meshes, *nodes* can also be 'E', 'F', 'G', or 'H'. Note that
        order is preserved. E.g. if we want to return the C, D and A node indices in
        that particular order, we input *nodes* = 'CDA'.
    grid_shape : list of int
        Number of nodes along the i,j,k directions; e.g. [ni,nj,nk]
    nc : list of int
        Number of cells along the i,j,k directions; e.g. [nci,ncj,nck]

    Returns
    -------
    index : tuple of numpy.ndarray
        Each entry of the tuple is a 1D :class:`numpy.ndarray` containing the indices of
        the nodes specified in the input *nodes* in the order asked;
        e.g. if *nodes* = 'DCBA', the tuple returned is ordered (D,C,B,A).

    Examples
    --------
    Here, we construct a small 2D tensor mesh
    (works for a curvilinear mesh as well) and use *index_cube*
    to find the indices of the 'A' and 'C' nodes. We then
    plot the mesh, as well as the 'A' and 'C' node locations.

    >>> from discretize import TensorMesh
    >>> from discretize.utils import index_cube
    >>> from matplotlib import pyplot as plt
    >>> import numpy as np

    Create a simple tensor mesh.

    >>> n_cells = 5
    >>> h = 2*np.ones(n_cells)
    >>> mesh = TensorMesh([h, h], x0='00')

    Get indices of 'A' and 'C' nodes for all cells.

    >>> A, C = index_cube('AC', [n_cells+1, n_cells+1])

    Plot mesh and the locations of the A and C nodes

    .. collapse:: Expand to show scripting for plot

        >>> fig1 = plt.figure(figsize=(5, 5))
        >>> ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
        >>> mesh.plot_grid(ax=ax1)
        >>> ax1.scatter(mesh.nodes[A, 0], mesh.nodes[A, 1], 100, 'r', marker='^')
        >>> ax1.scatter(mesh.nodes[C, 0], mesh.nodes[C, 1], 100, 'g', marker='v')
        >>> ax1.set_title('A nodes (red) and C nodes (green)')
        >>> plt.show()
    """
    if not isinstance(nodes, str):
        raise TypeError("Nodes must be a str variable: e.g. 'ABCD'")
    nodes = nodes.upper()
    try:
        dim = len(grid_shape)
        if n is None:
            n = tuple(x - 1 for x in grid_shape)
    except TypeError:
        return TypeError("grid_shape must be iterable")
    # Make sure that we choose from the possible nodes.
    possibleNodes = "ABCD" if dim == 2 else "ABCDEFGH"
    for node in nodes:
        if node not in possibleNodes:
            raise ValueError("Nodes must be chosen from: '{0!s}'".format(possibleNodes))

    if dim == 2:
        ij = ndgrid(np.arange(n[0]), np.arange(n[1]))
        i, j = ij[:, 0], ij[:, 1]
    elif dim == 3:
        ijk = ndgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]))
        i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
    else:
        raise Exception("Only 2 and 3 dimensions supported.")

    nodeMap = {
        "A": [0, 0, 0],
        "B": [0, 1, 0],
        "C": [1, 1, 0],
        "D": [1, 0, 0],
        "E": [0, 0, 1],
        "F": [0, 1, 1],
        "G": [1, 1, 1],
        "H": [1, 0, 1],
    }
    out = ()
    for node in nodes:
        shift = nodeMap[node]
        if dim == 2:
            out += (sub2ind(grid_shape, np.c_[i + shift[0], j + shift[1]]).flatten(),)
        elif dim == 3:
            out += (
                sub2ind(
                    grid_shape, np.c_[i + shift[0], j + shift[1], k + shift[2]]
                ).flatten(),
            )

    return out


def face_info(xyz, A, B, C, D, average=True, normalize_normals=True, **kwargs):
    r"""Return normal surface vectors and areas for a given set of faces.

    Let *xyz* be an (n, 3) array denoting a set of vertex locations.
    Now let vertex locations *a, b, c* and *d* define a quadrilateral
    (regular or irregular) in 2D or 3D space. For this quadrilateral,
    we organize the vertices as follows:

    CELL VERTICES::

            a -------Vab------- b
           /                   /
          /                   /
        Vda       (X)       Vbc
        /                   /
       /                   /
      d -------Vcd------- c

    where the normal vector *(X)* is pointing into the page. For a set
    of quadrilaterals whose vertices are indexed in arrays *A, B, C* and *D* ,
    this function returns the normal surface vector(s) and the area
    for each quadrilateral.

    At each vertex, there are 4 cross-products that can be used to compute the
    vector normal the surface defined by the quadrilateral. In 3D space however,
    the vertices indexed may not define a quadrilateral exactly and thus the normal vectors
    computed at each vertex might not be identical. In this case, you may choose output
    the normal vector at *a, b, c* and *d* or compute
    the average normal surface vector as follows:

    .. math::
        \bf{n} = \frac{1}{4} \big (
        \bf{v_{ab} \times v_{da}} +
        \bf{v_{bc} \times v_{ab}} +
        \bf{v_{cd} \times v_{bc}} +
        \bf{v_{da} \times v_{cd}} \big )


    For computing the surface area, we assume the vertices define a quadrilateral.

    Parameters
    ----------
    xyz : (n, 3) numpy.ndarray
        The x,y, and z locations for all verticies
    A : (n_face) numpy.ndarray
        Vector containing the indicies for the **a** vertex locations
    B : (n_face) numpy.ndarray
        Vector containing the indicies for the **b** vertex locations
    C : (n_face) numpy.ndarray
        Vector containing the indicies for the **c** vertex locations
    D : (n_face) numpy.ndarray
        Vector containing the indicies for the **d** vertex locations
    average : bool, optional
        If *True*, the function returns the average surface
        normal vector for each surface. If *False* , the function will
        return the normal vectors computed at the *A, B, C* and *D*
        vertices in a cell array {nA,nB,nC,nD}.
    normalize_normal : bool, optional
        If *True*, the function will normalize the surface normal
        vectors. This is applied regardless of whether the *average* parameter
        is set to *True* or *False*. If *False*, the vectors are not normalized.

    Returns
    -------
    N : (n_face) numpy.ndarray or (4) list of (n_face) numpy.ndarray
        Normal vector(s) for each surface. If *average* = *True*, the function
        returns an ndarray with the average surface normal vectos. If *average* = *False* ,
        the function returns a cell array {nA,nB,nC,nD} containing the normal vectors
        computed using each vertex of the surface.
    area : (n_face) numpy.ndarray
        The surface areas.

    Examples
    --------
    Here we define a set of vertices for a tensor mesh. We then
    index 4 vertices for an irregular quadrilateral. The
    function *face_info* is used to compute the normal vector
    and the surface area.

    >>> from discretize.utils import face_info
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib as mpl
    >>> mpl.rcParams.update({"font.size": 14})

    Define Corners of a uniform cube.

    >>> h = [1, 1]
    >>> mesh = TensorMesh([h, h, h])
    >>> xyz = mesh.nodes

    Choose the face indices,

    >>> A = np.array([0])
    >>> B = np.array([4])
    >>> C = np.array([26])
    >>> D = np.array([18])

    Compute average surface normal vector (normalized),

    >>> nvec, area = face_info(xyz, A, B, C, D)
    >>> nvec, area
    (array([[-0.70710678,  0.70710678,  0.        ]]), array([4.24264069]))

    Plot surface for example 1 on mesh

    .. collapse:: Expand to show scripting for plot

        >>> fig = plt.figure(figsize=(7, 7))
        >>> ax = fig.gca(projection='3d')
        >>> mesh.plot_grid(ax=ax)
        >>> k = [0, 4, 26, 18, 0]
        >>> xyz_quad = xyz[k, :]
        >>> ax.plot(xyz_quad[:, 0], xyz_quad[:, 1], xyz_quad[:, 2], 'r')
        >>> ax.text(-0.25, 0., 3., 'Area of the surface: {:g} $m^2$'.format(area[0]))
        >>> ax.text(-0.25, 0., 2.8, 'Normal vector: ({:.2f}, {:.2f}, {:.2f})'.format(
        ...     nvec[0, 0], nvec[0, 1], nvec[0, 2])
        ... )
        >>> plt.show()

    In our second example, the vertices are unable to define a flat
    surface in 3D space. However, we will demonstrate the *face_info*
    returns the average normal vector and an approximate surface area.

    Define the face indicies
    >>> A = np.array([0])
    >>> B = np.array([5])
    >>> C = np.array([26])
    >>> D = np.array([18])

    Compute average surface normal vector

    >>> nvec, area = face_info(xyz, A, B, C, D)
    >>> nvec, area
    (array([[-0.4472136 ,  0.89442719,  0.        ]]), array([2.23606798]))

    Plot surface for example 2 on mesh

    .. collapse:: Expand to show scripting for plot

        >>> fig = plt.figure(figsize=(7, 7))
        >>> ax = fig.gca(projection='3d')
        >>> mesh.plot_grid(ax=ax)
        >>> k = [0, 5, 26, 18, 0]
        >>> xyz_quad = xyz[k, :]
        >>> ax.plot(xyz_quad[:, 0], xyz_quad[:, 1], xyz_quad[:, 2], 'g')
        >>> ax.text(-0.25, 0., 3., 'Area of the surface: {:g} $m^2$'.format(area[0]))
        >>> ax.text(-0.25, 0., 2.8, 'Average normal vector: ({:.2f}, {:.2f}, {:.2f})'.format(
        ...     nvec[0, 0], nvec[0, 1], nvec[0, 2])
        ... )
        >>> plt.show()
    """
    if "normalizeNormals" in kwargs:
        warnings.warn(
            "The normalizeNormals keyword argument has been deprecated, please use normalize_normals. "
            "This will be removed in discretize 1.0.0",
            FutureWarning,
        )
        normalize_normals = kwargs["normalizeNormals"]
    if not isinstance(average, bool):
        raise TypeError("average must be a boolean")
    if not isinstance(normalize_normals, bool):
        raise TypeError("normalize_normals must be a boolean")

    AB = xyz[B, :] - xyz[A, :]
    BC = xyz[C, :] - xyz[B, :]
    CD = xyz[D, :] - xyz[C, :]
    DA = xyz[A, :] - xyz[D, :]

    def cross(X, Y):
        return np.c_[
            X[:, 1] * Y[:, 2] - X[:, 2] * Y[:, 1],
            X[:, 2] * Y[:, 0] - X[:, 0] * Y[:, 2],
            X[:, 0] * Y[:, 1] - X[:, 1] * Y[:, 0],
        ]

    nA = cross(AB, DA)
    nB = cross(BC, AB)
    nC = cross(CD, BC)
    nD = cross(DA, CD)

    def length(x):
        return np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)

    def normalize(x):
        return x / np.kron(np.ones((1, x.shape[1])), mkvc(length(x), 2))

    if average:
        # average the normals at each vertex.
        N = (nA + nB + nC + nD) / 4  # this is intrinsically weighted by area
        # normalize
        N = normalize(N)
    else:
        if normalize_normals:
            N = [normalize(nA), normalize(nB), normalize(nC), normalize(nD)]
        else:
            N = [nA, nB, nC, nD]

    # Area calculation
    #
    # Approximate by 4 different triangles, and divide by 2.
    # Each triangle is one half of the length of the cross product
    #
    # So also could be viewed as the average parallelogram.
    #
    # TODO: This does not compute correctly for concave quadrilaterals
    area = (length(nA) + length(nB) + length(nC) + length(nD)) / 4

    return N, area


exampleLrmGrid = deprecate_function(
    example_curvilinear_grid,
    "exampleLrmGrid",
    removal_version="1.0.0",
    future_warn=True,
)
volTetra = deprecate_function(
    volume_tetrahedron, "volTetra", removal_version="1.0.0", future_warn=True
)
indexCube = deprecate_function(
    index_cube, "indexCube", removal_version="1.0.0", future_warn=True
)
faceInfo = deprecate_function(
    face_info, "faceInfo", removal_version="1.0.0", future_warn=True
)
