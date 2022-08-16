#      ___          ___       ___          ___          ___          ___
#     /\  \        /\  \     /\  \        /\  \        /\  \        /\  \
#    /::\  \      /::\  \    \:\  \      /::\  \      /::\  \      /::\  \
#   /:/\:\  \    /:/\:\  \    \:\  \    /:/\:\  \    /:/\:\  \    /:/\:\  \
#  /:/  \:\  \  /:/  \:\  \   /::\  \  /::\~\:\  \  /::\~\:\  \  /::\~\:\  \
# /:/__/ \:\__\/:/__/ \:\__\ /:/\:\__\/:/\:\ \:\__\/:/\:\ \:\__\/:/\:\ \:\__\
# \:\  \ /:/  /\:\  \  \/__//:/  \/__/\/_|::\/:/  /\:\~\:\ \/__/\:\~\:\ \/__/
#  \:\  /:/  /  \:\  \     /:/  /        |:|::/  /  \:\ \:\__\   \:\ \:\__\
#   \:\/:/  /    \:\  \    \/__/         |:|\/__/    \:\ \/__/    \:\ \/__/
#    \::/  /      \:\__\                 |:|  |       \:\__\       \:\__\
#     \/__/        \/__/                  \|__|        \/__/        \/__/
#
#
#
#                      .----------------.----------------.
#                     /|               /|               /|
#                    / |              / |              / |
#                   /  |      6      /  |     7       /  |
#                  /   |            /   |            /   |
#                 .----------------.----+-----------.    |
#                /|    . ---------/|----.----------/|----.
#               / |   /|         / |   /|         / |   /|
#              /  |  / |  4     /  |  / |   5    /  |  / |
#             /   | /  |       /   | /  |       /   | /  |
#            . -------------- .----------------.    |/   |
#            |    . ---+------|----.----+------|----.    |
#            |   /|    .______|___/|____.______|___/|____.
#            |  / |   /    2  |  / |   /     3 |  / |   /
#            | /  |  /        | /  |  /        | /  |  /
#            . ---+---------- . ---+---------- .    | /
#            |    |/          |    |/          |    |/             z
#            |    . ----------|----.-----------|----.              ^   y
#            |   /      0     |   /       1    |   /               |  /
#            |  /             |  /             |  /                | /
#            | /              | /              | /                 o----> x
#            . -------------- . -------------- .
#
#
# Face Refinement:
#
#      2_______________3                    _______________
#      |               |                   |       |       |
#   ^  |               |                   |   2   |   3   |
#   |  |               |                   |       |       |
#   |  |       x       |        --->       |-------+-------|
#   t1 |               |                   |       |       |
#      |               |                   |   0   |   1   |
#      |_______________|                   |_______|_______|
#      0      t0-->    1
#
#
# Face and Edge naming conventions:
#
#                      fZp
#                       |
#                 6 ------eX3------ 7
#                /|     |         / |
#               /eZ2    .        / eZ3
#             eY2 |        fYp eY3  |
#             /   |            / fXp|
#            4 ------eX2----- 5     |
#            |fXm 2 -----eX1--|---- 3          z
#           eZ0  /            |  eY1           ^   y
#            | eY0   .  fYm  eZ1 /             |  /
#            | /     |        | /              | /
#            0 ------eX0------1                o----> x
#                    |
#                   fZm
#
#
#            fX                                  fY
#      2___________3                       2___________3
#      |     e1    |                       |     e1    |
#      |           |                       |           |
#   e0 |     x     | e2      z          e0 |     x     | e2      z
#      |           |         ^             |           |         ^
#      |___________|         |___> y       |___________|         |___> x
#      0    e3     1                       0    e3     1
#           fZ
#      2___________3
#      |     e1    |
#      |           |
#   e0 |     x     | e2      y
#      |           |         ^
#      |___________|         |___> x
#      0    e3     1

from discretize.base import BaseTensorMesh
from discretize.operators import InnerProducts, DiffOperators
from discretize.mixins import InterfaceMixins, TreeMeshIO
from discretize.utils import as_array_n_by_dim
from discretize._extensions.tree_ext import _TreeMesh, TreeCell
import numpy as np
import scipy.sparse as sp
import warnings
from discretize.utils.code_utils import deprecate_property


class TreeMesh(
    _TreeMesh, InnerProducts, DiffOperators, BaseTensorMesh, TreeMeshIO, InterfaceMixins,
):
    """Class for QuadTree (2D) and OcTree (3D) meshes.

    Tree meshes are numerical grids where the dimensions of each cell are powers of 2
    larger than some base cell dimension. Unlike the :class:`~discretize.TensorMesh`
    class, gridded locations and numerical operators for instances of ``TreeMesh``
    cannot be simply constructed using tensor products. Furthermore, each cell
    is an instance of ``TreeMesh`` is an instance of the
    :class:`~discretize.tree_mesh.TreeCell` .

    Parameters
    ----------
    h : (dim) iterable of int, numpy.ndarray, or tuple
        Defines the cell widths of the *underlying tensor mesh* along each axis. The
        length of the iterable object is equal to the dimension of the mesh (2 or 3).
        For a 3D mesh, the list would have the form *[hx, hy, hz]*. The number of cells
        along each axis **must be a power of 2** .

        Along each axis, the user has 3 choices for defining the cells widths for the
        underlying tensor mesh:

        - :class:`int` -> A unit interval is equally discretized into `N` cells.
        - :class:`numpy.ndarray` -> The widths are explicity given for each cell
        - the widths are defined as a :class:`list` of :class:`tuple` of the form *(dh, nc, [npad])*
          where *dh* is the cell width, *nc* is the number of cells, and *npad* (optional)
          is a padding factor denoting exponential increase/decrease in the cell width
          for each cell; e.g. *[(2., 10, -1.3), (2., 50), (2., 10, 1.3)]*

    origin : (dim) iterable, default: 0
        Define the origin or 'anchor point' of the mesh; i.e. the bottom-left-frontmost
        corner. By default, the mesh is anchored such that its origin is at [0, 0, 0].

        For each dimension (x, y or z), The user may set the origin 2 ways:

        - a ``scalar`` which explicitly defines origin along that dimension.
        - **{'0', 'C', 'N'}** a :class:`str` specifying whether the zero coordinate along
          each axis is the first node location ('0'), in the center ('C') or the last
          node location ('N') (see Examples).

    Examples
    --------
    Here we generate a basic 2D tree mesh.

    >>> from discretize import TreeMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    Define base mesh (domain and finest discretization),

    >>> dh = 5    # minimum cell width (base mesh cell width)
    >>> nbc = 64  # number of base mesh cells
    >>> h = dh * np.ones(nbc)
    >>> mesh = TreeMesh([h, h])

    Define corner points for a rectangular box, and subdived the mesh within the box
    to the maximum refinement level.

    >>> x0s = [120.0, 80.0]
    >>> x1s = [240.0, 160.0]
    >>> levels = [mesh.max_level]
    >>> mesh.refine_box(x0s, x1s, levels)

    >>> mesh.plot_grid()
    >>> plt.show()
    """

    _meshType = "TREE"
    _aliases = {
        **BaseTensorMesh._aliases,
        **DiffOperators._aliases,
        **{
            "ntN": "n_total_nodes",
            "ntEx": "n_total_edges_x",
            "ntEy": "n_total_edges_y",
            "ntEz": "n_total_edges_z",
            "ntE": "n_total_edges",
            "ntFx": "n_total_faces_x",
            "ntFy": "n_total_faces_y",
            "ntFz": "n_total_faces_z",
            "ntF": "n_total_faces",
            "nhN": "n_hanging_nodes",
            "nhEx": "n_hanging_edges_x",
            "nhEy": "n_hanging_edges_y",
            "nhEz": "n_hanging_edges_z",
            "nhE": "n_hanging_edges",
            "nhFx": "n_hanging_faces_x",
            "nhFy": "n_hanging_faces_y",
            "nhFz": "n_hanging_faces_z",
            "nhF": "n_hanging_faces",
            "gridhN": "hanging_nodes",
            "gridhFx": "hanging_faces_x",
            "gridhFy": "hanging_faces_y",
            "gridhFz": "hanging_faces_z",
            "gridhEx": "hanging_edges_x",
            "gridhEy": "hanging_edges_y",
            "gridhEz": "hanging_edges_z",
        },
    }
    _items = {"h", "origin", "cell_state"}

    # inheriting stuff from BaseTensorMesh that isn't defined in _QuadTree
    def __init__(self, h=None, origin=None, **kwargs):
        if "x0" in kwargs:
            origin = kwargs.pop("x0")
        super().__init__(h=h, origin=origin)

        cell_state = kwargs.pop("cell_state", None)
        cell_indexes = kwargs.pop("cell_indexes", None)
        cell_levels = kwargs.pop("cell_levels", None)
        if cell_state is None:
            if cell_indexes is not None and cell_levels is not None:
                cell_state = {}
                cell_state["indexes"] = cell_indexes
                cell_state["levels"] = cell_levels
        if cell_state is not None:
            indexes = cell_state["indexes"]
            levels = cell_state["levels"]
            self.__setstate__((indexes, levels))

    def __repr__(self):
        """Plain text representation."""
        mesh_name = "{0!s}TreeMesh".format(("Oc" if self.dim == 3 else "Quad"))

        top = "\n" + mesh_name + ": {0:2.2f}% filled\n\n".format(self.fill * 100)

        # Number of cells per level
        level_count = self._count_cells_per_index()
        non_zero_levels = np.nonzero(level_count)[0]
        cell_display = ["Level : Number of cells"]
        cell_display.append("-----------------------")
        for level in non_zero_levels:
            cell_display.append("{:^5} : {:^15}".format(level, level_count[level]))
        cell_display.append("-----------------------")
        cell_display.append("Total : {:^15}".format(self.nC))

        extent_display = ["            Mesh Extent       "]
        extent_display.append("        min     ,     max     ")
        extent_display.append("   ---------------------------")
        dim_label = {0: "x", 1: "y", 2: "z"}
        for dim in range(self.dim):
            n_vector = getattr(self, "nodes_" + dim_label[dim])
            extent_display.append(
                "{}: {:^13},{:^13}".format(dim_label[dim], n_vector[0], n_vector[-1])
            )

        for i, line in enumerate(extent_display):
            if i == len(cell_display):
                cell_display.append(" " * (len(cell_display[0]) - 3 - len(line)))
            cell_display[i] += 3 * " " + line

        h_display = ["     Cell Widths    "]
        h_display.append("    min   ,   max   ")
        h_display.append("-" * (len(h_display[0])))
        h_gridded = self.h_gridded
        mins = np.min(h_gridded, axis=0)
        maxs = np.max(h_gridded, axis=0)
        for dim in range(self.dim):
            h_display.append("{:^10}, {:^10}".format(mins[dim], maxs[dim]))

        for i, line in enumerate(h_display):
            if i == len(cell_display):
                cell_display.append(" " * len(cell_display[0]))
            cell_display[i] += 3 * " " + line

        return top + "\n".join(cell_display)

    def _repr_html_(self):
        """html representation"""
        mesh_name = "{0!s}TreeMesh".format(("Oc" if self.dim == 3 else "Quad"))
        level_count = self._count_cells_per_index()
        non_zero_levels = np.nonzero(level_count)[0]
        dim_label = {0: "x", 1: "y", 2: "z"}
        h_gridded = self.h_gridded
        mins = np.min(h_gridded, axis=0)
        maxs = np.max(h_gridded, axis=0)

        style = " style='padding: 5px 20px 5px 20px;'"
        # Cell level table:
        cel_tbl = "<table>\n"
        cel_tbl += "<tr>\n"
        cel_tbl += "<th" + style + ">Level</th>\n"
        cel_tbl += "<th" + style + ">Number of cells</th>\n"
        cel_tbl += "</tr>\n"
        for level in non_zero_levels:
            cel_tbl += "<tr>\n"
            cel_tbl += "<td" + style + ">{}</td>\n".format(level)
            cel_tbl += "<td" + style + ">{}</td>\n".format(level_count[level])
            cel_tbl += "</tr>\n"
        cel_tbl += "<tr>\n"
        cel_tbl += (
            "<td style='font-weight: bold; padding: 5px 20px 5px 20px;'> Total </td>\n"
        )
        cel_tbl += "<td" + style + "> {} </td>\n".format(self.nC)
        cel_tbl += "</tr>\n"
        cel_tbl += "</table>\n"

        det_tbl = "<table>\n"
        det_tbl += "<tr>\n"
        det_tbl += "<th></th>\n"
        det_tbl += "<th" + style + " colspan='2'>Mesh extent</th>\n"
        det_tbl += "<th" + style + " colspan='2'>Cell widths</th>\n"
        det_tbl += "</tr>\n"

        det_tbl += "<tr>\n"
        det_tbl += "<th></th>\n"
        det_tbl += "<th" + style + ">min</th>\n"
        det_tbl += "<th" + style + ">max</th>\n"
        det_tbl += "<th" + style + ">min</th>\n"
        det_tbl += "<th" + style + ">max</th>\n"
        det_tbl += "</tr>\n"
        for dim in range(self.dim):
            n_vector = getattr(self, "nodes_" + dim_label[dim])
            det_tbl += "<tr>\n"
            det_tbl += "<td" + style + ">{}</td>\n".format(dim_label[dim])
            det_tbl += "<td" + style + ">{}</td>\n".format(n_vector[0])
            det_tbl += "<td" + style + ">{}</td>\n".format(n_vector[-1])
            det_tbl += "<td" + style + ">{}</td>\n".format(mins[dim])
            det_tbl += "<td" + style + ">{}</td>\n".format(maxs[dim])
            det_tbl += "</tr>\n"
        det_tbl += "</table>\n"

        full_tbl = "<table>\n"
        full_tbl += "<tr>\n"
        full_tbl += "<td style='font-weight: bold; font-size: 1.2em; text-align: center;'>{}</td>\n".format(
            mesh_name
        )
        full_tbl += "<td style='font-size: 1.2em; text-align: center;' colspan='2'>{0:2.2f}% filled</td>\n".format(
            100 * self.fill
        )
        full_tbl += "</tr>\n"
        full_tbl += "<tr>\n"

        full_tbl += "<td>\n"
        full_tbl += cel_tbl
        full_tbl += "</td>\n"

        full_tbl += "<td>\n"
        full_tbl += det_tbl
        full_tbl += "</td>\n"

        full_tbl += "</tr>\n"
        full_tbl += "</table>\n"

        return full_tbl

    @BaseTensorMesh.origin.setter
    def origin(self, value):
        # first use the BaseTensorMesh to set the origin to handle "0, C, N"
        BaseTensorMesh.origin.fset(self, value)
        # then update the TreeMesh with the hidden value
        self._set_origin(self._origin)

    @property
    def vntF(self):
        """
        Vector number of total faces along each axis

        This property returns the total number of hanging and
        non-hanging faces along each axis direction. The returned
        quantity is a list of integers of the form [nFx,nFy,nFz].

        Returns
        -------
        list of int
            Vector number of total faces along each axis
        """
        return [self.ntFx, self.ntFy] + ([] if self.dim == 2 else [self.ntFz])

    @property
    def vntE(self):
        """
        Vector number of total edges along each axis

        This property returns the total number of hanging and
        non-hanging edges along each axis direction. The returned
        quantity is a list of integers of the form [nEx,nEy,nEz].

        Returns
        -------
        list of int
            Vector number of total edges along each axis
        """
        return [self.ntEx, self.ntEy] + ([] if self.dim == 2 else [self.ntEz])

    @property
    def stencil_cell_gradient(self):
        if getattr(self, "_stencil_cell_gradient", None) is None:

            self._stencil_cell_gradient = sp.vstack(
                [self.stencil_cell_gradient_x, self.stencil_cell_gradient_y]
            )
            if self.dim == 3:
                self._stencil_cell_gradient = sp.vstack(
                    [self._stencil_cell_gradient, self.stencil_cell_gradient_z]
                )

        return self._stencil_cell_gradient

    @property
    def cell_gradient(self):
        if getattr(self, "_cell_gradient", None) is None:

            i_s = self.face_boundary_indices

            ix = np.ones(self.nFx)
            ix[i_s[0]] = 0.0
            ix[i_s[1]] = 0.0
            Pafx = sp.diags(ix)

            iy = np.ones(self.nFy)
            iy[i_s[2]] = 0.0
            iy[i_s[3]] = 0.0
            Pafy = sp.diags(iy)

            MfI = self.get_face_inner_product(invMat=True)

            if self.dim == 2:
                Pi = sp.block_diag([Pafx, Pafy])

            elif self.dim == 3:
                iz = np.ones(self.nFz)
                iz[i_s[4]] = 0.0
                iz[i_s[5]] = 0.0
                Pafz = sp.diags(iz)
                Pi = sp.block_diag([Pafx, Pafy, Pafz])

            self._cell_gradient = (
                -Pi * MfI * self.face_divergence.T * sp.diags(self.cell_volumes)
            )

        return self._cell_gradient

    @property
    def cell_gradient_x(self):
        if getattr(self, "_cell_gradient_x", None) is None:

            nFx = self.nFx
            i_s = self.face_boundary_indices

            ix = np.ones(self.nFx)
            ix[i_s[0]] = 0.0
            ix[i_s[1]] = 0.0
            Pafx = sp.diags(ix)

            MfI = self.get_face_inner_product(invMat=True)
            MfIx = sp.diags(MfI.diagonal()[:nFx])

            self._cell_gradient_x = (
                -Pafx * MfIx * self.face_x_divergence.T * sp.diags(self.cell_volumes)
            )

        return self._cell_gradient_x

    @property
    def cell_gradient_y(self):
        if getattr(self, "_cell_gradient_y", None) is None:

            nFx = self.nFx
            nFy = self.nFy
            i_s = self.face_boundary_indices

            iy = np.ones(self.nFy)
            iy[i_s[2]] = 0.0
            iy[i_s[3]] = 0.0
            Pafy = sp.diags(iy)

            MfI = self.get_face_inner_product(invMat=True)
            MfIy = sp.diags(MfI.diagonal()[nFx : nFx + nFy])

            self._cell_gradient_y = (
                -Pafy * MfIy * self.face_y_divergence.T * sp.diags(self.cell_volumes)
            )

        return self._cell_gradient_y

    @property
    def cell_gradient_z(self):
        if self.dim == 2:
            raise TypeError("z derivative not defined in 2D")
        if getattr(self, "_cell_gradient_z", None) is None:

            nFx = self.nFx
            nFy = self.nFy
            i_s = self.face_boundary_indices

            iz = np.ones(self.nFz)
            iz[i_s[4]] = 0.0
            iz[i_s[5]] = 0.0
            Pafz = sp.diags(iz)

            MfI = self.get_face_inner_product(invMat=True)
            MfIz = sp.diags(MfI.diagonal()[nFx + nFy :])

            self._cell_gradient_z = (
                -Pafz * MfIz * self.face_z_divergence.T * sp.diags(self.cell_volumes)
            )

        return self._cell_gradient_z

    @property
    def face_x_divergence(self):
        if getattr(self, "_face_x_divergence", None) is None:
            self._face_x_divergence = self.face_divergence[:, : self.nFx]
        return self._face_x_divergence

    @property
    def face_y_divergence(self):
        if getattr(self, "_face_y_divergence", None) is None:
            self._face_y_divergence = self.face_divergence[
                :, self.nFx : self.nFx + self.nFy
            ]
        return self._face_y_divergence

    @property
    def face_z_divergence(self):
        if getattr(self, "_face_z_divergence", None) is None:
            self._face_z_divergence = self.face_divergence[:, self.nFx + self.nFy :]
        return self._face_z_divergence

    def point2index(self, locs):
        locs = as_array_n_by_dim(locs, self.dim)
        inds = self._get_containing_cell_indexes(locs)
        return inds

    def cell_levels_by_index(self, indices):
        """Fast function to return a list of levels for the given cell indices

        Parameters
        ----------
        index: (N) array_like
            Cell indexes to query

        Returns
        -------
        (N) numpy.ndarray of int
            Levels for the cells.
        """

        return self._cell_levels_by_indexes(indices)

    def get_interpolation_matrix(
        self, locs, location_type="cell_centers", zeros_outside=False, **kwargs
    ):
        if "locType" in kwargs:
            warnings.warn(
                "The locType keyword argument has been deprecated, please use location_type. "
                "This will be removed in discretize 1.0.0",
                FutureWarning,
            )
            location_type = kwargs["locType"]
        if "zerosOutside" in kwargs:
            warnings.warn(
                "The zerosOutside keyword argument has been deprecated, please use zeros_outside. "
                "This will be removed in discretize 1.0.0",
                FutureWarning,
            )
            zeros_outside = kwargs["zerosOutside"]
        locs = as_array_n_by_dim(locs, self.dim)
        location_type = self._parse_location_type(location_type)

        if self.dim == 2 and "z" in location_type:
            raise NotImplementedError("Unable to interpolate from Z edges/faces in 2D")

        locs = np.require(np.atleast_2d(locs), dtype=np.float64, requirements="C")

        if location_type == "nodes":
            Av = self._getNodeIntMat(locs, zeros_outside)
        elif location_type in ["edges_x", "edges_y", "edges_z"]:
            Av = self._getEdgeIntMat(locs, zeros_outside, location_type[-1])
        elif location_type in ["faces_x", "faces_y", "faces_z"]:
            Av = self._getFaceIntMat(locs, zeros_outside, location_type[-1])
        elif location_type in ["cell_centers"]:
            Av = self._getCellIntMat(locs, zeros_outside)
        else:
            raise ValueError(
                "Location must be a grid location, not {}".format(location_type)
            )
        return Av

    @property
    def permute_cells(self):
        """Permutation matrix re-ordering of cells sorted by x, then y, then z

        Returns
        -------
        (n_cells, n_cells) scipy.sparse.csr_matrix
        """
        # TODO: cache these?
        P = np.lexsort(self.gridCC.T)  # sort by x, then y, then z
        return sp.identity(self.nC).tocsr()[P]

    @property
    def permute_faces(self):
        """Permutation matrix re-ordering of faces sorted by x, then y, then z

        Returns
        -------
        (n_faces, n_faces) scipy.sparse.csr_matrix
        """
        # TODO: cache these?
        Px = np.lexsort(self.gridFx.T)
        Py = np.lexsort(self.gridFy.T) + self.nFx
        if self.dim == 2:
            P = np.r_[Px, Py]
        else:
            Pz = np.lexsort(self.gridFz.T) + (self.nFx + self.nFy)
            P = np.r_[Px, Py, Pz]
        return sp.identity(self.nF).tocsr()[P]

    @property
    def permute_edges(self):
        """Permutation matrix re-ordering of edges sorted by x, then y, then z

        Returns
        -------
        (n_edges, n_edges) scipy.sparse.csr_matrix
        """
        # TODO: cache these?
        Px = np.lexsort(self.gridEx.T)
        Py = np.lexsort(self.gridEy.T) + self.nEx
        if self.dim == 2:
            P = np.r_[Px, Py]
        if self.dim == 3:
            Pz = np.lexsort(self.gridEz.T) + (self.nEx + self.nEy)
            P = np.r_[Px, Py, Pz]
        return sp.identity(self.nE).tocsr()[P]

    @property
    def cell_state(self):
        """ The current state of the cells on the mesh.

        This represents the x, y, z indices of the cells in the base tensor mesh, as
        well as their levels. It can be used to reconstruct the mesh.

        Returns
        -------
        dict
            dictionary with two entries:

            - ``"indexes"``: the indexes of the cells
            - ``"levels"``: the levels of the cells
        """
        indexes, levels = self.__getstate__()
        return {"indexes": indexes.tolist(), "levels": levels.tolist()}

    def validate(self):
        return self.finalized

    def equals(self, other):
        try:
            if self.finalized and other.finalized:
                return super().equals(other)
        except AttributeError:
            pass
        return False

    def __reduce__(self):
        return TreeMesh, (self.h, self.origin), self.__getstate__()

    cellGrad = deprecate_property("cell_gradient", "cellGrad", removal_version="1.0.0", future_warn=True)
    cellGradx = deprecate_property(
        "cell_gradient_x", "cellGradx", removal_version="1.0.0", future_warn=True
    )
    cellGrady = deprecate_property(
        "cell_gradient_y", "cellGrady", removal_version="1.0.0", future_warn=True
    )
    cellGradz = deprecate_property(
        "cell_gradient_z", "cellGradz", removal_version="1.0.0", future_warn=True
    )
    cellGradStencil = deprecate_property(
        "cell_gradient_stencil", "cellGradStencil", removal_version="1.0.0", future_warn=True
    )
    faceDivx = deprecate_property(
        "face_x_divergence", "faceDivx", removal_version="1.0.0", future_warn=True
    )
    faceDivy = deprecate_property(
        "face_y_divergence", "faceDivy", removal_version="1.0.0", future_warn=True
    )
    faceDivz = deprecate_property(
        "face_z_divergence", "faceDivz", removal_version="1.0.0", future_warn=True
    )
    maxLevel = deprecate_property("max_used_level", "maxLevel", removal_version="1.0.0", future_warn=True)
    areaFx = deprecate_property("face_x_areas", "areaFx", removal_version="1.0.0", future_warn=True)
    areaFy = deprecate_property("face_y_areas", "areaFy", removal_version="1.0.0", future_warn=True)
    areaFz = deprecate_property("face_z_areas", "areaFz", removal_version="1.0.0", future_warn=True)
    edgeEx = deprecate_property("edge_x_lengths", "edgeEx", removal_version="1.0.0", future_warn=True)
    edgeEy = deprecate_property("edge_y_lengths", "edgeEy", removal_version="1.0.0", future_warn=True)
    edgeEz = deprecate_property("edge_z_lengths", "edgeEz", removal_version="1.0.0", future_warn=True)
    permuteCC = deprecate_property(
        "permute_cells", "permuteCC", removal_version="1.0.0", future_warn=True
    )
    permuteF = deprecate_property("permute_faces", "permuteF", removal_version="1.0.0", future_warn=True)
    permuteE = deprecate_property("permute_edges", "permuteE", removal_version="1.0.0", future_warn=True)
    faceBoundaryInd = deprecate_property(
        "face_boundary_indices", "faceBoundaryInd", removal_version="1.0.0", future_warn=True
    )
    cellBoundaryInd = deprecate_property(
        "cell_boundary_indices", "cellBoundaryInd", removal_version="1.0.0", future_warn=True
    )
    _aveCC2FxStencil = deprecate_property(
        "average_cell_to_total_face_x", "_aveCC2FxStencil", removal_version="1.0.0", future_warn=True
    )
    _aveCC2FyStencil = deprecate_property(
        "average_cell_to_total_face_y", "_aveCC2FyStencil", removal_version="1.0.0", future_warn=True
    )
    _aveCC2FzStencil = deprecate_property(
        "average_cell_to_total_face_z", "_aveCC2FzStencil", removal_version="1.0.0", future_warn=True
    )
    _cellGradStencil = deprecate_property(
        "stencil_cell_gradient", "_cellGradStencil", removal_version="1.0.0", future_warn=True
    )
    _cellGradxStencil = deprecate_property(
        "stencil_cell_gradient_x", "_cellGradxStencil", removal_version="1.0.0", future_warn=True
    )
    _cellGradyStencil = deprecate_property(
        "stencil_cell_gradient_y", "_cellGradyStencil", removal_version="1.0.0", future_warn=True
    )
    _cellGradzStencil = deprecate_property(
        "stencil_cell_gradient_z", "_cellGradzStencil", removal_version="1.0.0", future_warn=True
    )
