"""
Base class for tensor-product style meshes
"""

import numpy as np
import scipy.sparse as sp
import properties

from discretize.base.base_mesh import BaseMesh
from discretize.utils import (
    is_scalar,
    as_array_n_by_dim,
    mesh_tensor,
    mkvc,
    ndgrid,
    spzeros,
    sdiag,
    sdinv,
    TensorType,
    interpolation_matrix,
)
from discretize.utils.code_utils import deprecate_method, deprecate_property
import warnings


class BaseTensorMesh(BaseMesh):
    """
    Base class for tensor-product style meshes

    This class contains properites and methods that are common to cartesian
    and cylindrical meshes defined by tensor-produts of vectors describing
    cell spacings.

    Do not use this class directly, instead, inherit it if you plan to develop
    a tensor-style mesh (e.g. a spherical mesh) or use the
    :meth:`discretize.TensorMesh` class to create a cartesian tensor mesh.

    """

    _meshType = "BASETENSOR"
    _aliases = {
        **BaseMesh._aliases,
        **{
            "gridCC": "grid_cell_centers",
            "gridN": "grid_nodes",
            "gridFx": "grid_faces_x",
            "gridFy": "grid_faces_y",
            "gridFz": "grid_faces_z",
            "gridEx": "grid_edges_x",
            "gridEy": "grid_edges_y",
            "gridEz": "grid_edges_z",
        },
    }

    _unitDimensions = [1, 1, 1]

    # properties
    h = properties.Tuple(
        "h is a list containing the cell widths of the tensor mesh in each "
        "dimension.",
        properties.Array(
            "widths of the tensor mesh in a single dimension",
            dtype=float,
            shape=("*",),
        ),
        min_length=1,
        max_length=3,
        coerce=True,
        required=True,
    )

    def __init__(self, h=None, x0=None, **kwargs):

        h_in = h
        x0_in = x0

        # Sanity Checks
        assert type(h_in) in [list, tuple], "h_in must be a list, not {}".format(
            type(h_in)
        )
        assert len(h_in) in [
            1,
            2,
            3,
        ], "h_in must be of dimension 1, 2, or 3 not {}".format(len(h_in))

        # build h
        h = list(range(len(h_in)))
        for i, h_i in enumerate(h_in):
            if is_scalar(h_i) and type(h_i) is not np.ndarray:
                # This gives you something over the unit cube.
                h_i = self._unitDimensions[i] * np.ones(int(h_i)) / int(h_i)
            elif type(h_i) is list:
                h_i = mesh_tensor(h_i)
            assert isinstance(h_i, np.ndarray), "h[{0:d}] is not a numpy array.".format(
                i
            )
            assert len(h_i.shape) == 1, "h[{0:d}] must be a 1D numpy array.".format(i)
            h[i] = h_i[:]  # make a copy.

        # Origin of the mesh
        x0 = np.zeros(len(h))

        if x0_in is not None:
            assert len(h) == len(x0_in), "Dimension mismatch. x0 != len(h)"
            for i in range(len(h)):
                x_i, h_i = x0_in[i], h[i]
                if is_scalar(x_i):
                    x0[i] = x_i
                elif x_i == "0":
                    x0[i] = 0.0
                elif x_i == "C":
                    x0[i] = -h_i.sum() * 0.5
                elif x_i == "N":
                    x0[i] = -h_i.sum()
                else:
                    raise Exception(
                        "x0[{0:d}] must be a scalar or '0' to be zero, "
                        "'C' to center, or 'N' to be negative. The input value"
                        " {1} {2} is invalid".format(i, x_i, type(x_i))
                    )

        if "n" in kwargs.keys():
            n = kwargs.pop("n")
            assert (
                n == np.array([x.size for x in h])
            ).all(), "Dimension mismatch. The provided n doesn't "
        else:
            n = np.array([x.size for x in h])

        super(BaseTensorMesh, self).__init__(n, x0=x0, **kwargs)

        # Ensure h contains 1D vectors
        self.h = [mkvc(x.astype(float)) for x in h]

    @property
    def grid_nodes_x(self):
        """Nodal grid vector (1D) in the x direction."""
        return np.r_[self.x0[0], self.h[0]].cumsum()

    @property
    def grid_nodes_y(self):
        """Nodal grid vector (1D) in the y direction."""
        return None if self.dim < 2 else np.r_[self.x0[1], self.h[1]].cumsum()

    @property
    def grid_nodes_z(self):
        """Nodal grid vector (1D) in the z direction."""
        return None if self.dim < 3 else np.r_[self.x0[2], self.h[2]].cumsum()

    @property
    def grid_cell_centers_x(self):
        """Cell-centered grid vector (1D) in the x direction."""
        nodes = self.grid_nodes_x
        return (nodes[1:] + nodes[:-1]) / 2

    @property
    def grid_cell_centers_y(self):
        """Cell-centered grid vector (1D) in the y direction."""
        if self.dim < 2:
            return None
        nodes = self.grid_nodes_y
        return (nodes[1:] + nodes[:-1]) / 2

    @property
    def grid_cell_centers_z(self):
        """Cell-centered grid vector (1D) in the z direction."""
        if self.dim < 3:
            return None
        nodes = self.grid_nodes_z
        return (nodes[1:] + nodes[:-1]) / 2

    @property
    def grid_cell_centers(self):
        """Cell-centered grid."""
        return self._getTensorGrid("CC")

    @property
    def grid_nodes(self):
        """Nodal grid."""
        return self._getTensorGrid("N")

    @property
    def h_gridded(self):
        """
        Returns an (nC, dim) numpy array with the widths of all cells in order
        """
        return ndgrid(*self.h)

    @property
    def grid_faces_x(self):
        """Face staggered grid in the x direction."""
        if self.nFx == 0:
            return
        return self._getTensorGrid("Fx")

    @property
    def grid_faces_y(self):
        """Face staggered grid in the y direction."""
        if self.nFy == 0 or self.dim < 2:
            return
        return self._getTensorGrid("Fy")

    @property
    def grid_faces_z(self):
        """Face staggered grid in the z direction."""
        if self.nFz == 0 or self.dim < 3:
            return
        return self._getTensorGrid("Fz")

    @property
    def grid_edges_x(self):
        """Edge staggered grid in the x direction."""
        if self.nEx == 0:
            return
        return self._getTensorGrid("Ex")

    @property
    def grid_edges_y(self):
        """Edge staggered grid in the y direction."""
        if self.nEy == 0 or self.dim < 2:
            return
        return self._getTensorGrid("Ey")

    @property
    def grid_edges_z(self):
        """Edge staggered grid in the z direction."""
        if self.nEz == 0 or self.dim < 3:
            return
        return self._getTensorGrid("Ez")

    def _getTensorGrid(self, key):
        if getattr(self, "_grid" + key, None) is None:
            setattr(self, "_grid" + key, ndgrid(self.get_tensor(key)))
        return getattr(self, "_grid" + key)

    def get_tensor(self, key):
        """Returns a tensor list.

        Parameters
        ----------
        key : str
            Which tensor (see below)

            key can be::

                'CC'    -> scalar field defined on cell centers
                'N'     -> scalar field defined on nodes
                'Fx'    -> x-component of field defined on faces
                'Fy'    -> y-component of field defined on faces
                'Fz'    -> z-component of field defined on faces
                'Ex'    -> x-component of field defined on edges
                'Ey'    -> y-component of field defined on edges
                'Ez'    -> z-component of field defined on edges

        Returns
        -------
        list
            list of the tensors that make up the mesh.

        """

        if key == "Fx":
            ten = [
                self.grid_nodes_x,
                self.grid_cell_centers_y,
                self.grid_cell_centers_z,
            ]
        elif key == "Fy":
            ten = [
                self.grid_cell_centers_x,
                self.grid_nodes_y,
                self.grid_cell_centers_z,
            ]
        elif key == "Fz":
            ten = [
                self.grid_cell_centers_x,
                self.grid_cell_centers_y,
                self.grid_nodes_z,
            ]
        elif key == "Ex":
            ten = [self.grid_cell_centers_x, self.grid_nodes_y, self.grid_nodes_z]
        elif key == "Ey":
            ten = [self.grid_nodes_x, self.grid_cell_centers_y, self.grid_nodes_z]
        elif key == "Ez":
            ten = [self.grid_nodes_x, self.grid_nodes_y, self.grid_cell_centers_z]
        elif key == "CC":
            ten = [
                self.grid_cell_centers_x,
                self.grid_cell_centers_y,
                self.grid_cell_centers_z,
            ]
        elif key == "N":
            ten = [self.grid_nodes_x, self.grid_nodes_y, self.grid_nodes_z]

        return [t for t in ten if t is not None]

    # --------------- Methods ---------------------

    def is_inside(self, pts, loc_type="N", **kwargs):
        """
        Determines if a set of points are inside a mesh.

        :param numpy.ndarray pts: Location of points to test
        :rtype: numpy.ndarray
        :return: inside, numpy array of booleans
        """
        if "locType" in kwargs:
            warnings.warn(
                "The locType keyword argument has been deprecated, please use loc_type. "
                "This will be removed in discretize 1.0.0",
                FutureWarning,
            )
            loc_type = kwargs["locType"]
        pts = as_array_n_by_dim(pts, self.dim)

        tensors = self.get_tensor(loc_type)

        if loc_type == "N" and self._meshType == "CYL":
            # NOTE: for a CYL mesh we add a node to check if we are inside in
            # the radial direction!
            tensors[0] = np.r_[0.0, tensors[0]]
            tensors[1] = np.r_[tensors[1], 2.0 * np.pi]

        inside = np.ones(pts.shape[0], dtype=bool)
        for i, tensor in enumerate(tensors):
            TOL = np.diff(tensor).min() * 1.0e-10
            inside = (
                inside
                & (pts[:, i] >= tensor.min() - TOL)
                & (pts[:, i] <= tensor.max() + TOL)
            )
        return inside

    def _getInterpolationMat(self, loc, loc_type="CC", zeros_outside=False):
        """Produces interpolation matrix

        Parameters
        ----------
        loc : numpy.ndarray
            Location of points to interpolate to

        loc_type: stc
            What to interpolate

            loc_type can be::

                'Ex'    -> x-component of field defined on edges
                'Ey'    -> y-component of field defined on edges
                'Ez'    -> z-component of field defined on edges
                'Fx'    -> x-component of field defined on faces
                'Fy'    -> y-component of field defined on faces
                'Fz'    -> z-component of field defined on faces
                'N'     -> scalar field defined on nodes
                'CC'    -> scalar field defined on cell centers
                'CCVx'  -> x-component of vector field defined on cell centers
                'CCVy'  -> y-component of vector field defined on cell centers
                'CCVz'  -> z-component of vector field defined on cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            M, the interpolation matrix

        """

        loc = as_array_n_by_dim(loc, self.dim)

        if not zeros_outside:
            assert np.all(self.is_inside(loc)), "Points outside of mesh"
        else:
            indZeros = np.logical_not(self.is_inside(loc))
            loc[indZeros, :] = np.array([v.mean() for v in self.get_tensor("CC")])

        if loc_type in ["Fx", "Fy", "Fz", "Ex", "Ey", "Ez"]:
            ind = {"x": 0, "y": 1, "z": 2}[loc_type[1]]
            assert self.dim >= ind, "mesh is not high enough dimension."
            nF_nE = self.vnF if "F" in loc_type else self.vnE
            components = [spzeros(loc.shape[0], n) for n in nF_nE]
            components[ind] = interpolation_matrix(loc, *self.get_tensor(loc_type))
            # remove any zero blocks (hstack complains)
            components = [comp for comp in components if comp.shape[1] > 0]
            Q = sp.hstack(components)

        elif loc_type in ["CC", "N"]:
            Q = interpolation_matrix(loc, *self.get_tensor(loc_type))

        elif loc_type in ["CCVx", "CCVy", "CCVz"]:
            Q = interpolation_matrix(loc, *self.get_tensor("CC"))
            Z = spzeros(loc.shape[0], self.nC)
            if loc_type == "CCVx":
                Q = sp.hstack([Q, Z, Z])
            elif loc_type == "CCVy":
                Q = sp.hstack([Z, Q, Z])
            elif loc_type == "CCVz":
                Q = sp.hstack([Z, Z, Q])

        else:
            raise NotImplementedError(
                "getInterpolationMat: loc_type=="
                + loc_type
                + " and mesh.dim=="
                + str(self.dim)
            )

        if zeros_outside:
            Q[indZeros, :] = 0

        return Q.tocsr()

    def get_interpolation_matrix(
        self, loc, loc_type="CC", zeros_outside=False, **kwargs
    ):
        """Produces linear interpolation matrix

        Parameters
        ----------
        loc : numpy.ndarray
            Location of points to interpolate to

        loc_type : str
            What to interpolate (see below)

            loc_type can be::

                'Ex'    -> x-component of field defined on edges
                'Ey'    -> y-component of field defined on edges
                'Ez'    -> z-component of field defined on edges
                'Fx'    -> x-component of field defined on faces
                'Fy'    -> y-component of field defined on faces
                'Fz'    -> z-component of field defined on faces
                'N'     -> scalar field defined on nodes
                'CC'    -> scalar field defined on cell centers
                'CCVx'  -> x-component of vector field defined on cell centers
                'CCVy'  -> y-component of vector field defined on cell centers
                'CCVz'  -> z-component of vector field defined on cell centers

        Returns
        -------

        scipy.sparse.csr_matrix
            M, the interpolation matrix

        """
        if "locType" in kwargs:
            warnings.warn(
                "The locType keyword argument has been deprecated, please use loc_type. "
                "This will be removed in discretize 1.0.0",
                FutureWarning,
            )
            loc_type = kwargs["locType"]
        if "zerosOutside" in kwargs:
            warnings.warn(
                "The zerosOutside keyword argument has been deprecated, please use loc_type. "
                "This will be removed in discretize 1.0.0",
                FutureWarning,
            )
            zeros_outside = kwargs["zerosOutside"]
        return self._getInterpolationMat(loc, loc_type, zeros_outside)

    def _fastInnerProduct(self, proj_type, prop=None, inv_prop=False, inv_mat=False):
        """Fast version of getFaceInnerProduct.
            This does not handle the case of a full tensor prop.

        Parameters
        ----------

        prop : numpy.array
            material property (tensor properties are possible) at each cell center (nC, (1, 3, or 6))

        proj_type : str
            'E' or 'F'

        returnP : bool
            returns the projection matrices

        inv_prop : bool
            inverts the material property

        inv_mat : bool
            inverts the matrix

        Returns
        -------
        scipy.sparse.csr_matrix
            M, the inner product matrix (nF, nF)

        """
        assert proj_type in [
            "F",
            "E",
        ], "proj_type must be 'F' for faces or 'E' for edges"

        if prop is None:
            prop = np.ones(self.nC)

        if inv_prop:
            prop = 1.0 / prop

        if is_scalar(prop):
            prop = prop * np.ones(self.nC)

        # number of elements we are averaging (equals dim for regular
        # meshes, but for cyl, where we use symmetry, it is 1 for edge
        # variables and 2 for face variables)
        if self._meshType == "CYL":
            shape = getattr(self, "vn" + proj_type)
            n_elements = sum([1 if x != 0 else 0 for x in shape])
        else:
            n_elements = self.dim

        # Isotropic? or anisotropic?
        if prop.size == self.nC:
            Av = getattr(self, "ave" + proj_type + "2CC")
            Vprop = self.cell_volumes * mkvc(prop)
            M = n_elements * sdiag(Av.T * Vprop)

        elif prop.size == self.nC * self.dim:
            Av = getattr(self, "ave" + proj_type + "2CCV")

            # if cyl, then only certain components are relevant due to symmetry
            # for faces, x, z matters, for edges, y (which is theta) matters
            if self._meshType == "CYL":
                if proj_type == "E":
                    prop = prop[:, 1]  # this is the action of a projection mat
                elif proj_type == "F":
                    prop = prop[:, [0, 2]]

            V = sp.kron(sp.identity(n_elements), sdiag(self.cell_volumes))
            M = sdiag(Av.T * V * mkvc(prop))
        else:
            return None

        if inv_mat:
            return sdinv(M)
        else:
            return M

    def _fastInnerProductDeriv(self, proj_type, prop, inv_prop=False, inv_mat=False):
        """

        Parameters
        ----------

        proj_type : str
            'E' or 'F'

        tensorType : TensorType
            type of the tensor

        inv_prop : bool
            inverts the material property

        inv_mat : bool
            inverts the matrix


        Returns
        -------
        function
            dMdmu, the derivative of the inner product matrix

        """
        assert proj_type in ["F", "E"], (
            "proj_type must be 'F' for faces or 'E'" " for edges"
        )

        tensorType = TensorType(self, prop)

        dMdprop = None

        if inv_mat or inv_prop:
            MI = self._fastInnerProduct(
                proj_type, prop, inv_prop=inv_prop, inv_mat=inv_mat
            )

        # number of elements we are averaging (equals dim for regular
        # meshes, but for cyl, where we use symmetry, it is 1 for edge
        # variables and 2 for face variables)
        if self._meshType == "CYL":
            shape = getattr(self, "vn" + proj_type)
            n_elements = sum([1 if x != 0 else 0 for x in shape])
        else:
            n_elements = self.dim

        if tensorType == 0:  # isotropic, constant
            Av = getattr(self, "ave" + proj_type + "2CC")
            V = sdiag(self.cell_volumes)
            ones = sp.csr_matrix(
                (np.ones(self.nC), (range(self.nC), np.zeros(self.nC))),
                shape=(self.nC, 1),
            )
            if not inv_mat and not inv_prop:
                dMdprop = n_elements * Av.T * V * ones
            elif inv_mat and inv_prop:
                dMdprop = n_elements * (
                    sdiag(MI.diagonal() ** 2) * Av.T * V * ones * sdiag(1.0 / prop ** 2)
                )
            elif inv_prop:
                dMdprop = n_elements * Av.T * V * sdiag(-1.0 / prop ** 2)
            elif inv_mat:
                dMdprop = n_elements * (sdiag(-MI.diagonal() ** 2) * Av.T * V)

        elif tensorType == 1:  # isotropic, variable in space
            Av = getattr(self, "ave" + proj_type + "2CC")
            V = sdiag(self.cell_volumes)
            if not inv_mat and not inv_prop:
                dMdprop = n_elements * Av.T * V
            elif inv_mat and inv_prop:
                dMdprop = n_elements * (
                    sdiag(MI.diagonal() ** 2) * Av.T * V * sdiag(1.0 / prop ** 2)
                )
            elif inv_prop:
                dMdprop = n_elements * Av.T * V * sdiag(-1.0 / prop ** 2)
            elif inv_mat:
                dMdprop = n_elements * (sdiag(-MI.diagonal() ** 2) * Av.T * V)

        elif tensorType == 2:  # anisotropic
            Av = getattr(self, "ave" + proj_type + "2CCV")
            V = sp.kron(sp.identity(self.dim), sdiag(self.cell_volumes))

            if self._meshType == "CYL":
                Zero = sp.csr_matrix((self.nC, self.nC))
                Eye = sp.eye(self.nC)
                if proj_type == "E":
                    P = sp.hstack([Zero, Eye, Zero])
                    # print(P.todense())
                elif proj_type == "F":
                    P = sp.vstack(
                        [sp.hstack([Eye, Zero, Zero]), sp.hstack([Zero, Zero, Eye])]
                    )
                    # print(P.todense())
            else:
                P = sp.eye(self.nC * self.dim)

            if not inv_mat and not inv_prop:
                dMdprop = Av.T * P * V
            elif inv_mat and inv_prop:
                dMdprop = (
                    sdiag(MI.diagonal() ** 2) * Av.T * P * V * sdiag(1.0 / prop ** 2)
                )
            elif inv_prop:
                dMdprop = Av.T * P * V * sdiag(-1.0 / prop ** 2)
            elif inv_mat:
                dMdprop = sdiag(-MI.diagonal() ** 2) * Av.T * P * V

        if dMdprop is not None:

            def innerProductDeriv(v=None):
                if v is None:
                    warnings.warn(
                        "Depreciation Warning: TensorMesh.innerProductDeriv."
                        " You should be supplying a vector. "
                        "Use: sdiag(u)*dMdprop",
                        FutureWarning,
                    )
                    return dMdprop
                return sdiag(v) * dMdprop

            return innerProductDeriv
        else:
            return None

    # DEPRECATED
    @property
    def hx(self):
        """Width of cells in the x direction

        Returns
        -------
        numpy.ndarray

        .. deprecated:: 0.5.0
          `hx` will be removed in discretize 1.0.0 to reduce namespace clutter,
          please use `mesh.h[0]`.
        """
        warnings.warn(
            "hx has been deprecated, please access as mesh.h[0]", FutureWarning
        )
        return self.h[0]

    @property
    def hy(self):
        """Width of cells in the y direction

        Returns
        -------
        numpy.ndarray or None

        .. deprecated:: 0.5.0
          `hy` will be removed in discretize 1.0.0 to reduce namespace clutter,
          please use `mesh.h[1]`.
        """
        warnings.warn(
            "hy has been deprecated, please access as mesh.h[1]", FutureWarning
        )
        return None if self.dim < 2 else self.h[1]

    @property
    def hz(self):
        """Width of cells in the z direction

        Returns
        -------
        numpy.ndarray or None

        .. deprecated:: 0.5.0
          `hz` will be removed in discretize 1.0.0 to reduce namespace clutter,
          please use `mesh.h[2]`.
        """
        warnings.warn(
            "hz has been deprecated, please access as mesh.h[2]", FutureWarning
        )
        return None if self.dim < 3 else self.h[2]

    vectorNx = deprecate_property("grid_nodes_x", "vectorNx", removal_version="1.0.0")
    vectorNy = deprecate_property("grid_nodes_y", "vectorNy", removal_version="1.0.0")
    vectorNz = deprecate_property("grid_nodes_z", "vectorNz", removal_version="1.0.0")
    vectorCCx = deprecate_property(
        "grid_cell_centers_x", "vectorCCx", removal_version="1.0.0"
    )
    vectorCCy = deprecate_property(
        "grid_cell_centers_y", "vectorCCy", removal_version="1.0.0"
    )
    vectorCCz = deprecate_property(
        "grid_cell_centers_z", "vectorCCz", removal_version="1.0.0"
    )
    getInterpolationMat = deprecate_method(
        "get_interpolation_matrix", "getInterpolationMat", removal_version="1.0.0"
    )
    isInside = deprecate_method("is_inside", "isInside", removal_version="1.0.0")
    getTensor = deprecate_method("get_tensor", "getTensor", removal_version="1.0.0")
