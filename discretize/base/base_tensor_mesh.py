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
    unpack_widths,
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
            "gridCC": "cell_centers",
            "gridN": "nodes",
            "gridFx": "faces_x",
            "gridFy": "faces_y",
            "gridFz": "faces_z",
            "gridEx": "edges_x",
            "gridEy": "edges_y",
            "gridEz": "edges_z",
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
        if not isinstance(h_in, (list, tuple)):
            raise TypeError("h_in must be a list, not {}".format(type(h_in)))
        if len(h_in) not in [1, 2, 3]:
            raise ValueError(
                "h_in must be of dimension 1, 2, or 3 not {}".format(len(h_in))
            )

        # build h
        h = list(range(len(h_in)))
        for i, h_i in enumerate(h_in):
            if is_scalar(h_i) and not isinstance(h_i, np.ndarray):
                # This gives you something over the unit cube.
                h_i = self._unitDimensions[i] * np.ones(int(h_i)) / int(h_i)
            elif isinstance(h_i, (list, tuple)):
                h_i = unpack_widths(h_i)
            if not isinstance(h_i, np.ndarray):
                raise TypeError("h[{0:d}] is not a numpy array.".format(i))
            if len(h_i.shape) != 1:
                raise ValueError("h[{0:d}] must be a 1D numpy array.".format(i))
            h[i] = h_i[:]  # make a copy.

        # Origin of the mesh
        x0 = np.zeros(len(h))

        if x0_in is not None:
            if len(h) != len(x0_in):
                raise ValueError("Dimension mismatch. x0 != len(h)")
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
            if np.any(n != np.array([x.size for x in h])):
                raise ValueError("Dimension mismatch. The provided n doesn't h")
        else:
            n = np.array([x.size for x in h])

        super(BaseTensorMesh, self).__init__(n, x0=x0, **kwargs)

        # Ensure h contains 1D vectors
        self.h = [mkvc(x.astype(float)) for x in h]

    @property
    def nodes_x(self):
        """Nodal grid vector (1D) in the x direction."""
        return np.r_[self.x0[0], self.h[0]].cumsum()

    @property
    def nodes_y(self):
        """Nodal grid vector (1D) in the y direction."""
        return None if self.dim < 2 else np.r_[self.x0[1], self.h[1]].cumsum()

    @property
    def nodes_z(self):
        """Nodal grid vector (1D) in the z direction."""
        return None if self.dim < 3 else np.r_[self.x0[2], self.h[2]].cumsum()

    @property
    def cell_centers_x(self):
        """Cell-centered grid vector (1D) in the x direction."""
        nodes = self.nodes_x
        return (nodes[1:] + nodes[:-1]) / 2

    @property
    def cell_centers_y(self):
        """Cell-centered grid vector (1D) in the y direction."""
        if self.dim < 2:
            return None
        nodes = self.nodes_y
        return (nodes[1:] + nodes[:-1]) / 2

    @property
    def cell_centers_z(self):
        """Cell-centered grid vector (1D) in the z direction."""
        if self.dim < 3:
            return None
        nodes = self.nodes_z
        return (nodes[1:] + nodes[:-1]) / 2

    @property
    def cell_centers(self):
        """Cell-centered grid."""
        return self._getTensorGrid("CC")

    @property
    def nodes(self):
        """Nodal grid."""
        return self._getTensorGrid("N")

    @property
    def h_gridded(self):
        """
        Returns an (nC, dim) numpy array with the widths of all cells in order
        """
        if self.dim == 1:
            return self.h[0][:, None]
        return ndgrid(*self.h)

    @property
    def faces_x(self):
        """Face staggered grid in the x direction."""
        if self.nFx == 0:
            return
        return self._getTensorGrid("Fx")

    @property
    def faces_y(self):
        """Face staggered grid in the y direction."""
        if self.nFy == 0 or self.dim < 2:
            return
        return self._getTensorGrid("Fy")

    @property
    def faces_z(self):
        """Face staggered grid in the z direction."""
        if self.nFz == 0 or self.dim < 3:
            return
        return self._getTensorGrid("Fz")

    @property
    def edges_x(self):
        """Edge staggered grid in the x direction."""
        if self.nEx == 0:
            return
        return self._getTensorGrid("Ex")

    @property
    def edges_y(self):
        """Edge staggered grid in the y direction."""
        if self.nEy == 0 or self.dim < 2:
            return
        return self._getTensorGrid("Ey")

    @property
    def edges_z(self):
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
                self.nodes_x,
                self.cell_centers_y,
                self.cell_centers_z,
            ]
        elif key == "Fy":
            ten = [
                self.cell_centers_x,
                self.nodes_y,
                self.cell_centers_z,
            ]
        elif key == "Fz":
            ten = [
                self.cell_centers_x,
                self.cell_centers_y,
                self.nodes_z,
            ]
        elif key == "Ex":
            ten = [self.cell_centers_x, self.nodes_y, self.nodes_z]
        elif key == "Ey":
            ten = [self.nodes_x, self.cell_centers_y, self.nodes_z]
        elif key == "Ez":
            ten = [self.nodes_x, self.nodes_y, self.cell_centers_z]
        elif key == "CC":
            ten = [
                self.cell_centers_x,
                self.cell_centers_y,
                self.cell_centers_z,
            ]
        elif key == "N":
            ten = [self.nodes_x, self.nodes_y, self.nodes_z]

        return [t for t in ten if t is not None]

    # --------------- Methods ---------------------

    def is_inside(self, pts, location_type="N", **kwargs):
        """
        Determines if a set of points are inside a mesh.

        :param numpy.ndarray pts: Location of points to test
        :rtype: numpy.ndarray
        :return: inside, numpy array of booleans
        """
        if "locType" in kwargs:
            warnings.warn(
                "The locType keyword argument has been deprecated, please use location_type. "
                "This will be removed in discretize 1.0.0",
                FutureWarning,
            )
            location_type = kwargs["locType"]
        pts = as_array_n_by_dim(pts, self.dim)

        tensors = self.get_tensor(location_type)

        if location_type == "N" and self._meshType == "CYL":
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

    def _getInterpolationMat(self, loc, location_type="CC", zeros_outside=False):
        """Produces interpolation matrix

        Parameters
        ----------
        loc : numpy.ndarray
            Location of points to interpolate to

        location_type: stc
            What to interpolate

            location_type can be::

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
            if not np.all(self.is_inside(loc)):
                raise ValueError("Points outside of mesh")
        else:
            indZeros = np.logical_not(self.is_inside(loc))
            loc[indZeros, :] = np.array([v.mean() for v in self.get_tensor("CC")])

        if location_type in ["Fx", "Fy", "Fz", "Ex", "Ey", "Ez"]:
            ind = {"x": 0, "y": 1, "z": 2}[location_type[1]]
            if self.dim < ind:
                raise ValueError("mesh is not high enough dimension.")
            if "F" in location_type:
                items = (self.nFx, self.nFy, self.nFz)[: self.dim]
            else:
                items = (self.nEx, self.nEy, self.nEz)[: self.dim]
            components = [spzeros(loc.shape[0], n) for n in items]
            components[ind] = interpolation_matrix(loc, *self.get_tensor(location_type))
            # remove any zero blocks (hstack complains)
            components = [comp for comp in components if comp.shape[1] > 0]
            Q = sp.hstack(components)

        elif location_type in ["CC", "N"]:
            Q = interpolation_matrix(loc, *self.get_tensor(location_type))

        elif location_type in ["CCVx", "CCVy", "CCVz"]:
            Q = interpolation_matrix(loc, *self.get_tensor("CC"))
            Z = spzeros(loc.shape[0], self.nC)
            if location_type == "CCVx":
                Q = sp.hstack([Q, Z, Z])
            elif location_type == "CCVy":
                Q = sp.hstack([Z, Q, Z])
            elif location_type == "CCVz":
                Q = sp.hstack([Z, Z, Q])

        else:
            raise NotImplementedError(
                "getInterpolationMat: location_type=="
                + location_type
                + " and mesh.dim=="
                + str(self.dim)
            )

        if zeros_outside:
            Q[indZeros, :] = 0

        return Q.tocsr()

    def get_interpolation_matrix(
        self, loc, location_type="CC", zeros_outside=False, **kwargs
    ):
        """Produces linear interpolation matrix

        Parameters
        ----------
        loc : numpy.ndarray
            Location of points to interpolate to

        location_type : str
            What to interpolate (see below)

            location_type can be::

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
        return self._getInterpolationMat(loc, location_type, zeros_outside)

    def _fastInnerProduct(self, projection_type, model=None, inverse_property=False, inverse_matrix=False):
        """Fast version of getFaceInnerProduct.
            This does not handle the case of a full tensor property.

        Parameters
        ----------

        model : numpy.array
            material property (tensor properties are possible) at each cell center (nC, (1, 3, or 6))

        projection_type : str
            'E' or 'F'

        returnP : bool
            returns the projection matrices

        inverse_property : bool
            inverts the material property

        inverse_matrix : bool
            inverts the matrix

        Returns
        -------
        scipy.sparse.csr_matrix
            M, the inner product matrix (nF, nF)

        """
        if projection_type not in ["F", "E"]:
            raise ValueError("projection_type must be 'F' for faces or 'E' for edges")

        if model is None:
            property = np.ones(self.nC)

        if inverse_property:
            model = 1.0 / model

        if is_scalar(model):
            model = model * np.ones(self.nC)

        # number of elements we are averaging (equals dim for regular
        # meshes, but for cyl, where we use symmetry, it is 1 for edge
        # variables and 2 for face variables)
        if self._meshType == "CYL":
            shape = getattr(self, "vn" + projection_type)
            n_elements = sum([1 if x != 0 else 0 for x in shape])
        else:
            n_elements = self.dim

        # Isotropic? or anisotropic?
        if model.size == self.nC:
            Av = getattr(self, "ave" + projection_type + "2CC")
            Vprop = self.cell_volumes * mkvc(model)
            M = n_elements * sdiag(Av.T * Vprop)

        elif model.size == self.nC * self.dim:
            Av = getattr(self, "ave" + projection_type + "2CCV")

            # if cyl, then only certain components are relevant due to symmetry
            # for faces, x, z matters, for edges, y (which is theta) matters
            if self._meshType == "CYL":
                if projection_type == "E":
                    model = model[:, 1]  # this is the action of a projection mat
                elif projection_type == "F":
                    model = model[:, [0, 2]]

            V = sp.kron(sp.identity(n_elements), sdiag(self.cell_volumes))
            M = sdiag(Av.T * V * mkvc(model))
        else:
            return None

        if inverse_matrix:
            return sdinv(M)
        else:
            return M

    def _fastInnerProductDeriv(self, projection_type, model, inverse_property=False, inverse_matrix=False):
        """

        Parameters
        ----------

        projection_type : str
            'E' or 'F'

        tensorType : TensorType
            type of the tensor

        inverse_property : bool
            inverts the material property

        inverse_matrix : bool
            inverts the matrix


        Returns
        -------
        function
            dMdmu, the derivative of the inner product matrix

        """

        if projection_type not in ["F", "E"]:
            raise ValueError("projection_type must be 'F' for faces or 'E' for edges")

        tensorType = TensorType(self, model)

        dMdprop = None

        if inverse_matrix or inverse_property:
            MI = self._fastInnerProduct(
                projection_type, model, inverse_property=inverse_property, inverse_matrix=inverse_matrix
            )

        # number of elements we are averaging (equals dim for regular
        # meshes, but for cyl, where we use symmetry, it is 1 for edge
        # variables and 2 for face variables)
        if self._meshType == "CYL":
            shape = getattr(self, "vn" + projection_type)
            n_elements = sum([1 if x != 0 else 0 for x in shape])
        else:
            n_elements = self.dim

        if tensorType == 0:  # isotropic, constant
            Av = getattr(self, "ave" + projection_type + "2CC")
            V = sdiag(self.cell_volumes)
            ones = sp.csr_matrix(
                (np.ones(self.nC), (range(self.nC), np.zeros(self.nC))),
                shape=(self.nC, 1),
            )
            if not inverse_matrix and not inverse_property:
                dMdprop = n_elements * Av.T * V * ones
            elif inverse_matrix and inverse_property:
                dMdprop = n_elements * (
                    sdiag(MI.diagonal() ** 2) * Av.T * V * ones * sdiag(1.0 / model ** 2)
                )
            elif inverse_property:
                dMdprop = n_elements * Av.T * V * sdiag(-1.0 / model ** 2)
            elif inverse_matrix:
                dMdprop = n_elements * (sdiag(-MI.diagonal() ** 2) * Av.T * V)

        elif tensorType == 1:  # isotropic, variable in space
            Av = getattr(self, "ave" + projection_type + "2CC")
            V = sdiag(self.cell_volumes)
            if not inverse_matrix and not inverse_property:
                dMdprop = n_elements * Av.T * V
            elif inverse_matrix and inverse_property:
                dMdprop = n_elements * (
                    sdiag(MI.diagonal() ** 2) * Av.T * V * sdiag(1.0 / model ** 2)
                )
            elif inverse_property:
                dMdprop = n_elements * Av.T * V * sdiag(-1.0 / model ** 2)
            elif inverse_matrix:
                dMdprop = n_elements * (sdiag(-MI.diagonal() ** 2) * Av.T * V)

        elif tensorType == 2:  # anisotropic
            Av = getattr(self, "ave" + projection_type + "2CCV")
            V = sp.kron(sp.identity(self.dim), sdiag(self.cell_volumes))

            if self._meshType == "CYL":
                Zero = sp.csr_matrix((self.nC, self.nC))
                Eye = sp.eye(self.nC)
                if projection_type == "E":
                    P = sp.hstack([Zero, Eye, Zero])
                    # print(P.todense())
                elif projection_type == "F":
                    P = sp.vstack(
                        [sp.hstack([Eye, Zero, Zero]), sp.hstack([Zero, Zero, Eye])]
                    )
                    # print(P.todense())
            else:
                P = sp.eye(self.nC * self.dim)

            if not inverse_matrix and not inverse_property:
                dMdprop = Av.T * P * V
            elif inverse_matrix and inverse_property:
                dMdprop = (
                    sdiag(MI.diagonal() ** 2) * Av.T * P * V * sdiag(1.0 / model ** 2)
                )
            elif inverse_property:
                dMdprop = Av.T * P * V * sdiag(-1.0 / model ** 2)
            elif inverse_matrix:
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

    vectorNx = deprecate_property("nodes_x", "vectorNx", removal_version="1.0.0")
    vectorNy = deprecate_property("nodes_y", "vectorNy", removal_version="1.0.0")
    vectorNz = deprecate_property("nodes_z", "vectorNz", removal_version="1.0.0")
    vectorCCx = deprecate_property(
        "cell_centers_x", "vectorCCx", removal_version="1.0.0"
    )
    vectorCCy = deprecate_property(
        "cell_centers_y", "vectorCCy", removal_version="1.0.0"
    )
    vectorCCz = deprecate_property(
        "cell_centers_z", "vectorCCz", removal_version="1.0.0"
    )
    getInterpolationMat = deprecate_method(
        "get_interpolation_matrix", "getInterpolationMat", removal_version="1.0.0"
    )
    isInside = deprecate_method("is_inside", "isInside", removal_version="1.0.0")
    getTensor = deprecate_method("get_tensor", "getTensor", removal_version="1.0.0")
