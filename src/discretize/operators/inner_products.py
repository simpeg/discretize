"""Construct inner product operators for tensor like meshes."""
from scipy import sparse as sp
from discretize.base import BaseMesh
from discretize.utils import (
    sub2ind,
    sdiag,
    inverse_property_tensor,
    TensorType,
    make_property_tensor,
    ndgrid,
    inverse_2x2_block_diagonal,
    get_subarray,
    inverse_3x3_block_diagonal,
    spzeros,
    sdinv,
    mkvc,
    is_scalar,
)
import numpy as np


class InnerProducts(BaseMesh):
    """Class for constructing inner product matrices.

    ``InnerProducts`` is a mixin class for constructing inner product matrices,
    their inverses and their derivatives with respect to model parameters.
    The ``InnerProducts`` class is inherited by all ``discretize`` mesh classes.
    In practice, we don't create instances of the ``InnerProducts`` class in
    order to construct inner product matrices, their inverses or their derivatives.
    These quantities are instead constructed from instances of ``discretize``
    meshes using the appropriate method.
    """

    def get_face_inner_product(  # NOQA D102
        self,
        model=None,
        invert_model=False,
        invert_matrix=False,
        do_fast=True,
        **kwargs
    ):
        # Inherited documentation from discretize.base.BaseMesh
        if "invProp" in kwargs:
            raise TypeError(
                "The invProp keyword argument has been removed, please use invert_model. "
                "This will be removed in discretize 1.0.0",
            )
        if "invMat" in kwargs:
            raise TypeError(
                "The invMat keyword argument has been removed, please use invert_matrix. "
                "This will be removed in discretize 1.0.0",
            )
        if "doFast" in kwargs:
            raise TypeError(
                "The doFast keyword argument has been removed, please use do_fast. "
                "This will be removed in discretize 1.0.0",
            )
            do_fast = kwargs["doFast"]

        return self._getInnerProduct(
            "F",
            model=model,
            invert_model=invert_model,
            invert_matrix=invert_matrix,
            do_fast=do_fast,
        )

    def get_edge_inner_product(  # NOQA D102
        self,
        model=None,
        invert_model=False,
        invert_matrix=False,
        do_fast=True,
        **kwargs
    ):
        # Inherited documentation from discretize.base.BaseMesh
        if "invProp" in kwargs:
            raise TypeError(
                "The invProp keyword argument has been removed, please use invert_model. "
                "This will be removed in discretize 1.0.0",
            )
        if "invMat" in kwargs:
            raise TypeError(
                "The invMat keyword argument has been removed, please use invert_matrix. "
                "This will be removed in discretize 1.0.0",
            )
        if "doFast" in kwargs:
            raise TypeError(
                "The doFast keyword argument has been removed, please use do_fast. "
                "This will be removed in discretize 1.0.0",
            )
        return self._getInnerProduct(
            "E",
            model=model,
            invert_model=invert_model,
            invert_matrix=invert_matrix,
            do_fast=do_fast,
        )

    def get_edge_inner_product_surface(  # NOQA D102
        self, model=None, invert_model=False, invert_matrix=False, **kwargs
    ):
        # Inherited documentation from discretize.base.BaseMesh
        if model is None:
            model = np.ones(self.nF)

        if invert_model:
            model = 1.0 / model

        if is_scalar(model):
            model = model * np.ones(self.nF)
        # COULD ADD THIS CASE IF DESIRED
        # elif len(model) == self.dim:
        #     model = np.r_[
        #         [model[ii]*self.vnF[ii] for ii in range(0, self.dim)]
        #     ]

        # Isotropic case only
        if model.size != self.nF:
            raise ValueError(
                "Unexpected shape of tensor: {}".format(model.shape),
                "Must be scalar or have length equal to total number of faces.",
            )

        # number of elements we are averaging (equals dim for regular
        # meshes, but for cyl, where we use symmetry, it is 1 for edge
        # variables and 2 for face variables)
        if self._meshType == "CYL":
            shape = self.vnE
            if self.is_symmetric:
                n_elements = 1
            else:
                n_elements = sum([1 if x != 0 else 0 for x in shape]) - 1
        else:
            n_elements = self.dim - 1

        Aprop = self.face_areas * mkvc(model)
        Av = self.average_edge_to_face
        M = n_elements * sdiag(Av.T * Aprop)

        if invert_matrix:
            return sdinv(M)
        else:
            return M

    def _getInnerProduct(
        self,
        projection_type,
        model=None,
        invert_model=False,
        invert_matrix=False,
        do_fast=True,
        **kwargs
    ):
        """Get the inner product matrix.

        Parameters
        ----------
        str : projection_type
            'F' for faces 'E' for edges
        numpy.ndarray : model
            material property (tensor properties are possible) at each cell center (nC, (1, 3, or 6))
        bool : invert_model
            inverts the material property
        bool : invert_matrix
            inverts the matrix
        bool : do_fast
            do a faster implementation if available.

        Returns
        -------
        scipy.sparse.csr_matrix
            M, the inner product matrix (nE, nE)
        """
        if "invProp" in kwargs:
            raise TypeError(
                "The invProp keyword argument has been removed, please use invert_model. "
                "This will be removed in discretize 1.0.0",
            )
        if "invMat" in kwargs:
            raise TypeError(
                "The invMat keyword argument has been removed, please use invert_matrix. "
                "This will be removed in discretize 1.0.0",
            )
        if "doFast" in kwargs:
            raise TypeError(
                "The doFast keyword argument has been removed, please use do_fast. "
                "This will be removed in discretize 1.0.0",
            )
        if projection_type not in ["F", "E"]:
            raise TypeError("projection_type must be 'F' for faces or 'E' for edges")

        fast = None
        if hasattr(self, "_fastInnerProduct") and do_fast:
            fast = self._fastInnerProduct(
                projection_type,
                model=model,
                invert_model=invert_model,
                invert_matrix=invert_matrix,
            )
        if fast is not None:
            return fast

        if invert_model:
            model = inverse_property_tensor(self, model)

        tensorType = TensorType(self, model)

        Mu = make_property_tensor(self, model)
        Ps = self._getInnerProductProjectionMatrices(projection_type, tensorType)
        A = np.sum([P.T * Mu * P for P in Ps])

        if invert_matrix and tensorType < 3:
            A = sdinv(A)
        elif invert_matrix and tensorType == 3:
            raise Exception("Solver needed to invert A.")

        return A

    def _getInnerProductProjectionMatrices(self, projection_type, tensorType):
        """Get the inner product projection matrices.

        Parameters
        ----------
        projection_type : str
            'F' for faces 'E' for edges
        tensorType : TensorType
            type of the tensor: TensorType(mesh, sigma)

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        if not isinstance(tensorType, TensorType):
            raise TypeError("tensorType must be an instance of TensorType.")
        if projection_type not in ["F", "E"]:
            raise TypeError("projection_type must be 'F' for faces or 'E' for edges")

        d = self.dim
        # We will multiply by sqrt on each side to keep symmetry
        V = sp.kron(sp.identity(d), sdiag(np.sqrt((2 ** (-d)) * self.cell_volumes)))

        nodes = ["000", "100", "010", "110", "001", "101", "011", "111"][: 2**d]

        if projection_type == "F":
            locs = {
                "000": [("fXm",), ("fXm", "fYm"), ("fXm", "fYm", "fZm")],
                "100": [("fXp",), ("fXp", "fYm"), ("fXp", "fYm", "fZm")],
                "010": [None, ("fXm", "fYp"), ("fXm", "fYp", "fZm")],
                "110": [None, ("fXp", "fYp"), ("fXp", "fYp", "fZm")],
                "001": [None, None, ("fXm", "fYm", "fZp")],
                "101": [None, None, ("fXp", "fYm", "fZp")],
                "011": [None, None, ("fXm", "fYp", "fZp")],
                "111": [None, None, ("fXp", "fYp", "fZp")],
            }
            proj = getattr(self, "_getFaceP" + ("x" * d))()

        elif projection_type == "E":
            locs = {
                "000": [("eX0",), ("eX0", "eY0"), ("eX0", "eY0", "eZ0")],
                "100": [("eX0",), ("eX0", "eY1"), ("eX0", "eY1", "eZ1")],
                "010": [None, ("eX1", "eY0"), ("eX1", "eY0", "eZ2")],
                "110": [None, ("eX1", "eY1"), ("eX1", "eY1", "eZ3")],
                "001": [None, None, ("eX2", "eY2", "eZ0")],
                "101": [None, None, ("eX2", "eY3", "eZ1")],
                "011": [None, None, ("eX3", "eY2", "eZ2")],
                "111": [None, None, ("eX3", "eY3", "eZ3")],
            }
            proj = getattr(self, "_getEdgeP" + ("x" * d))()

        return [V * proj(*locs[node][d - 1]) for node in nodes]

    def get_face_inner_product_deriv(  # NOQA D102
        self, model, do_fast=True, invert_model=False, invert_matrix=False, **kwargs
    ):
        # Inherited documentation from discretize.base.BaseMesh
        if "invProp" in kwargs:
            raise TypeError(
                "The invProp keyword argument has been removed, please use invert_model. "
                "This will be removed in discretize 1.0.0",
            )
        if "invMat" in kwargs:
            raise TypeError(
                "The invMat keyword argument has been removed, please use invert_matrix. "
                "This will be removed in discretize 1.0.0",
            )
        if "doFast" in kwargs:
            raise TypeError(
                "The doFast keyword argument has been removed, please use do_fast. "
                "This will be removed in discretize 1.0.0",
            )
        return self._getInnerProductDeriv(
            model,
            "F",
            do_fast=do_fast,
            invert_model=invert_model,
            invert_matrix=invert_matrix,
        )

    def get_edge_inner_product_deriv(  # NOQA D102
        self, model, do_fast=True, invert_model=False, invert_matrix=False, **kwargs
    ):
        # Inherited documentation from discretize.base.BaseMesh
        if "invProp" in kwargs:
            raise TypeError(
                "The invProp keyword argument has been removed, please use invert_model. "
                "This will be removed in discretize 1.0.0",
            )
        if "invMat" in kwargs:
            raise TypeError(
                "The invMat keyword argument has been removed, please use invert_matrix. "
                "This will be removed in discretize 1.0.0",
            )
        if "doFast" in kwargs:
            raise TypeError(
                "The doFast keyword argument has been removed, please use do_fast. "
                "This will be removed in discretize 1.0.0",
            )
        return self._getInnerProductDeriv(
            model,
            "E",
            do_fast=do_fast,
            invert_model=invert_model,
            invert_matrix=invert_matrix,
        )

    def get_edge_inner_product_surface_deriv(  # NOQA D102
        self, model, invert_model=False, invert_matrix=False, **kwargs
    ):
        # Inherited documentation from discretize.base.BaseMesh
        if model is None:
            tensorType = -1
        elif is_scalar(model):
            tensorType = 0
        elif model.size == self.nF:
            tensorType = 1
        else:
            raise ValueError(
                "Unexpected shape of tensor: {}".format(model.shape),
                "Must be scalar or have length equal to total number of faces.",
            )

        dMdprop = None

        if invert_matrix or invert_model:
            MI = self.get_edge_inner_product_surface(
                model,
                invert_model=invert_model,
                invert_matrix=invert_matrix,
            )

        # number of elements we are averaging (equals dim for regular
        # meshes, but for cyl, where we use symmetry, it is 1 for edge
        # variables and 2 for face variables)
        if self._meshType == "CYL":
            shape = self.vnE
            if self.is_symmetric:
                n_elements = 1
            else:
                n_elements = sum([1 if x != 0 else 0 for x in shape]) - 1
        else:
            n_elements = self.dim - 1

        A = sdiag(self.face_areas)
        Av = self.average_edge_to_face

        if tensorType == 0:  # isotropic, constant
            ones = sp.csr_matrix(
                (np.ones(self.nF), (range(self.nF), np.zeros(self.nF))),
                shape=(self.nF, 1),
            )
            if not invert_matrix and not invert_model:
                dMdprop = n_elements * Av.T * A * ones
            elif invert_matrix and invert_model:
                dMdprop = n_elements * (
                    sdiag(MI.diagonal() ** 2)
                    * Av.T
                    * A
                    * ones
                    * sdiag(1.0 / model**2)
                )
            elif invert_model:
                dMdprop = n_elements * Av.T * A * sdiag(-1.0 / model**2)
            elif invert_matrix:
                dMdprop = n_elements * (sdiag(-MI.diagonal() ** 2) * Av.T * A)

        elif tensorType == 1:  # isotropic, variable in space
            if not invert_matrix and not invert_model:
                dMdprop = n_elements * Av.T * A
            elif invert_matrix and invert_model:
                dMdprop = n_elements * (
                    sdiag(MI.diagonal() ** 2) * Av.T * A * sdiag(1.0 / model**2)
                )
            elif invert_model:
                dMdprop = n_elements * Av.T * A * sdiag(-1.0 / model**2)
            elif invert_matrix:
                dMdprop = n_elements * (sdiag(-MI.diagonal() ** 2) * Av.T * A)

        if dMdprop is not None:

            def innerProductDeriv(v):
                return sdiag(v) * dMdprop

            return innerProductDeriv
        else:
            return None

    def _getInnerProductDeriv(
        self,
        model,
        projection_type,
        do_fast=True,
        invert_model=False,
        invert_matrix=False,
    ):
        """Get the inner product projection derivative function.

        Parameters
        ----------
        model : numpy.ndarray
            material property (tensor properties are possible) at each cell center (nC, (1, 3, or 6))
        projection_type : str
            'F' for faces 'E' for edges
        do_fast : bool
            do a faster implementation if available.
        invert_model : bool
            inverts the material property
        invert_matrix : bool
            inverts the matrix

        Returns
        -------
        callable
            dMdm, the derivative of the inner product matrix (nE, nC*nA)
        """
        fast = None
        if hasattr(self, "_fastInnerProductDeriv") and do_fast:
            fast = self._fastInnerProductDeriv(
                projection_type,
                model,
                invert_model=invert_model,
                invert_matrix=invert_matrix,
            )
        if fast is not None:
            return fast

        if invert_model or invert_matrix:
            raise NotImplementedError(
                "inverting the property or the matrix is not yet implemented for this mesh/tensorType. You should write it!"
            )

        tensorType = TensorType(self, model)
        P = self._getInnerProductProjectionMatrices(
            projection_type, tensorType=tensorType
        )

        def innerProductDeriv(v):
            return self._getInnerProductDerivFunction(tensorType, P, projection_type, v)

        return innerProductDeriv

    def _getInnerProductDerivFunction(self, tensorType, P, projection_type, v):
        """Get the inner product projection derivative function depending on the tensor type.

        Parameters
        ----------
        model : numpy.ndarray
            material property (tensor properties are possible) at each cell center (nC, (1, 3, or 6))
        v : numpy.ndarray
            vector to multiply (required in the general implementation)
        P : list
            list of projection matrices
        projection_type : str
            'F' for faces 'E' for edges

        Returns
        -------
        scipy.sparse.csr_matrix
            dMdm, the derivative of the inner product matrix (n, nC*nA)
        """
        if projection_type not in ["F", "E"]:
            raise TypeError("projection_type must be 'F' for faces or 'E' for edges")

        n = getattr(self, "n" + projection_type)

        if tensorType == -1:
            return None

        if v is None:
            raise Exception("v must be supplied for this implementation.")

        d = self.dim
        Z = spzeros(self.nC, self.nC)

        if tensorType == 0:
            dMdm = spzeros(n, 1)
            for p in P:
                dMdm = dMdm + sp.csr_matrix(
                    (p.T * (p * v), (range(n), np.zeros(n))), shape=(n, 1)
                )
        if d == 1:
            if tensorType == 1:
                dMdm = spzeros(n, self.nC)
                for p in P:
                    dMdm = dMdm + p.T * sdiag(p * v)
        elif d == 2:
            if tensorType == 1:
                dMdm = spzeros(n, self.nC)
                for p in P:
                    Y = p * v
                    y1 = Y[: self.nC]
                    y2 = Y[self.nC :]
                    dMdm = dMdm + p.T * sp.vstack((sdiag(y1), sdiag(y2)))
            elif tensorType == 2:
                dMdms = [spzeros(n, self.nC) for _ in range(2)]
                for p in P:
                    Y = p * v
                    y1 = Y[: self.nC]
                    y2 = Y[self.nC :]
                    dMdms[0] = dMdms[0] + p.T * sp.vstack((sdiag(y1), Z))
                    dMdms[1] = dMdms[1] + p.T * sp.vstack((Z, sdiag(y2)))
                dMdm = sp.hstack(dMdms)
            elif tensorType == 3:
                dMdms = [spzeros(n, self.nC) for _ in range(3)]
                for p in P:
                    Y = p * v
                    y1 = Y[: self.nC]
                    y2 = Y[self.nC :]
                    dMdms[0] = dMdms[0] + p.T * sp.vstack((sdiag(y1), Z))
                    dMdms[1] = dMdms[1] + p.T * sp.vstack((Z, sdiag(y2)))
                    dMdms[2] = dMdms[2] + p.T * sp.vstack((sdiag(y2), sdiag(y1)))
                dMdm = sp.hstack(dMdms)
        elif d == 3:
            if tensorType == 1:
                dMdm = spzeros(n, self.nC)
                for p in P:
                    Y = p * v
                    y1 = Y[: self.nC]
                    y2 = Y[self.nC : self.nC * 2]
                    y3 = Y[self.nC * 2 :]
                    dMdm = dMdm + p.T * sp.vstack((sdiag(y1), sdiag(y2), sdiag(y3)))
            elif tensorType == 2:
                dMdms = [spzeros(n, self.nC) for _ in range(3)]
                for p in P:
                    Y = p * v
                    y1 = Y[: self.nC]
                    y2 = Y[self.nC : self.nC * 2]
                    y3 = Y[self.nC * 2 :]
                    dMdms[0] = dMdms[0] + p.T * sp.vstack((sdiag(y1), Z, Z))
                    dMdms[1] = dMdms[1] + p.T * sp.vstack((Z, sdiag(y2), Z))
                    dMdms[2] = dMdms[2] + p.T * sp.vstack((Z, Z, sdiag(y3)))
                dMdm = sp.hstack(dMdms)
            elif tensorType == 3:
                dMdms = [spzeros(n, self.nC) for _ in range(6)]
                for p in P:
                    Y = p * v
                    y1 = Y[: self.nC]
                    y2 = Y[self.nC : self.nC * 2]
                    y3 = Y[self.nC * 2 :]
                    dMdms[0] = dMdms[0] + p.T * sp.vstack((sdiag(y1), Z, Z))
                    dMdms[1] = dMdms[1] + p.T * sp.vstack((Z, sdiag(y2), Z))
                    dMdms[2] = dMdms[2] + p.T * sp.vstack((Z, Z, sdiag(y3)))
                    dMdms[3] = dMdms[3] + p.T * sp.vstack((sdiag(y2), sdiag(y1), Z))
                    dMdms[4] = dMdms[4] + p.T * sp.vstack((sdiag(y3), Z, sdiag(y1)))
                    dMdms[5] = dMdms[5] + p.T * sp.vstack((Z, sdiag(y3), sdiag(y2)))
                dMdm = sp.hstack(dMdms)

        return dMdm

    # ------------------------ Geometries ------------------------------
    #
    #
    #         node(i,j,k+1) ------ edge2(i,j,k+1) ----- node(i,j+1,k+1)
    #              /                                    /
    #             /                                    / |
    #         edge3(i,j,k)     face1(i,j,k)        edge3(i,j+1,k)
    #           /                                    /   |
    #          /                                    /    |
    #    node(i,j,k) ------ edge2(i,j,k) ----- node(i,j+1,k)
    #         |                                     |    |
    #         |                                     |   node(i+1,j+1,k+1)
    #         |                                     |    /
    #    edge1(i,j,k)      face3(i,j,k)        edge1(i,j+1,k)
    #         |                                     |  /
    #         |                                     | /
    #         |                                     |/
    #    node(i+1,j,k) ------ edge2(i+1,j,k) ----- node(i+1,j+1,k)

    def _getFacePx(M):
        """Return a function for creating face projection matrices in 1D."""
        ii = np.arange(M.shape_cells[0])

        def Px(xFace):
            # xFace is 'fXp' or 'fXm'
            posFx = 0 if xFace == "fXm" else 1
            IND = ii + posFx
            PX = sp.csr_matrix((np.ones(M.nC), (range(M.nC), IND)), shape=(M.nC, M.nF))
            return PX

        return Px

    def _getFacePxx(M):
        """Return a function for creating face projection matrices in 2D."""
        # returns a function for creating projection matrices
        #
        # Mats takes you from faces a subset of all faces on only the
        # faces that you ask for.
        #
        # These are centered around a single nodes.
        #
        # For example, if this was your entire mesh:
        #
        #                 f3(Yp)
        #           2_______________3
        #           |               |
        #           |               |
        #           |               |
        #   f0(Xm)  |       x       |  f1(Xp)
        #           |               |
        #           |               |
        #           |_______________|
        #           0               1
        #                 f2(Ym)
        #
        # Pxx('fXm','fYm') = | 1, 0, 0, 0 |
        #                    | 0, 0, 1, 0 |
        #
        # Pxx('fXp','fYm') = | 0, 1, 0, 0 |
        #                    | 0, 0, 1, 0 |
        i, j = np.arange(M.shape_cells[0]), np.arange(M.shape_cells[1])

        iijj = ndgrid(i, j)
        ii, jj = iijj[:, 0], iijj[:, 1]

        if M._meshType == "Curv":
            fN1 = M.reshape(M.face_normals, "F", "Fx", "M")
            fN2 = M.reshape(M.face_normals, "F", "Fy", "M")

        def Pxx(xFace, yFace):
            # xFace is 'fXp' or 'fXm'
            # yFace is 'fYp' or 'fYm'

            # no | node      | f1     | f2
            # 00 | i  ,j     | i  , j | i, j
            # 10 | i+1,j     | i+1, j | i, j
            # 01 | i  ,j+1   | i  , j | i, j+1
            # 11 | i+1,j+1   | i+1, j | i, j+1

            posFx = 0 if xFace == "fXm" else 1
            posFy = 0 if yFace == "fYm" else 1

            ind1 = sub2ind(M.vnFx, np.c_[ii + posFx, jj])
            ind2 = sub2ind(M.vnFy, np.c_[ii, jj + posFy]) + M.nFx

            IND = np.r_[ind1, ind2].flatten()

            PXX = sp.csr_matrix(
                (np.ones(2 * M.nC), (range(2 * M.nC), IND)), shape=(2 * M.nC, M.nF)
            )

            if M._meshType == "Curv":
                I2x2 = inverse_2x2_block_diagonal(
                    get_subarray(fN1[0], [i + posFx, j]),
                    get_subarray(fN1[1], [i + posFx, j]),
                    get_subarray(fN2[0], [i, j + posFy]),
                    get_subarray(fN2[1], [i, j + posFy]),
                )
                PXX = I2x2 * PXX

            return PXX

        return Pxx

    def _getFacePxxx(M):
        """Return a function for creating face projection matrices in 3D.

        Mats takes you from faces a subset of all faces on only the
        faces that you ask for.

        These are centered around a single nodes.
        """
        i, j, k = (
            np.arange(M.shape_cells[0]),
            np.arange(M.shape_cells[1]),
            np.arange(M.shape_cells[2]),
        )

        iijjkk = ndgrid(i, j, k)
        ii, jj, kk = iijjkk[:, 0], iijjkk[:, 1], iijjkk[:, 2]

        if M._meshType == "Curv":
            fN1 = M.reshape(M.face_normals, "F", "Fx", "M")
            fN2 = M.reshape(M.face_normals, "F", "Fy", "M")
            fN3 = M.reshape(M.face_normals, "F", "Fz", "M")

        def Pxxx(xFace, yFace, zFace):
            # xFace is 'fXp' or 'fXm'
            # yFace is 'fYp' or 'fYm'
            # zFace is 'fZp' or 'fZm'

            # no  | node        | f1        | f2        | f3
            # 000 | i  ,j  ,k   | i  , j, k | i, j  , k | i, j, k
            # 100 | i+1,j  ,k   | i+1, j, k | i, j  , k | i, j, k
            # 010 | i  ,j+1,k   | i  , j, k | i, j+1, k | i, j, k
            # 110 | i+1,j+1,k   | i+1, j, k | i, j+1, k | i, j, k
            # 001 | i  ,j  ,k+1 | i  , j, k | i, j  , k | i, j, k+1
            # 101 | i+1,j  ,k+1 | i+1, j, k | i, j  , k | i, j, k+1
            # 011 | i  ,j+1,k+1 | i  , j, k | i, j+1, k | i, j, k+1
            # 111 | i+1,j+1,k+1 | i+1, j, k | i, j+1, k | i, j, k+1

            posX = 0 if xFace == "fXm" else 1
            posY = 0 if yFace == "fYm" else 1
            posZ = 0 if zFace == "fZm" else 1

            ind1 = sub2ind(M.vnFx, np.c_[ii + posX, jj, kk])
            ind2 = sub2ind(M.vnFy, np.c_[ii, jj + posY, kk]) + M.nFx
            ind3 = sub2ind(M.vnFz, np.c_[ii, jj, kk + posZ]) + M.nFx + M.nFy

            IND = np.r_[ind1, ind2, ind3].flatten()

            PXXX = sp.coo_matrix(
                (np.ones(3 * M.nC), (range(3 * M.nC), IND)), shape=(3 * M.nC, M.nF)
            ).tocsr()

            if M._meshType == "Curv":
                I3x3 = inverse_3x3_block_diagonal(
                    get_subarray(fN1[0], [i + posX, j, k]),
                    get_subarray(fN1[1], [i + posX, j, k]),
                    get_subarray(fN1[2], [i + posX, j, k]),
                    get_subarray(fN2[0], [i, j + posY, k]),
                    get_subarray(fN2[1], [i, j + posY, k]),
                    get_subarray(fN2[2], [i, j + posY, k]),
                    get_subarray(fN3[0], [i, j, k + posZ]),
                    get_subarray(fN3[1], [i, j, k + posZ]),
                    get_subarray(fN3[2], [i, j, k + posZ]),
                )
                PXXX = I3x3 * PXXX

            return PXXX

        return Pxxx

    def _getEdgePx(M):
        """Return a function for creating edge projection matrices in 1D."""

        def Px(xEdge):
            if xEdge != "eX0":
                raise TypeError("xEdge = {0!s}, not eX0".format(xEdge))
            return sp.identity(M.nC)

        return Px

    def _getEdgePxx(M):
        """Return a function for creating edge projection matrices in 2D."""
        i, j = np.arange(M.shape_cells[0]), np.arange(M.shape_cells[1])

        iijj = ndgrid(i, j)
        ii, jj = iijj[:, 0], iijj[:, 1]

        if M._meshType == "Curv":
            eT1 = M.reshape(M.edge_tangents, "E", "Ex", "M")
            eT2 = M.reshape(M.edge_tangents, "E", "Ey", "M")

        def Pxx(xEdge, yEdge):
            # no | node      | e1      | e2
            # 00 | i  ,j     | i  ,j   | i  ,j
            # 10 | i+1,j     | i  ,j   | i+1,j
            # 01 | i  ,j+1   | i  ,j+1 | i  ,j
            # 11 | i+1,j+1   | i  ,j+1 | i+1,j
            posX = 0 if xEdge == "eX0" else 1
            posY = 0 if yEdge == "eY0" else 1

            ind1 = sub2ind(M.vnEx, np.c_[ii, jj + posX])
            ind2 = sub2ind(M.vnEy, np.c_[ii + posY, jj]) + M.nEx

            IND = np.r_[ind1, ind2].flatten()

            PXX = sp.coo_matrix(
                (np.ones(2 * M.nC), (range(2 * M.nC), IND)), shape=(2 * M.nC, M.nE)
            ).tocsr()

            if M._meshType == "Curv":
                I2x2 = inverse_2x2_block_diagonal(
                    get_subarray(eT1[0], [i, j + posX]),
                    get_subarray(eT1[1], [i, j + posX]),
                    get_subarray(eT2[0], [i + posY, j]),
                    get_subarray(eT2[1], [i + posY, j]),
                )
                PXX = I2x2 * PXX

            return PXX

        return Pxx

    def _getEdgePxxx(M):
        """Return a function for creating edge projection matrices in 3D."""
        i, j, k = (
            np.arange(M.shape_cells[0]),
            np.arange(M.shape_cells[1]),
            np.arange(M.shape_cells[2]),
        )

        iijjkk = ndgrid(i, j, k)
        ii, jj, kk = iijjkk[:, 0], iijjkk[:, 1], iijjkk[:, 2]

        if M._meshType == "Curv":
            eT1 = M.reshape(M.edge_tangents, "E", "Ex", "M")
            eT2 = M.reshape(M.edge_tangents, "E", "Ey", "M")
            eT3 = M.reshape(M.edge_tangents, "E", "Ez", "M")

        def Pxxx(xEdge, yEdge, zEdge):
            # no  | node        | e1          | e2          | e3
            # 000 | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k
            # 100 | i+1,j  ,k   | i  ,j  ,k   | i+1,j  ,k   | i+1,j  ,k
            # 010 | i  ,j+1,k   | i  ,j+1,k   | i  ,j  ,k   | i  ,j+1,k
            # 110 | i+1,j+1,k   | i  ,j+1,k   | i+1,j  ,k   | i+1,j+1,k
            # 001 | i  ,j  ,k+1 | i  ,j  ,k+1 | i  ,j  ,k+1 | i  ,j  ,k
            # 101 | i+1,j  ,k+1 | i  ,j  ,k+1 | i+1,j  ,k+1 | i+1,j  ,k
            # 011 | i  ,j+1,k+1 | i  ,j+1,k+1 | i  ,j  ,k+1 | i  ,j+1,k
            # 111 | i+1,j+1,k+1 | i  ,j+1,k+1 | i+1,j  ,k+1 | i+1,j+1,k

            posX = (
                [0, 0]
                if xEdge == "eX0"
                else [1, 0]
                if xEdge == "eX1"
                else [0, 1]
                if xEdge == "eX2"
                else [1, 1]
            )
            posY = (
                [0, 0]
                if yEdge == "eY0"
                else [1, 0]
                if yEdge == "eY1"
                else [0, 1]
                if yEdge == "eY2"
                else [1, 1]
            )
            posZ = (
                [0, 0]
                if zEdge == "eZ0"
                else [1, 0]
                if zEdge == "eZ1"
                else [0, 1]
                if zEdge == "eZ2"
                else [1, 1]
            )

            ind1 = sub2ind(M.vnEx, np.c_[ii, jj + posX[0], kk + posX[1]])
            ind2 = sub2ind(M.vnEy, np.c_[ii + posY[0], jj, kk + posY[1]]) + M.nEx
            ind3 = (
                sub2ind(M.vnEz, np.c_[ii + posZ[0], jj + posZ[1], kk]) + M.nEx + M.nEy
            )

            IND = np.r_[ind1, ind2, ind3].flatten()

            PXXX = sp.coo_matrix(
                (np.ones(3 * M.nC), (range(3 * M.nC), IND)), shape=(3 * M.nC, M.nE)
            ).tocsr()

            if M._meshType == "Curv":
                I3x3 = inverse_3x3_block_diagonal(
                    get_subarray(eT1[0], [i, j + posX[0], k + posX[1]]),
                    get_subarray(eT1[1], [i, j + posX[0], k + posX[1]]),
                    get_subarray(eT1[2], [i, j + posX[0], k + posX[1]]),
                    get_subarray(eT2[0], [i + posY[0], j, k + posY[1]]),
                    get_subarray(eT2[1], [i + posY[0], j, k + posY[1]]),
                    get_subarray(eT2[2], [i + posY[0], j, k + posY[1]]),
                    get_subarray(eT3[0], [i + posZ[0], j + posZ[1], k]),
                    get_subarray(eT3[1], [i + posZ[0], j + posZ[1], k]),
                    get_subarray(eT3[2], [i + posZ[0], j + posZ[1], k]),
                )
                PXXX = I3x3 * PXXX

            return PXXX

        return Pxxx
