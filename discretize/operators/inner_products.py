from scipy import sparse as sp
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
)
import numpy as np
from discretize.utils.code_utils import deprecate_method
import warnings


class InnerProducts(object):
    """Class for constructing inner product matrices.

    ``InnerProducts`` is a mixin class for constructing inner product matrices,
    their inverses and their derivatives with respect to model parameters.
    The ``InnerProducts`` class is inherited by all ``discretize`` mesh classes.
    In practice, we don't create instances of the ``InnerProducts`` class in
    order to construct inner product matrices, their inverses or their derivatives.
    These quantities are instead constructed from instances of ``discretize``
    meshes using the appropriate method.
    """

    def get_face_inner_product(
        self,
        model=None,
        invert_model=False,
        invert_matrix=False,
        do_fast=True,
        **kwargs
    ):
        r"""Generate the face inner product matrix or its inverse.

        This method generates the inner product matrix (or its inverse)
        when discrete variables are defined on mesh faces. It is also capable of
        constructing the inner product matrix when physical properties
        are defined in the form of constitutive relations. For a comprehensive
        description of the inner product matrices that can be constructed
        with **get_face_inner_product**, see *Notes*.

        Parameters
        ----------
        model : None or numpy.ndarray, optional
            Parameters defining the material properties for every cell in the mesh.
            Inner product matrices can be constructed for the following cases:

            - *None* : returns the basic inner product matrix
            - *(n_cells)* :class:`numpy.ndarray` : returns inner product matrix for an
              isotropic model. The array contains a scalar physical property value for
              each cell.
            - *(n_cells, dim)* :class:`numpy.ndarray` : returns inner product matrix for
              diagonal anisotropic case. Columns are ordered ``np.c_[σ_xx, σ_yy, σ_zz]``.
              This can also a be a 1D array with the same number of total elements in
              column major order.
            - *(n_cells, 3)* :class:`numpy.ndarray` (``dim`` is 2) or
              *(n_cells, 6)* :class:`numpy.ndarray` (``dim`` is 3) :
              returns inner product matrix for full tensor properties case. Columns are
              ordered ``np.c_[σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]`` This can also be a
              1D array with the same number of total elements in column major order.

        invert_model : bool, optional
            The inverse of *model* is used as the physical property.
        invert_matrix : bool, optional
            Returns the inverse of the inner product matrix.
            The inverse not implemented for full tensor properties.
        do_fast : bool, optional
            Do a faster implementation (if available).

        Returns
        -------
        (n_faces, n_faces) scipy.sparse.csr_matrix
            inner product matrix

        Notes
        -----
        For continuous vector quantities :math:`\vec{u}` and :math:`\vec{w}`
        whose discrete representations :math:`\mathbf{u}` and :math:`\mathbf{w}`
        live on the faces, **get_face_inner_product** constructs the inner product matrix
        :math:`\mathbf{M_\ast}` (or its inverse :math:`\mathbf{M_\ast^{-1}}`) for the
        following cases:

        **Basic Inner Product:** the inner product between :math:`\vec{u}` and :math:`\vec{w}`

        .. math::
            \langle \vec{u}, \vec{w} \rangle = \mathbf{u^T \, M \, w}

        **Isotropic Case:** the inner product between :math:`\vec{u}` and :math:`\sigma \vec{w}`
        where :math:`\sigma` is a scalar function.

        .. math::
            \langle \vec{u}, \sigma \vec{w} \rangle = \mathbf{u^T \, M_\sigma \, w}

        **Tensor Case:** the inner product between :math:`\vec{u}` and :math:`\Sigma \vec{w}`
        where :math:`\Sigma` is tensor function; :math:`\sigma_{xy} = \sigma_{xz} = \sigma_{yz} = 0`
        for diagonal anisotropy.

        .. math::
            \langle \vec{u}, \Sigma \vec{w} \rangle = \mathbf{u^T \, M_\Sigma \, w}
            \;\;\; \textrm{where} \;\;\;
            \Sigma = \begin{bmatrix}
            \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
            \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\
            \sigma_{xz} & \sigma_{yz} & \sigma_{zz}
            \end{bmatrix}

        Examples
        --------
        Here we provide some examples of face inner product matrices.
        For simplicity, we will work on a 2 x 2 x 2 tensor mesh.
        As seen below, we begin by constructing and imaging the basic
        face inner product matrix.

        >>> from discretize import TensorMesh
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import matplotlib as mpl

        >>> h = np.ones(2)
        >>> mesh = TensorMesh([h, h, h])
        >>> Mf = mesh.get_face_inner_product()

        >>> fig = plt.figure(figsize=(6, 6))
        >>> ax = fig.add_subplot(111)
        >>> ax.imshow(Mf.todense())
        >>> ax.set_title('Basic Face Inner Product Matrix', fontsize=18)
        >>> plt.show()

        Next, we consider the case where the physical properties
        of the cells are defined by consistutive relations. For
        the isotropic, diagonal anisotropic and full tensor cases,
        we show the physical property tensor for a single cell.

        Define 4 constitutive parameters and define the tensor
        for each cell for isotropic, diagonal and tensor cases.

        >>> sig1, sig2, sig3, sig4, sig5, sig6 = 6, 5, 4, 3, 2, 1
        >>> sig_iso_tensor = sig1 * np.eye(3)
        >>> sig_diag_tensor = np.diag(np.array([sig1, sig2, sig3]))
        >>> sig_full_tensor = np.array([
        ...     [sig1, sig4, sig5],
        ...     [sig4, sig2, sig6],
        ...     [sig5, sig6, sig3]
        ... ])

        Then plot matrix entries,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(15, 5))
            >>> ax1 = fig.add_subplot(131)
            >>> ax1.imshow(sig_iso_tensor)
            >>> ax1.axis('off')
            >>> ax1.set_title("Tensor (isotropic)", fontsize=16)
            >>> ax2 = fig.add_subplot(132)
            >>> ax2.imshow(sig_diag_tensor)
            >>> ax2.axis('off')
            >>> ax2.set_title("Tensor (diagonal anisotropic)", fontsize=16)
            >>> ax3 = fig.add_subplot(133)
            >>> ax3.imshow(sig_full_tensor)
            >>> ax3.axis('off')
            >>> ax3.set_title("Tensor (full anisotropic)", fontsize=16)
            >>> plt.show()

        Here, construct and image the face inner product matrices for
        the isotropic, diagonal anisotropic and full tensor cases.
        Spy plots are used to demonstrate the sparsity of the inner
        product matrices.

        Isotropic case:

        >>> v = np.ones(mesh.nC)
        >>> sig = sig1 * v
        >>> M1 = mesh.get_face_inner_product(sig)

        Diagonal anisotropic case:

        >>> sig = np.c_[sig1*v, sig2*v, sig3*v]
        >>> M2 = mesh.get_face_inner_product(sig)

        Full anisotropic case:

        >>> sig = np.tile(np.c_[sig1, sig2, sig3, sig4, sig5, sig6], (mesh.nC, 1))
        >>> M3 = mesh.get_face_inner_product(sig)

        And then we can plot the sparse representation,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(12, 4))
            >>> ax1 = fig.add_subplot(131)
            >>> ax1.spy(M1, ms=5)
            >>> ax1.set_title("M (isotropic)", fontsize=16)
            >>> ax2 = fig.add_subplot(132)
            >>> ax2.spy(M2, ms=5)
            >>> ax2.set_title("M (diagonal anisotropic)", fontsize=16)
            >>> ax3 = fig.add_subplot(133)
            >>> ax3.spy(M3, ms=5)
            >>> ax3.set_title("M (full anisotropic)", fontsize=16)
            >>> plt.show()
        """
        if "invProp" in kwargs:
            warnings.warn(
                "The invProp keyword argument has been deprecated, please use invert_model. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            invert_model = kwargs["invProp"]
        if "invMat" in kwargs:
            warnings.warn(
                "The invMat keyword argument has been deprecated, please use invert_matrix. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            invert_matrix = kwargs["invMat"]
        if "doFast" in kwargs:
            warnings.warn(
                "The doFast keyword argument has been deprecated, please use do_fast. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            do_fast = kwargs["doFast"]

        return self._getInnerProduct(
            "F",
            model=model,
            invert_model=invert_model,
            invert_matrix=invert_matrix,
            do_fast=do_fast,
        )

    def get_edge_inner_product(
        self,
        model=None,
        invert_model=False,
        invert_matrix=False,
        do_fast=True,
        **kwargs
    ):
        r"""Generate the edge inner product matrix or its inverse.

        This method generates the inner product matrix (or its inverse)
        when discrete variables are defined on mesh edges. It is also capable of
        constructing the inner product matrix when physical properties
        are defined in the form of constitutive relations. For a comprehensive
        description of the inner product matrices that can be constructed
        with **get_edge_inner_product**, see *Notes*.

        Parameters
        ----------
        model : None or numpy.ndarray
            Parameters defining the material properties for every cell in the mesh.
            Inner product matrices can be constructed for the following cases:

            - *None* : returns the basic inner product matrix
            - *(n_cells)* :class:`numpy.ndarray` : returns inner product matrix for an
              isotropic model. The array contains a scalar physical property value for
              each cell.
            - *(n_cells, dim)* :class:`numpy.ndarray` : returns inner product matrix for
              diagonal anisotropic case. Columns are ordered ``np.c_[σ_xx, σ_yy, σ_zz]``.
              This can also a be a 1D array with the same number of total elements in
              column major order.
            - *(n_cells, 3)* :class:`numpy.ndarray` (``dim`` is 2) or
              *(n_cells, 6)* :class:`numpy.ndarray` (``dim`` is 3) :
              returns inner product matrix for full tensor properties case. Columns are
              ordered ``np.c_[σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]`` This can also be a
              1D array with the same number of total elements in column major order.

        invert_model : bool, optional
            The inverse of *model* is used as the physical property.
        invert_matrix : bool, optional
            Teturns the inverse of the inner product matrix.
            The inverse not implemented for full tensor properties.
        do_fast : bool, optional
            Do a faster implementation (if available).

        Returns
        -------
        (n_edges, n_edges) scipy.sparse.csr_matrix
            inner product matrix

        Notes
        -----
        For continuous vector quantities :math:`\vec{u}` and :math:`\vec{w}`
        whose discrete representations :math:`\mathbf{u}` and :math:`\mathbf{w}`
        live on the edges, **get_edge_inner_product** constructs the inner product
        matrix :math:`\mathbf{M_\ast}` (or its inverse :math:`\mathbf{M_\ast^{-1}}`) for
        the following cases:

        **Basic Inner Product:** the inner product between :math:`\vec{u}` and
        :math:`\vec{w}`.

        .. math::
            \langle \vec{u}, \vec{w} \rangle = \mathbf{u^T \, M \, w}

        **Isotropic Case:** the inner product between :math:`\vec{u}` and
        :math:`\sigma \vec{w}` where :math:`\sigma` is a scalar function.

        .. math::
            \langle \vec{u}, \sigma \vec{w} \rangle = \mathbf{u^T \, M_\sigma \, w}

        **Tensor Case:** the inner product between :math:`\vec{u}` and
        :math:`\Sigma \vec{w}` where :math:`\Sigma` is tensor function;
        :math:`\sigma_{xy} = \sigma_{xz} = \sigma_{yz} = 0` for diagonal anisotropy.

        .. math::
            \langle \vec{u}, \Sigma \vec{w} \rangle =
            \mathbf{u^T \, M_\Sigma \, w} \;\;\; \textrm{where} \;\;\;
            \Sigma = \begin{bmatrix}
            \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
            \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\
            \sigma_{xz} & \sigma_{yz} & \sigma_{zz}
            \end{bmatrix}

        Examples
        --------
        Here we provide some examples of edge inner product matrices.
        For simplicity, we will work on a 2 x 2 x 2 tensor mesh.
        As seen below, we begin by constructing and imaging the basic
        edge inner product matrix.

        >>> from discretize import TensorMesh
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import matplotlib as mpl

        >>> h = np.ones(2)
        >>> mesh = TensorMesh([h, h, h])
        >>> Me = mesh.get_edge_inner_product()

        >>> fig = plt.figure(figsize=(6, 6))
        >>> ax = fig.add_subplot(111)
        >>> ax.imshow(Me.todense())
        >>> ax.set_title('Basic Edge Inner Product Matrix', fontsize=18)
        >>> plt.show()

        Next, we consider the case where the physical properties
        of the cells are defined by consistutive relations. For
        the isotropic, diagonal anisotropic and full tensor cases,
        we show the physical property tensor for a single cell.

        Define 4 constitutive parameters and define the tensor
        for each cell for isotropic, diagonal and tensor cases.

        >>> sig1, sig2, sig3, sig4, sig5, sig6 = 6, 5, 4, 3, 2, 1
        >>> sig_iso_tensor = sig1 * np.eye(3)
        >>> sig_diag_tensor = np.diag(np.array([sig1, sig2, sig3]))
        >>> sig_full_tensor = np.array([
        ...     [sig1, sig4, sig5],
        ...     [sig4, sig2, sig6],
        ...     [sig5, sig6, sig3]
        ... ])

        Then plot the matrix entries,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(15, 5))
            >>> ax1 = fig.add_subplot(131)
            >>> ax1.imshow(sig_iso_tensor)
            >>> ax1.axis('off')
            >>> ax1.set_title("Tensor (isotropic)", fontsize=16)
            >>> ax2 = fig.add_subplot(132)
            >>> ax2.imshow(sig_diag_tensor)
            >>> ax2.axis('off')
            >>> ax2.set_title("Tensor (diagonal anisotropic)", fontsize=16)
            >>> ax3 = fig.add_subplot(133)
            >>> ax3.imshow(sig_full_tensor)
            >>> ax3.axis('off')
            >>> ax3.set_title("Tensor (full anisotropic)", fontsize=16)
            >>> plt.show()

        Here construct and image the edge inner product matrices for
        the isotropic, diagonal anisotropic and full tensor cases.
        Spy plots are used to demonstrate the sparsity of the inner
        product matrices.

        Isotropic case:

        >>> v = np.ones(mesh.nC)
        >>> sig = sig1 * v
        >>> M1 = mesh.get_edge_inner_product(sig)

        Diagonal anisotropic case:

        >>> sig = np.c_[sig1*v, sig2*v, sig3*v]
        >>> M2 = mesh.get_edge_inner_product(sig)

        Full anisotropic

        >>> sig = np.tile(np.c_[sig1, sig2, sig3, sig4, sig5, sig6], (mesh.nC, 1))
        >>> M3 = mesh.get_edge_inner_product(sig)

        Then plot the sparse representation,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(12, 4))
            >>> ax1 = fig.add_subplot(131)
            >>> ax1.spy(M1, ms=5)
            >>> ax1.set_title("M (isotropic)", fontsize=16)
            >>> ax2 = fig.add_subplot(132)
            >>> ax2.spy(M2, ms=5)
            >>> ax2.set_title("M (diagonal anisotropic)", fontsize=16)
            >>> ax3 = fig.add_subplot(133)
            >>> ax3.spy(M3, ms=5)
            >>> ax3.set_title("M (full anisotropic)", fontsize=16)
            >>> plt.show()

        """
        if "invProp" in kwargs:
            warnings.warn(
                "The invProp keyword argument has been deprecated, please use invert_model. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            invert_model = kwargs["invProp"]
        if "invMat" in kwargs:
            warnings.warn(
                "The invMat keyword argument has been deprecated, please use invert_matrix. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            invert_matrix = kwargs["invMat"]
        if "doFast" in kwargs:
            warnings.warn(
                "The doFast keyword argument has been deprecated, please use do_fast. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            do_fast = kwargs["doFast"]
        return self._getInnerProduct(
            "E",
            model=model,
            invert_model=invert_model,
            invert_matrix=invert_matrix,
            do_fast=do_fast,
        )

    def _getInnerProduct(
        self,
        projection_type,
        model=None,
        invert_model=False,
        invert_matrix=False,
        do_fast=True,
        **kwargs
    ):
        """get the inner product matrix

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
            warnings.warn(
                "The invProp keyword argument has been deprecated, please use invert_model. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            invert_model = kwargs["invProp"]
        if "invMat" in kwargs:
            warnings.warn(
                "The invMat keyword argument has been deprecated, please use invert_matrix. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            invert_matrix = kwargs["invMat"]
        if "doFast" in kwargs:
            warnings.warn(
                "The doFast keyword argument has been deprecated, please use do_fast. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            do_fast = kwargs["doFast"]
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
        """
        Parameters
        ----------
        projection_type : str
            'F' for faces 'E' for edges

        tensorType : TensorType
            type of the tensor: TensorType(mesh, sigma)
        """
        if not isinstance(tensorType, TensorType):
            raise TypeError("tensorType must be an instance of TensorType.")
        if projection_type not in ["F", "E"]:
            raise TypeError("projection_type must be 'F' for faces or 'E' for edges")

        d = self.dim
        # We will multiply by sqrt on each side to keep symmetry
        V = sp.kron(sp.identity(d), sdiag(np.sqrt((2 ** (-d)) * self.cell_volumes)))

        nodes = ["000", "100", "010", "110", "001", "101", "011", "111"][: 2 ** d]

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

    def get_face_inner_product_deriv(
        self, model, do_fast=True, invert_model=False, invert_matrix=False, **kwargs
    ):
        r"""Function handle to multiply vector with derivative of face inner product matrix (or its inverse).

        Let :math:`\mathbf{M}(\mathbf{m})` be the face inner product matrix
        constructed with a set of physical property parameters :math:`\mathbf{m}`
        (or its inverse). **get_face_inner_product_deriv** constructs a function handle

        .. math::
            \mathbf{F}(\mathbf{u}) = \mathbf{u}^T \, \frac{\partial \mathbf{M}(\mathbf{m})}{\partial \mathbf{m}}

        which accepts any numpy.array :math:`\mathbf{u}` of shape (n_faces,). That is,
        **get_face_inner_product_deriv** constructs a function handle for computing
        the dot product between a vector :math:`\mathbf{u}` and the derivative of the
        face inner product matrix (or its inverse) with respect to the property parameters.
        When computed, :math:`\mathbf{F}(\mathbf{u})` returns a ``scipy.sparse.csr_matrix``
        of shape (n_faces, n_param).

        The function handle can be created for isotropic, diagonal
        isotropic and full tensor physical properties; see notes.

        Parameters
        ----------
        model : numpy.ndarray
            Parameters defining the material properties for every cell in the mesh.
            Inner product matrices can be constructed for the following cases:

            - *(n_cells)* :class:`numpy.ndarray` : Isotropic case. *model* contains a
              scalar physical property value for each cell.
            - *(n_cells, dim)* :class:`numpy.ndarray` : Diagonal anisotropic case.
              Columns are ordered ``np.c_[σ_xx, σ_yy, σ_zz]``. This can also a be a 1D
              array with the same number of total elements in column major order.
            - *(n_cells, 3)* :class:`numpy.ndarray` (``dim`` is 2) or
              *(n_cells, 6)* :class:`numpy.ndarray` (``dim`` is 3) : Full tensor properties case. Columns
              are ordered ``np.c_[σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]`` This can also be
              a 1D array with the same number of total elements in column major order.

        invert_model : bool, optional
            The inverse of *model* is used as the physical property.
        invert_matrix : bool, optional
            Returns the inverse of the inner product matrix.
            The inverse not implemented for full tensor properties.
        do_fast : bool, optional
            Do a faster implementation (if available).

        Returns
        -------
        function
            The function handle :math:`\mathbf{F}(\mathbf{u})` which accepts a
            (``n_faces``) :class:`numpy.ndarray` :math:`\mathbf{u}`. The function
            returns a (``n_faces``, ``n_params``) :class:`scipy.sparse.csr_matrix`.

        Notes
        -----
        Let :math:`\mathbf{M}(\mathbf{m})` be the face inner product matrix (or its inverse)
        for the set of physical property parameters :math:`\mathbf{m}`. And let :math:`\mathbf{u}`
        be a discrete quantity that lives on the faces. **get_face_inner_product_deriv**
        creates a function handle for computing the following:

        .. math::
            \mathbf{F}(\mathbf{u}) = \mathbf{u}^T \, \frac{\partial \mathbf{M}(\mathbf{m})}{\partial \mathbf{m}}

        The dimensions of the sparse matrix constructed by computing :math:`\mathbf{F}(\mathbf{u})`
        for some :math:`\mathbf{u}` depends on the constitutive relation defined for each cell.
        These cases are summarized below.

        **Isotropic Case:** The physical property for each cell is defined by a scalar value.
        Therefore :math:`\mathbf{m}` is a (``n_cells``) :class:`numpy.ndarray`. The sparse matrix
        output by computing :math:`\mathbf{F}(\mathbf{u})` has shape (``n_faces``, ``n_cells``).

        **Diagonal Anisotropic Case:** In this case, the physical properties for each cell are
        defined by a diagonal tensor

        .. math::
            \Sigma = \begin{bmatrix}
            \sigma_{xx} & 0 & 0 \\
            0 & \sigma_{yy} & 0 \\
            0 & 0 & \sigma_{zz}
            \end{bmatrix}

        Thus there are ``dim * n_cells`` physical property parameters and :math:`\mathbf{m}` is
        a (``dim * n_cells``) :class:`numpy.ndarray`.  The sparse matrix
        output by computing :math:`\mathbf{F}(\mathbf{u})` has shape (``n_faces``, ``dim * n_cells``).

        **Full Tensor Case:** In this case, the physical properties for each cell are
        defined by a full tensor

        .. math::
            \Sigma = \begin{bmatrix}
            \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
            \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\
            \sigma_{xz} & \sigma_{yz} & \sigma_{zz}
            \end{bmatrix}

        Thus there are ``6 * n_cells`` physical property parameters in 3 dimensions, or
        ``3 * n_cells`` physical property parameters in 2 dimensions, and
        :math:`\mathbf{m}` is a (``n_params``) :class:`numpy.ndarray`.
        The sparse matrix output by computing :math:`\mathbf{F}(\mathbf{u})`
        has shape (``n_faces``, ``n_params``).

        Examples
        --------
        Here, we construct a 4 cell by 4 cell tensor mesh. For our first example we
        consider isotropic physical properties; that is, the physical properties
        of each cell are defined a scalar value. We construct the face inner product
        matrix and visualize it with a spy plot. We then use
        **get_face_inner_product_deriv** to construct the function handle
        :math:`\mathbf{F}(\mathbf{u})` and plot the evaluation
        of this function on a spy plot.

        >>> from discretize import TensorMesh
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import matplotlib as mpl
        >>> mpl.rcParams.update({'font.size': 14})
        >>> np.random.seed(45)
        >>> mesh = TensorMesh([[(1, 4)], [(1, 4)]])

        Define a model, and a random vector to multiply the derivative with,
        then we grab the respective derivative function and calculate the
        sparse matrix,

        >>> m = np.random.rand(mesh.nC)  # physical property parameters
        >>> u = np.random.rand(mesh.nF)  # vector of shape (n_faces)
        >>> Mf = mesh.get_face_inner_product(m)
        >>> F = mesh.get_face_inner_product_deriv(m)  # Function handle
        >>> dFdm_u = F(u)

        Spy plot for the inner product matrix and its derivative

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(15, 5))
            >>> ax1 = fig.add_axes([0.05, 0.05, 0.3, 0.85])
            >>> ax1.spy(Mf, ms=6)
            >>> ax1.set_title("Face Inner Product Matrix (Isotropic)", fontsize=14, pad=5)
            >>> ax1.set_xlabel("Face Index", fontsize=12)
            >>> ax1.set_ylabel("Face Index", fontsize=12)
            >>> ax2 = fig.add_axes([0.43, 0.05, 0.17, 0.8])
            >>> ax2.spy(dFdm_u, ms=6)
            >>> ax2.set_title(
            ...     "$u^T \, \dfrac{\partial M(m)}{\partial m}$ (Isotropic)",
            ...     fontsize=14, pad=5
            ... )
            >>> ax2.set_xlabel("Parameter Index", fontsize=12)
            >>> ax2.set_ylabel("Face Index", fontsize=12)
            >>> plt.show()

        For our second example, the physical properties on the mesh are fully
        anisotropic; that is, the physical properties of each cell are defined
        by a tensor with parameters :math:`\sigma_1`, :math:`\sigma_2` and :math:`\sigma_3`.
        Once again we construct the face inner product matrix and visualize it with a
        spy plot. We then use **get_face_inner_product_deriv** to construct the
        function handle :math:`\mathbf{F}(\mathbf{u})` and plot the evaluation
        of this function on a spy plot.

        >>> m = np.random.rand(mesh.nC, 3)  # anisotropic physical property parameters
        >>> u = np.random.rand(mesh.nF)     # vector of shape (n_faces)
        >>> Mf = mesh.get_face_inner_product(m)
        >>> F = mesh.get_face_inner_product_deriv(m)  # Function handle
        >>> dFdm_u = F(u)

        Plot the anisotropic inner product matrix and its derivative matrix,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(15, 5))
            >>> ax1 = fig.add_axes([0.05, 0.05, 0.3, 0.8])
            >>> ax1.spy(Mf, ms=6)
            >>> ax1.set_title("Face Inner Product (Full Tensor)", fontsize=14, pad=5)
            >>> ax1.set_xlabel("Face Index", fontsize=12)
            >>> ax1.set_ylabel("Face Index", fontsize=12)
            >>> ax2 = fig.add_axes([0.4, 0.05, 0.45, 0.85])
            >>> ax2.spy(dFdm_u, ms=6)
            >>> ax2.set_title(
            ...     "$u^T \, \dfrac{\partial M(m)}{\partial m} \;$ (Full Tensor)",
            ...     fontsize=14, pad=5
            ... )
            >>> ax2.set_xlabel("Parameter Index", fontsize=12)
            >>> ax2.set_ylabel("Face Index", fontsize=12)
            >>> plt.show()

        """
        if "invProp" in kwargs:
            warnings.warn(
                "The invProp keyword argument has been deprecated, please use invert_model. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            invert_model = kwargs["invProp"]
        if "invMat" in kwargs:
            warnings.warn(
                "The invMat keyword argument has been deprecated, please use invert_matrix. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            invert_matrix = kwargs["invMat"]
        if "doFast" in kwargs:
            warnings.warn(
                "The doFast keyword argument has been deprecated, please use do_fast. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            do_fast = kwargs["doFast"]
        return self._getInnerProductDeriv(
            model,
            "F",
            do_fast=do_fast,
            invert_model=invert_model,
            invert_matrix=invert_matrix,
        )

    def get_edge_inner_product_deriv(
        self, model, do_fast=True, invert_model=False, invert_matrix=False, **kwargs
    ):
        r"""Function handle to multiply vector with derivative of edge inner product matrix (or its inverse).

        Let :math:`\mathbf{M}(\mathbf{m})` be the edge inner product matrix
        constructed with a set of physical property parameters :math:`\mathbf{m}`
        (or its inverse). **get_edge_inner_product_deriv** constructs a function handle

        .. math::
            \mathbf{F}(\mathbf{u}) = \mathbf{u}^T \, \frac{\partial \mathbf{M}(\mathbf{m})}{\partial \mathbf{m}}

        which accepts any numpy.array :math:`\mathbf{u}` of shape (n_edges,). That is,
        **get_edge_inner_product_deriv** constructs a function handle for computing
        the dot product between a vector :math:`\mathbf{u}` and the derivative of the
        edge inner product matrix (or its inverse) with respect to the property parameters.
        When computed, :math:`\mathbf{F}(\mathbf{u})` returns a ``scipy.sparse.csr_matrix``
        of shape (n_edges, n_param).

        The function handle can be created for isotropic, diagonal
        isotropic and full tensor physical properties; see notes.

        Parameters
        ----------
        model : numpy.ndarray
            Parameters defining the material properties for every cell in the mesh.

            Allows for the following cases:

            - *(n_cells)* :class:`numpy.ndarray` : Isotropic case. *model* contains a
              scalar physical property value for each cell.
            - *(n_cells, dim)* :class:`numpy.ndarray` : Diagonal anisotropic case.
              Columns are ordered ``np.c_[σ_xx, σ_yy, σ_zz]``. This can also a be a 1D
              array with the same number of total elements in column major order.
            - *(n_cells, 3)* :class:`numpy.ndarray` (``dim`` is 2) or
              *(n_cells, 6)* :class:`numpy.ndarray` (``dim`` is 3) : Full tensor properties case. Columns
              are ordered ``np.c_[σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]`` This can also be
              a 1D array with the same number of total elements in column major order.

        invert_model : bool, optional
            The inverse of *model* is used as the physical property.
        invert_matrix : bool, optional
            Returns the function handle for the inverse of the inner product matrix
            The inverse not implemented for full tensor properties.
        do_fast : bool, optional
            Do a faster implementation (if available).

        Returns
        -------
        function
            The function handle :math:`\mathbf{F}(\mathbf{u})` which accepts a
            (``n_edges``) :class:`numpy.ndarray` :math:`\mathbf{u}`. The function
            returns a (``n_edges``, ``n_params``) :class:`scipy.sparse.csr_matrix`.

        Notes
        -----
        Let :math:`\mathbf{M}(\mathbf{m})` be the edge inner product matrix (or its inverse)
        for the set of physical property parameters :math:`\mathbf{m}`. And let :math:`\mathbf{u}`
        be a discrete quantity that lives on the edges. **get_edge_inner_product_deriv**
        creates a function handle for computing the following:

        .. math::
            \mathbf{F}(\mathbf{u}) = \mathbf{u}^T \, \frac{\partial \mathbf{M}(\mathbf{m})}{\partial \mathbf{m}}

        The dimensions of the sparse matrix constructed by computing :math:`\mathbf{F}(\mathbf{u})`
        for some :math:`\mathbf{u}` depends on the constitutive relation defined for each cell.
        These cases are summarized below.

        **Isotropic Case:** The physical property for each cell is defined by a scalar value.
        Therefore :math:`\mathbf{m}` is a (``n_cells``) :class:`numpy.ndarray`. The sparse matrix
        output by computing :math:`\mathbf{F}(\mathbf{u})` has shape (``n_edges``, ``n_cells``).

        **Diagonal Anisotropic Case:** In this case, the physical properties for each cell are
        defined by a diagonal tensor

        .. math::
            \Sigma = \begin{bmatrix}
            \sigma_{xx} & 0 & 0 \\
            0 & \sigma_{yy} & 0 \\
            0 & 0 & \sigma_{zz}
            \end{bmatrix}

        Thus there are ``dim * n_cells`` physical property parameters and :math:`\mathbf{m}` is
        a (``dim * n_cells``) :class:`numpy.ndarray`.  The sparse matrix
        output by computing :math:`\mathbf{F}(\mathbf{u})` has shape (``n_edges``, ``dim * n_cells``).

        **Full Tensor Case:** In this case, the physical properties for each cell are
        defined by a full tensor

        .. math::
            \Sigma = \begin{bmatrix}
            \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
            \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\
            \sigma_{xz} & \sigma_{yz} & \sigma_{zz}
            \end{bmatrix}

        Thus there are ``6 * n_cells`` physical property parameters in 3 dimensions, or
        ``3 * n_cells`` physical property parameters in 2 dimensions, and :math:`\mathbf{m}` is
        a (``n_params``) :class:`numpy.ndarray`. The sparse matrix
        output by computing :math:`\mathbf{F}(\mathbf{u})` has shape (``n_edges``, ``n_params``).

        Examples
        --------
        Here, we construct a 4 cell by 4 cell tensor mesh. For our first example we
        consider isotropic physical properties; that is, the physical properties
        of each cell are defined a scalar value. We construct the edge inner product
        matrix and visualize it with a spy plot. We then use
        **get_edge_inner_product_deriv** to construct the function handle
        :math:`\mathbf{F}(\mathbf{u})` and plot the evaluation of this function on a spy
        plot.

        >>> from discretize import TensorMesh
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import matplotlib as mpl
        >>> mpl.rcParams.update({'font.size': 14})
        >>> np.random.seed(45)
        >>> mesh = TensorMesh([[(1, 4)], [(1, 4)]])

        Next we create a random isotropic model vector, and a random vector to multiply
        the derivative with (for illustration purposes).

        >>> m = np.random.rand(mesh.nC)  # physical property parameters
        >>> u = np.random.rand(mesh.nF)  # vector of shape (n_edges)
        >>> Me = mesh.get_edge_inner_product(m)
        >>> F = mesh.get_edge_inner_product_deriv(m)  # Function handle
        >>> dFdm_u = F(u)

        Plot inner product matrix and its derivative matrix

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(15, 5))
            >>> ax1 = fig.add_axes([0.05, 0.05, 0.3, 0.8])
            >>> ax1.spy(Me, ms=6)
            >>> ax1.set_title("Edge Inner Product Matrix (Isotropic)", fontsize=14, pad=5)
            >>> ax1.set_xlabel("Edge Index", fontsize=12)
            >>> ax1.set_ylabel("Edge Index", fontsize=12)
            >>> ax2 = fig.add_axes([0.43, 0.05, 0.17, 0.8])
            >>> ax2.spy(dFdm_u, ms=6)
            >>> ax2.set_title(
            ...     "$u^T \, \dfrac{\partial M(m)}{\partial m}$ (Isotropic)",
            ...     fontsize=14, pad=5
            ... )
            >>> ax2.set_xlabel("Parameter Index", fontsize=12)
            >>> ax2.set_ylabel("Edge Index", fontsize=12)
            >>> plt.show()

        For our second example, the physical properties on the mesh are fully
        anisotropic; that is, the physical properties of each cell are defined
        by a tensor with parameters :math:`\sigma_1`, :math:`\sigma_2` and :math:`\sigma_3`.
        Once again we construct the edge inner product matrix and visualize it with a
        spy plot. We then use **get_edge_inner_product_deriv** to construct the
        function handle :math:`\mathbf{F}(\mathbf{u})` and plot the evaluation
        of this function on a spy plot.

        >>> m = np.random.rand(mesh.nC, 3)  # physical property parameters
        >>> u = np.random.rand(mesh.nF)     # vector of shape (n_edges)
        >>> Me = mesh.get_edge_inner_product(m)
        >>> F = mesh.get_edge_inner_product_deriv(m)  # Function handle
        >>> dFdm_u = F(u)

        Plot the anisotropic inner product matrix and its derivative matrix

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(15, 5))
            >>> ax1 = fig.add_axes([0.05, 0.05, 0.3, 0.8])
            >>> ax1.spy(Me, ms=6)
            >>> ax1.set_title("Edge Inner Product (Full Tensor)", fontsize=14, pad=5)
            >>> ax1.set_xlabel("Edge Index", fontsize=12)
            >>> ax1.set_ylabel("Edge Index", fontsize=12)
            >>> ax2 = fig.add_axes([0.4, 0.05, 0.45, 0.8])
            >>> ax2.spy(dFdm_u, ms=6)
            >>> ax2.set_title(
            ...     "$u^T \, \dfrac{\partial M(m)}{\partial m} \;$ (Full Tensor)",
            ...     fontsize=14, pad=5
            ... )
            >>> ax2.set_xlabel("Parameter Index", fontsize=12)
            >>> ax2.set_ylabel("Edge Index", fontsize=12)
            >>> plt.show()
        """
        if "invProp" in kwargs:
            warnings.warn(
                "The invProp keyword argument has been deprecated, please use invert_model. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            invert_model = kwargs["invProp"]
        if "invMat" in kwargs:
            warnings.warn(
                "The invMat keyword argument has been deprecated, please use invert_matrix. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            invert_matrix = kwargs["invMat"]
        if "doFast" in kwargs:
            warnings.warn(
                "The doFast keyword argument has been deprecated, please use do_fast. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            do_fast = kwargs["doFast"]
        return self._getInnerProductDeriv(
            model,
            "E",
            do_fast=do_fast,
            invert_model=invert_model,
            invert_matrix=invert_matrix,
        )

    def _getInnerProductDeriv(
        self,
        model,
        projection_type,
        do_fast=True,
        invert_model=False,
        invert_matrix=False,
    ):
        """
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
        scipy.sparse.csr_matrix
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
        """
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
            for i, p in enumerate(P):
                dMdm = dMdm + sp.csr_matrix(
                    (p.T * (p * v), (range(n), np.zeros(n))), shape=(n, 1)
                )
        if d == 1:
            if tensorType == 1:
                dMdm = spzeros(n, self.nC)
                for i, p in enumerate(P):
                    dMdm = dMdm + p.T * sdiag(p * v)
        elif d == 2:
            if tensorType == 1:
                dMdm = spzeros(n, self.nC)
                for i, p in enumerate(P):
                    Y = p * v
                    y1 = Y[: self.nC]
                    y2 = Y[self.nC :]
                    dMdm = dMdm + p.T * sp.vstack((sdiag(y1), sdiag(y2)))
            elif tensorType == 2:
                dMdms = [spzeros(n, self.nC) for _ in range(2)]
                for i, p in enumerate(P):
                    Y = p * v
                    y1 = Y[: self.nC]
                    y2 = Y[self.nC :]
                    dMdms[0] = dMdms[0] + p.T * sp.vstack((sdiag(y1), Z))
                    dMdms[1] = dMdms[1] + p.T * sp.vstack((Z, sdiag(y2)))
                dMdm = sp.hstack(dMdms)
            elif tensorType == 3:
                dMdms = [spzeros(n, self.nC) for _ in range(3)]
                for i, p in enumerate(P):
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
                for i, p in enumerate(P):
                    Y = p * v
                    y1 = Y[: self.nC]
                    y2 = Y[self.nC : self.nC * 2]
                    y3 = Y[self.nC * 2 :]
                    dMdm = dMdm + p.T * sp.vstack((sdiag(y1), sdiag(y2), sdiag(y3)))
            elif tensorType == 2:
                dMdms = [spzeros(n, self.nC) for _ in range(3)]
                for i, p in enumerate(P):
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
                for i, p in enumerate(P):
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
        """Returns a function for creating projection matrices"""
        ii = np.arange(M.shape_cells[0])

        def Px(xFace):
            """
            xFace is 'fXp' or 'fXm'
            """
            posFx = 0 if xFace == "fXm" else 1
            IND = ii + posFx
            PX = sp.csr_matrix((np.ones(M.nC), (range(M.nC), IND)), shape=(M.nC, M.nF))
            return PX

        return Px

    def _getFacePxx(M):
        """returns a function for creating projection matrices

        Mats takes you from faces a subset of all faces on only the
        faces that you ask for.

        These are centered around a single nodes.

        For example, if this was your entire mesh:

                        f3(Yp)
                  2_______________3
                  |               |
                  |               |
                  |               |
          f0(Xm)  |       x       |  f1(Xp)
                  |               |
                  |               |
                  |_______________|
                  0               1
                        f2(Ym)

        Pxx('fXm','fYm') = | 1, 0, 0, 0 |
                           | 0, 0, 1, 0 |

        Pxx('fXp','fYm') = | 0, 1, 0, 0 |
                           | 0, 0, 1, 0 |

        """
        i, j = np.arange(M.shape_cells[0]), np.arange(M.shape_cells[1])

        iijj = ndgrid(i, j)
        ii, jj = iijj[:, 0], iijj[:, 1]

        if M._meshType == "Curv":
            fN1 = M.reshape(M.face_normals, "F", "Fx", "M")
            fN2 = M.reshape(M.face_normals, "F", "Fy", "M")

        def Pxx(xFace, yFace):
            """
            xFace is 'fXp' or 'fXm'
            yFace is 'fYp' or 'fYm'
            """
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
        """returns a function for creating projection matrices

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
            """
            xFace is 'fXp' or 'fXm'
            yFace is 'fYp' or 'fYm'
            zFace is 'fZp' or 'fZm'
            """

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
        """Returns a function for creating projection matrices"""

        def Px(xEdge):
            if xEdge != "eX0":
                raise TypeError("xEdge = {0!s}, not eX0".format(xEdge))
            return sp.identity(M.nC)

        return Px

    def _getEdgePxx(M):
        i, j = np.arange(M.shape_cells[0]), np.arange(M.shape_cells[1])

        iijj = ndgrid(i, j)
        ii, jj = iijj[:, 0], iijj[:, 1]

        if M._meshType == "Curv":
            eT1 = M.reshape(M.edge_tangents, "E", "Ex", "M")
            eT2 = M.reshape(M.edge_tangents, "E", "Ey", "M")

        def Pxx(xEdge, yEdge):
            """
            no | node      | e1      | e2
            00 | i  ,j     | i  ,j   | i  ,j
            10 | i+1,j     | i  ,j   | i+1,j
            01 | i  ,j+1   | i  ,j+1 | i  ,j
            11 | i+1,j+1   | i  ,j+1 | i+1,j
            """
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
            """
            no  | node        | e1          | e2          | e3
            000 | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k
            100 | i+1,j  ,k   | i  ,j  ,k   | i+1,j  ,k   | i+1,j  ,k
            010 | i  ,j+1,k   | i  ,j+1,k   | i  ,j  ,k   | i  ,j+1,k
            110 | i+1,j+1,k   | i  ,j+1,k   | i+1,j  ,k   | i+1,j+1,k
            001 | i  ,j  ,k+1 | i  ,j  ,k+1 | i  ,j  ,k+1 | i  ,j  ,k
            101 | i+1,j  ,k+1 | i  ,j  ,k+1 | i+1,j  ,k+1 | i+1,j  ,k
            011 | i  ,j+1,k+1 | i  ,j+1,k+1 | i  ,j  ,k+1 | i  ,j+1,k
            111 | i+1,j+1,k+1 | i  ,j+1,k+1 | i+1,j  ,k+1 | i+1,j+1,k
            """

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

    # DEPRECATED
    getFaceInnerProduct = deprecate_method(
        "get_face_inner_product", "getFaceInnerProduct", removal_version="1.0.0", future_warn=False
    )
    getEdgeInnerProduct = deprecate_method(
        "get_edge_inner_product", "getEdgeInnerProduct", removal_version="1.0.0", future_warn=False
    )
    getFaceInnerProductDeriv = deprecate_method(
        "get_face_inner_product_deriv",
        "getFaceInnerProductDeriv",
        removal_version="1.0.0",
        future_warn=False
    )
    getEdgeInnerProductDeriv = deprecate_method(
        "get_edge_inner_product_deriv",
        "getEdgeInnerProductDeriv",
        removal_version="1.0.0",
        future_warn=False
    )
