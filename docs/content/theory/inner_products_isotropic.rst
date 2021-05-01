.. _inner_products_isotropic:

Isotropic Constitutive Relationships
************************************

Summary
-------

A constitutive relationship quantifies the response of a material to an external stimulus.
Examples include Ohm's law and Hooke's law. For practical applications of the finite volume method,
we may need to take the inner product of expressions containing constitutive relationships.

Let :math:`\vec{v}` and :math:`\vec{w}` be two physically related quantities.
If their relationship is isotropic (defined by a constant :math:`\sigma`),
then the constitutive relation is given by:

.. math::
    \vec{v} = \sigma \vec{w}
    :label: inner_product_isotropic

Here we show that for isotropic constitutive relationships, the inner
product between a vector :math:`\vec{u}` and the right-hand side of
equation :eq:`inner_product_isotropic` is approximated by:

.. math::
    (\vec{u}, \sigma \vec{w} ) = \int_\Omega \vec{u} \cdot \sigma \vec{w} \, dv \approx \boldsymbol{u^T M w}

where :math:`\boldsymbol{M}` represents an *inner-product matrix*, and vectors
:math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` are discrete variables that live
on the mesh. It is important to note a few things about the
inner-product matrix in this case:

    1. It depends on the dimensions and discretization of the mesh
    2. It depends on where the discrete variables live; e.g. edges, faces
    3. It depends on the spacial variation of the material property :math:`\sigma`

For this class of inner products, the corresponding inner product matricies for
discrete quantities living on various parts of the mesh are given by:

.. math::
    \textrm{Vectors on faces:} \; \boldsymbol{M_{\sigma f}} &= \frac{1}{4} \boldsymbol{P_f^T } \textrm{diag} \boldsymbol{\big ( e_k \otimes (v \odot \sigma ) \big )} \boldsymbol{P_f} \\
    \textrm{Vectors on edges:} \; \boldsymbol{M_{\sigma e}} &= \frac{1}{4^{k-1}} \boldsymbol{P_e^T } \textrm{diag} \boldsymbol{\big ( e_k \otimes (v \odot \sigma) \big )} \boldsymbol{P_e}

where

    - :math:`k = 1,2,3` represent the dimension (1D, 2D or 3D)
    - :math:`\boldsymbol{e_k}` is a vector of 1s of length :math:`k`
    - :math:`\odot` is the Hadamard product
    - :math:`\otimes` is the kronecker product
    - :math:`\boldsymbol{P_f}` and :math:`\boldsymbol{P_e}` are projection matricies that map quantities from faces and edges to cell centers, respectively
    - :math:`\boldsymbol{v}` is a vector that stores all of the volumes of the cells
    - :math:`\boldsymbol{\sigma}` is a vector containing the physical property values for the cells


**Tutorial:** To construct the inner product matrix and/or approximate inner products of this type, see the :ref:`tutorial on inner products with constitutive relationships <sphx_glr_tutorials_inner_products_2_physical_properties.py>`

.. _inner_products_isotropic_faces:

Vectors on Cell Faces
---------------------

We want to approximate the inner product between a vector quantity :math:`\vec{u}` and the product of
:math:`\sigma` and :math:`\vec{w}`. Here, we discretize such that :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` are defined
to live on cess faces. Our goal is to construct the inner product matrix :math:`\boldsymbol{M}` in the expression below: 

.. math::
    (\vec{u}, \sigma \vec{w}) = \int_\Omega \vec{u} \cdot \sigma \vec{w} \, dv \approx \boldsymbol{u^T M \, w}
    :label: inner_product_isotropic_faces

We must respect the dot product. For vectors defined on cell faces, we discretize such that the
x-component of the vectors live on the x-faces, the y-component lives y-faces and the z-component
lives on the z-faces. For a single cell, this is illustrated in 2D and 3D below. By decomposing the
domain into a set of finite cells, we assume the physical properties are spacial invariant within each cell.

.. figure:: ../../images/face_discretization.png
    :align: center
    :width: 600

As we can see there are 2 faces for each component. Therefore, we need to project each component of the
vector from its faces to the cell centers and take their averages separately.
For a single cell with volume :math:`v_i` and material property value :math:`\sigma_i`,
the contribution towards the inner product is:

.. math::
    \begin{align}
    \mathbf{In \; 2D:} \; \int_{\Omega_i} \vec{u} \cdot \vec{w} \, dv \approx & \;\; \frac{v_i \sigma_i}{4} \Big ( u_x^{(1)} + u_x^{(2)} \Big ) \Big ( w_x^{(1)} + w_x^{(2)} \Big ) \\
    & + \frac{v_i \sigma_i}{4} \Big ( u_y^{(1)} + u_y^{(2)} \Big ) \Big ( w_y^{(1)} + w_y^{(2)} \Big ) \\
    & \\
    \mathbf{In \; 3D:} \; \int_{\Omega_i} \vec{u} \cdot \vec{w} \, dv \approx & \;\; \frac{v_i \sigma_i}{4} \Big ( u_x^{(1)} + u_x^{(2)} \Big ) \Big ( w_x^{(1)} + w_x^{(2)} \Big ) \\
    & + \frac{v_i \sigma_i}{4} \Big ( u_y^{(1)} + u_y^{(2)} \Big ) \Big ( w_y^{(1)} + w_y^{(2)} \Big ) \\
    & + \frac{v_i \sigma_i}{4} \Big ( u_z^{(1)} + u_z^{(2)} \Big ) \Big ( w_z^{(1)} + w_z^{(2)} \Big )
    \end{align}
    :label: inner_product_isotropic_faces_1

where superscripts :math:`(1)` and :math:`(2)` denote face 1 and face 2, respectively.
Using the contribution for each cell described in expression :eq:`inner_product_isotropic_faces_1`,
we want to approximate the inner product in the form described by
equation :eq:`inner_product_isotropic_faces`. To accomlish this, we construct a sparse matrix
:math:`\boldsymbol{P_f}` which projects quantities on the x, y and z faces separately to the
the cell centers.

For discretize vectors :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` whose x, y (and z) components
are organized on cell faces as follows:

.. math::
    \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_y} \\ \end{bmatrix}
    \;\;\;\; \textrm{and} \;\;\;\;
    \boldsymbol{w} = \begin{bmatrix} \boldsymbol{e_x} \\ \boldsymbol{e_y} \\ \boldsymbol{e_y} \\ \end{bmatrix}

the approximation to the inner product is given by:

.. math::
     (\vec{u}, \sigma \vec{w}) = \int_\Omega \vec{u} \cdot \sigma \vec{w} \, dv \approx \boldsymbol{\boldsymbol{u} M_{\sigma f}} \, \boldsymbol{w}

where the mass matrix has the form:

.. math::
    \boldsymbol{M_{\sigma f}} = \frac{1}{4} \boldsymbol{P_f^T } \textrm{diag} \boldsymbol{\big ( e_k \otimes (v \odot \sigma ) \big )} \boldsymbol{P_f}

and

    - :math:`k = 1,2,3` represent the dimension (1D, 2D or 3D)
    - :math:`\boldsymbol{e_k}` is a vector of 1s of length :math:`k`
    - :math:`\odot` is the Hadamard product
    - :math:`\otimes` is the kronecker product
    - :math:`\boldsymbol{P_f}` is a projection matrix that maps from faces to cell centers
    - :math:`\boldsymbol{v}` is a vector that stores all of the volumes of the cells
    - :math:`\boldsymbol{\sigma}` is a vector containing the physical property values for the cells

.. _inner_products_isotropic_edges:

Vectors on Cell Edges
---------------------

We want to approximate the inner product between a vector quantity :math:`\vec{u}` and the product of
:math:`\sigma` and :math:`\vec{w}`. Here, we discretize such that :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` are defined
to live at cell edges. Our goal is to construct the inner product matrix :math:`\boldsymbol{M}` in the expression below: 

.. math::
    (\vec{u}, \sigma \vec{w}) = \int_\Omega \vec{u} \cdot \sigma \vec{w} \, dv \approx \boldsymbol{u^T M \, w}
    :label: inner_product_isotropic_edges

We must respect the dot product. For vectors defined on cell edges, we discretize such that the
x-component of the vectors live on the x-edges, the y-component lives y-edges and the z-component
lives on the z-edges. This is illustrated in 2D and 3D below. By decomposing the
domain into a set of finite cells, we assume the material properties are spacial invariant within each cell.

.. figure:: ../../images/edge_discretization.png
    :align: center
    :width: 600

As we can see there are 2 edges for each component in 2D and 4 edges for each component in 3D.
Therefore, we need to project each component of the
vector from its edges to the cell centers and take their averages separately. For a single cell with volume :math:`v_i`
and material property value :math:`\sigma_i`, the contribution towards the inner product is:

.. math::
    \begin{align}
    \mathbf{In \; 2D:} \; \int_{\Omega_i} \vec{u} \cdot \vec{w} \, dv \approx & \;\; \frac{v_i \sigma_i}{4} \Big ( u_x^{(1)} + u_x^{(2)} \Big ) \Big ( w_x^{(1)} + w_x^{(2)} \Big ) \\
    & + \frac{v_i \sigma_i}{4} \Big ( u_y^{(1)} + u_y^{(2)} \Big ) \Big ( w_y^{(1)} + w_y^{(2)} \Big ) \\
    & \\
    \mathbf{In \; 3D:} \; \int_{\Omega_i} \vec{u} \cdot \vec{w} \, dv \approx & \;\; \frac{v_i \sigma_i}{16} \Bigg ( \sum_{n=1}^4 u_x^{(n)} \Bigg ) \Bigg ( \sum_{n=1}^4 w_x^{(n)} \Bigg ) \\
    & + \frac{v_i \sigma_i}{16} \Bigg ( \sum_{n=1}^4 u_y^{(n)} \Bigg ) \Bigg ( \sum_{n=1}^4 w_y^{(n)} \Bigg ) \\
    & + \frac{v_i \sigma_i}{16} \Bigg ( \sum_{n=1}^4 u_z^{(n)} \Bigg ) \Bigg ( \sum_{n=1}^4 w_z^{(n)} \Bigg )
    \end{align}
    :label: inner_product_isotropic_edges_1

where the superscript :math:`(n)` denotes a particular edge.
Using the contribution for each cell described in expression :eq:`inner_product_isotropic_edges_1`,
we want to approximate the inner product in the form described by
equation :eq:`inner_product_isotropic_edges`. To accomlish this, we construct a sparse matrix
:math:`\boldsymbol{P_e}` which projects quantities on the x, y and z edges separately to the
the cell centers.

For discretize vectors :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` whose x, y (and z) components
are organized on cell edges as follows:

.. math::
    \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_y} \\ \end{bmatrix}
    \;\;\;\; \textrm{and} \;\;\;\;
    \boldsymbol{w} = \begin{bmatrix} \boldsymbol{e_x} \\ \boldsymbol{e_y} \\ \boldsymbol{e_y} \\ \end{bmatrix}

the approximation to the inner product is given by:

.. math::
     (\vec{u}, \vec{w}) = \int_\Omega \vec{u} \cdot \vec{w} \, dv \approx \boldsymbol{\boldsymbol{u} M_e \, \boldsymbol{w}}

where the mass matrix for face quantities has the form:

.. math::
    \boldsymbol{M_{\sigma e}} = \frac{1}{4^{k-1}} \boldsymbol{P_e^T } \textrm{diag} \boldsymbol{\big ( e_k \otimes (v \odot \sigma) \big )} \boldsymbol{P_e}

and

    - :math:`k = 1,2,3` represent the dimension (1D, 2D or 3D)
    - :math:`\boldsymbol{e_k}` is a vector of 1s of length :math:`k`
    - :math:`\odot` is the Hadamard product
    - :math:`\otimes` is the kronecker product
    - :math:`\boldsymbol{P_e}` is a projection matrix that maps from edges to cell centers
    - :math:`\boldsymbol{v}` is a vector that stores all of the volumes of the cells
    - :math:`\boldsymbol{\sigma}` is a vector containing the physical property values for the cells

.. _inner_products_isotropic_reciprocal:


Reciprocal Properties
---------------------

Let :math:`\vec{v}` and :math:`\vec{w}` be two physically related quantities.
If their relationship is isotropic and defined by the reciprocal of a physical property (defined by a constant :math:`\rho`),
then the constitutive relation is given by:

.. math::
    \vec{v} = \rho^{-1} \vec{w}
    :label: inner_product_isotropic_reciprocal

Because the relationship between :math:`\vec{v}` and :math:`\vec{w}` is a constant,
the derivation of the inner-product matrix at edges and faces is effectively the same.
For this case, the corresponding inner product matricies for
discrete quantities living on various parts of the mesh are given by:

.. math::
    \textrm{Vectors on faces:} \; \boldsymbol{M_{\rho f}} &= \frac{1}{4} \boldsymbol{P_f^T } \textrm{diag} \boldsymbol{\big ( e_k \otimes (v \odot \rho^{-1} ) \big )} \boldsymbol{P_f} \\
    \textrm{Vectors on edges:} \; \boldsymbol{M_{\rho e}} &= \frac{1}{4^{k-1}} \boldsymbol{P_e^T } \textrm{diag} \boldsymbol{\big ( e_k \otimes (v \odot \rho^{-1}) \big )} \boldsymbol{P_e}

where

    - :math:`k = 1,2,3` represent the dimension (1D, 2D or 3D)
    - :math:`\boldsymbol{e_k}` is a vector of 1s of length :math:`k`
    - :math:`\odot` is the Hadamard product
    - :math:`\otimes` is the kronecker product
    - :math:`\boldsymbol{P_f}` and :math:`\boldsymbol{P_e}` are projection matricies that map quantities from faces and edges to cell centers, respectively
    - :math:`\boldsymbol{v}` is a vector that stores all of the volumes of the cells
    - :math:`\boldsymbol{\rho^{-1}}` is a vector containing the reciprocal of :math:`\rho` for all cells


**Tutorial:** To construct the inner product matrix and/or approximate inner products of this type, see the :ref:`tutorial on inner products with constitutive relationships <sphx_glr_tutorials_inner_products_2_physical_properties.py>`

