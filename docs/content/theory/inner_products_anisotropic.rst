.. _inner_products_anisotropic:

Anisotropic Constitutive Relationships
**************************************

Summary
-------

A constitutive relationship quantifies the response of a material to an external stimulus.
Examples include Ohm's law and Hooke's law. For practical applications of the finite volume method,
we may need to take the inner product of expressions containing constitutive relationships.

Let :math:`\vec{v}` and :math:`\vec{w}` be two physically related quantities.
If their relationship is anisotropic (defined by a tensor :math:`\Sigma`), the constitutive relation is of the form:

.. math::
    \vec{v} = \Sigma \vec{w}
    :label: inner_product_anisotropic

where

.. math::
    \mathbf{In \; 2D:} \; 
    \Sigma = \begin{bmatrix} \sigma_{xx} & \sigma_{xy} \\
    \sigma_{yx} & \sigma_{yy} \end{bmatrix}
    \;\;\;\;\;\;\;\; \mathbf{In \; 3D:} \; 
    \Sigma = \begin{bmatrix} \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
    \sigma_{yx} & \sigma_{yy} & \sigma_{yz} \\
    \sigma_{zx} & \sigma_{zy} & \sigma_{zz} \end{bmatrix}

Note that for real materials, the tensor is symmetric and has 6 independent variables
(i.e. :math:`\sigma_{pq}=\sigma_{qp}` for :math:`p,q=x,y,z`).
Here we show that for anisotropic constitutive relationships, the inner
product between a vector :math:`\vec{u}` and the right-hand side of
equation :eq:`inner_product_anisotropic` is approximated by:

.. math::
    (\vec{u}, \Sigma \vec{w} ) = \int_\Omega \vec{u} \cdot \Sigma \vec{w} \, dv \approx \boldsymbol{u^T M w}
    :label: inner_product_anisotropic_general

where :math:`\boldsymbol{M}` represents an *inner-product matrix*, and vectors
:math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` are discrete variables that live
on the mesh. It is important to note a few things about the
inner-product matrix in this case:

    1. It depends on the dimensions and discretization of the mesh
    2. It depends on where the discrete variables live; e.g. edges, faces, nodes, centers
    3. It depends on the spacial variation of the tensor property :math:`\Sigma`

For this class of inner products, the corresponding form of the inner product matricies for
discrete quantities living on various parts of the mesh are shown below.

**Tutorial:** To construct the inner product matrix and/or approximate inner products of this type, see the :ref:`tutorial on inner products with constitutive relationships <sphx_glr_tutorials_inner_products_2_physical_properties.py>`

**Diagonal Anisotropic:**

For the diagonal anisotropic case, the tensor characterzing the material properties
has the form:

.. math::
    \mathbf{In \; 2D:} \; 
    \Sigma = \begin{bmatrix} \sigma_{x} & 0 \\
    0 & \sigma_{y} \end{bmatrix}
    \;\;\;\;\;\;\;\; \mathbf{In \; 3D:} \; 
    \Sigma = \begin{bmatrix} \sigma_{x} & 0 & 0 \\
    0 & \sigma_{y} & 0 \\
    0 & 0 & \sigma_{z} \end{bmatrix}
    :label: inner_product_tensor_diagonal


The inner product matrix defined in expression :eq:`inner_product_anisotropic_general` is given by:

.. math::
    \textrm{Vectors on faces:} \; \boldsymbol{M_{\Sigma f}} &= \frac{1}{4} \boldsymbol{P_f^T } \textrm{diag} \boldsymbol{\big ( (e_k \otimes v) \odot \sigma \big )} \boldsymbol{P_f} \\
    \textrm{Vectors on edges:} \; \boldsymbol{M_{\Sigma e}} &= \frac{1}{4^{k-1}} \boldsymbol{P_e^T } \textrm{diag} \boldsymbol{\big ( (e_k \otimes v) \odot \sigma \big )} \boldsymbol{P_e}

where :math:`\boldsymbol{\sigma}` organizes vectors :math:`\boldsymbol{\sigma_x}`,
:math:`\boldsymbol{\sigma_y}` and :math:`\boldsymbol{\sigma_z}` as:

.. math::
    \boldsymbol{\sigma} = \begin{bmatrix} \boldsymbol{\sigma_x} \\ \boldsymbol{\sigma_y} \\ \boldsymbol{\sigma_z} \\ \end{bmatrix}

and

    - :math:`k = 1,2,3` represent the dimension (1D, 2D or 3D)
    - :math:`\boldsymbol{e_k}` is a vector of 1s of length :math:`k`
    - :math:`\odot` is the Hadamard product
    - :math:`\otimes` is the kronecker product
    - :math:`\boldsymbol{v}` is a vector that stores all of the volumes of the cells
    - :math:`\boldsymbol{P_f}` and :math:`\boldsymbol{P_e}` are projection matricies that map quantities from faces and edges to cell centers, respectively

**Fully Anisotropic:**

For a fully anisotropic case, the tensor characterizing the material properties
has the form is given by:

.. math::
    \mathbf{In \; 2D:} \; 
    \Sigma = \begin{bmatrix} \sigma_{xx} & \sigma_{xy} \\
    \sigma_{yx} & \sigma_{yy} \end{bmatrix}
    \;\;\;\;\;\;\;\; \mathbf{In \; 3D:} \; 
    \Sigma = \begin{bmatrix} \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
    \sigma_{yx} & \sigma_{yy} & \sigma_{yz} \\
    \sigma_{zx} & \sigma_{zy} & \sigma_{zz} \end{bmatrix}
    :label: inner_product_tensor

The inner product matrix defined in expression :eq:`inner_product_anisotropic_general` is given by:

.. math::
    \textrm{Vectors on faces:} \; \boldsymbol{M_{\Sigma f}} &= \frac{1}{4} \boldsymbol{P_f^T Q_u^T} \textrm{diag} \boldsymbol{\big ( (e_k \otimes v) \odot \sigma \big )} \boldsymbol{Q_w P_f} \\
    \textrm{Vectors on edges:} \; \boldsymbol{M_{\Sigma e}} &= \frac{1}{4^{k-1}} \boldsymbol{P_e^T Q_u^T} \textrm{diag} \boldsymbol{\big ( (e_k \otimes v) \odot \sigma \big )} \boldsymbol{Q_w P_e}

where :math:`\boldsymbol{\sigma}` is a large vector that organizes vectors :math:`\boldsymbol{\sigma_{pq}}` for :math:`p,q=x,y,z` as:

.. math::
    \boldsymbol{\sigma} = \begin{bmatrix}
    \boldsymbol{\sigma_{xx}} , \; \boldsymbol{\sigma_{xy}} , \; \boldsymbol{\sigma_{xz}} , \;
    \boldsymbol{\sigma_{yx}} , \; \boldsymbol{\sigma_{yy}} , \; \boldsymbol{\sigma_{yz}} , \;
    \boldsymbol{\sigma_{zx}} , \; \boldsymbol{\sigma_{zy}} , \; \boldsymbol{\sigma_{zz}} \end{bmatrix}^T

and

    - :math:`k = 1,2,3` represent the dimension (1D, 2D or 3D)
    - :math:`\boldsymbol{e_k}` is a vector of 1s of length :math:`k`
    - :math:`\odot` is the Hadamard product
    - :math:`\otimes` is the kronecker product
    - :math:`\boldsymbol{v}` is a vector that stores all of the volumes of the cells
    - :math:`\boldsymbol{P_f}` and :math:`\boldsymbol{P_e}` are projection matricies that map quantities from faces and edges to cell centers, respectively
    - :math:`\boldsymbol{Q_u}` is a sparse replication matrix that augments a vector of the form :math:`[\boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}]^T` to create a vector of the form :math:`[\boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}, \; \boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}, \; \boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z} ]^T`
    - :math:`\boldsymbol{Q_w}` is a sparse replication matrix that augments a vector of the form :math:`[\boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}]^T` to create a vector of the form :math:`[\boldsymbol{u_x}, \; \boldsymbol{u_x}, \; \boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_y}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}, \; \boldsymbol{u_z} \; \boldsymbol{u_z} ]^T`


Diagonally Anisotropic Case
---------------------------

Vectors on Cell Faces
^^^^^^^^^^^^^^^^^^^^^

We want to approximate the inner product between a vector quantity :math:`\vec{u}` and the product of
:math:`\Sigma` and :math:`\vec{w}`, where :math:`\Sigma` given in expression :eq:`inner_product_tensor_diagonal`.
Here, we discretize such that :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` are defined
to live on cell faces. Our goal is to construct the inner product matrix :math:`\boldsymbol{M}` such that:

.. math::
    (\vec{u}, \Sigma \vec{w}) = \int_\Omega \vec{u} \cdot \Sigma \vec{w} \, dv \approx \boldsymbol{u^T \, M \, w}
    :label: inner_product_anisotropic_faces

We must respect the dot product and the tensor. For vectors defined on cell faces, we discretize such that the
x-component of the vectors live on the x-faces, the y-component lives y-faces and the z-component
lives on the z-faces. For a single cell, this is illustrated in 2D and 3D below. By decomposing the
domain into a set of finite cells, we assume the tensor properties are spacial invariant within each cell.

.. figure:: ../../images/face_discretization.png
    :align: center
    :width: 600

As we can see there are 2 faces for each component. Therefore, we need to project each component of the
vector from its faces to the cell centers and take their averages separately. We must also recognize that
x-components are only multiplied by :math:`\sigma_x`, y-components by :math:`\sigma_y` and z-components
by :math:`\sigma_z`.

For a single cell :math:`i` with volume :math:`v` and tensor properties defined by
:math:`\sigma_x`, :math:`\sigma_y`, :math:`\sigma_z`
the contribution towards the inner product is:

.. math::
    \begin{align}
    \mathbf{In \; 2D:} \; \int_{\Omega_i} \vec{u} \cdot \vec{w} \, dv \approx & \;\; \frac{v}{4}
    \sum_{p=x,y} \sigma_{p} \Big ( u_p^{(1)} + u_p^{(2)} \Big ) \Big ( w_p^{(1)} + w_p^{(2)} \Big ) \\
    & \\
    \mathbf{In \; 3D:} \; \int_{\Omega_i} \vec{u} \cdot \vec{w} \, dv \approx & \;\; \frac{v}{4}
    \sum_{p=x,y,z} \sigma_{p} \Big ( u_p^{(1)} + u_p^{(2)} \Big ) \Big ( w_p^{(1)} + w_p^{(2)} \Big )
    \end{align}
    :label: inner_product_anisotropic_faces_1

where superscripts :math:`(1)` and :math:`(2)` denote face 1 and face 2, respectively.
Using the contribution for each cell described in expression :eq:`inner_product_anisotropic_faces_1`,
we want to approximate the inner product in the form described by
equation :eq:`inner_product_anisotropic_faces`. To accomlish this, we construct a sparse matrix
:math:`\boldsymbol{P_f}` which projects quantities on the x, y and z faces separately to the
the cell centers.

For discretize vectors :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` whose x, y (and z) components
are organized on cell faces as follows:

.. math::
    \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_y} \\ \end{bmatrix}
    \;\;\;\; \textrm{and} \;\;\;\;
    \boldsymbol{w} = \begin{bmatrix} \boldsymbol{w_x} \\ \boldsymbol{w_y} \\ \boldsymbol{w_y} \\ \end{bmatrix}

the approximation to the inner product is given by:

.. math::
     (\vec{u}, \Sigma \vec{w}) = \int_\Omega \vec{u} \cdot \Sigma \vec{w} \, dv \approx \boldsymbol{\boldsymbol{u} \, M_{\Sigma f}} \, \boldsymbol{w}

The inner product matrix defined in the previous expression is given by:

.. math::
    \boldsymbol{M_{\Sigma f}} = \frac{1}{4} \boldsymbol{P_f^T } \textrm{diag} \boldsymbol{\big ( (e_k \otimes v) \odot \sigma \big )} \boldsymbol{P_f}

where :math:`\boldsymbol{\sigma}` organizes vectors :math:`\boldsymbol{\sigma_x}`,
:math:`\boldsymbol{\sigma_y}` and :math:`\boldsymbol{\sigma_z}` as:

.. math::
    \boldsymbol{\sigma} = \begin{bmatrix} \boldsymbol{\sigma_x} \\ \boldsymbol{\sigma_y} \\ \boldsymbol{\sigma_z} \\ \end{bmatrix}

and

    - :math:`k = 1,2,3` represent the dimension (1D, 2D or 3D)
    - :math:`\boldsymbol{e_k}` is a vector of 1s of length :math:`k`
    - :math:`\odot` is the Hadamard product
    - :math:`\otimes` is the kronecker product
    - :math:`\boldsymbol{v}` is a vector that stores all of the volumes of the cells
    - :math:`\boldsymbol{P_f}` is a projection matrix that maps quantities from faces to cell centers

Vectors on Cell Edges
^^^^^^^^^^^^^^^^^^^^^

We want to approximate the inner product between a vector quantity :math:`\vec{u}` and the product of
:math:`\Sigma` and :math:`\vec{w}`, where :math:`\Sigma` given in expression :eq:`inner_product_tensor_diagonal`.
Here, we discretize such that :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` are defined
to live on cell edges. Our goal is to construct the inner product matrix :math:`\boldsymbol{M}` such that:

.. math::
    (\vec{u}, \Sigma \vec{w}) = \int_\Omega \vec{u} \cdot \Sigma \vec{w} \, dv \approx \boldsymbol{u^T \, M \, w}
    :label: inner_product_anisotropic_edges

We must respect the dot product and the tensor. For vectors defined on cell edges, we discretize such that the
x-component of the vectors live on the x-edges, the y-component lives y-edges and the z-component
lives on the z-edges. This is illustrated in 2D and 3D below. By decomposing the
domain into a set of finite cells, we assume the tensor properties are spacial invariant within each cell.

.. figure:: ../../images/edge_discretization.png
    :align: center
    :width: 600

As we can see there are 2 edges for each component in 2D and 4 edges for each component in 3D.
Therefore, we need to project each component of the
vector from its edges to the cell centers and take their averages separately.
We must also recognize that
x-components are only multiplied by :math:`\sigma_x`, y-components by :math:`\sigma_y` and z-components
by :math:`\sigma_z`.

For a single cell :math:`i` with volume :math:`v` and tensor properties defined by
:math:`\sigma_x`, :math:`\sigma_y`, :math:`\sigma_z`
the contribution towards the inner product is:

.. math::
    \begin{align}
    \mathbf{In \; 2D:} \; \int_{\Omega_i} \vec{u} \cdot \vec{w} \, dv \approx & \;\; \frac{v}{4}
    \sum_{p=x,y} \sigma_{p} \Big ( u_p^{(1)} + u_p^{(2)} \Big ) \Big ( w_p^{(1)} + w_p^{(2)} \Big ) \\
    & \\
    \mathbf{In \; 3D:} \; \int_{\Omega_i} \vec{u} \cdot \vec{w} \, dv \approx & \;\; \frac{v}{16}
    \sum_{p=x,y,z} \sigma_{p} \Big ( u_p^{(1)} + u_p^{(2)} + u_p^{(3)} + u_p^{(4)} \Big )
    \Big ( w_p^{(1)} + w_p^{(2)} + w_p^{(3)} + w_p^{(4)} \Big )
    \end{align}
    :label: inner_product_anisotropic_edges_1

where the superscripts :math:`(1)` to :math:`(4)` denote a particular edges.
Using the contribution for each cell described in expression :eq:`inner_product_anisotropic_edges_1`,
we want to approximate the inner product in the form described by
equation :eq:`inner_product_anisotropic_edges`. To accomlish this, we construct a sparse matrix
:math:`\boldsymbol{P_e}` which projects quantities on the x, y and z edges separately to the
the cell centers.

For discretize vectors :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` whose x, y (and z) components
are organized on cell edges as follows:

.. math::
    \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_y} \\ \end{bmatrix}
    \;\;\;\; \textrm{and} \;\;\;\;
    \boldsymbol{w} = \begin{bmatrix} \boldsymbol{w_x} \\ \boldsymbol{w_y} \\ \boldsymbol{w_y} \\ \end{bmatrix}

the approximation to the inner product is given by:

.. math::
     (\vec{u}, \Sigma \vec{w}) = \int_\Omega \vec{u} \cdot \vec{w} \, dv \approx \boldsymbol{\boldsymbol{u} \, M_{\Sigma e} \, \boldsymbol{w}}

The inner product matrix defined in the previous expression is given by:

.. math::
    \boldsymbol{M_{\Sigma e}} = \frac{1}{4^{k-1}} \boldsymbol{P_e^T } \textrm{diag} \boldsymbol{\big ( (e_k \otimes v) \odot \sigma \big )} \boldsymbol{P_e}

where :math:`\boldsymbol{\sigma}` organizes vectors :math:`\boldsymbol{\sigma_x}`,
:math:`\boldsymbol{\sigma_y}` and :math:`\boldsymbol{\sigma_z}` as:

.. math::
    \boldsymbol{\sigma} = \begin{bmatrix} \boldsymbol{\sigma_x} \\ \boldsymbol{\sigma_y} \\ \boldsymbol{\sigma_z} \\ \end{bmatrix}
and

    - :math:`k = 1,2,3` represent the dimension (1D, 2D or 3D)
    - :math:`\boldsymbol{e_k}` is a vector of 1s of length :math:`k`
    - :math:`\odot` is the Hadamard product
    - :math:`\otimes` is the kronecker product
    - :math:`\boldsymbol{v}` is a vector that stores all of the volumes of the cells
    - :math:`\boldsymbol{P_e}` is a projection matrix that maps quantities from edges to cell centers

Fully Anisotropic Case
----------------------

Vectors on Cell Faces
^^^^^^^^^^^^^^^^^^^^^

We want to approximate the inner product between a vector quantity :math:`\vec{u}` and the product of
:math:`\Sigma` and :math:`\vec{w}`, where :math:`\Sigma` given in expression :eq:`inner_product_tensor`.
Here, we discretize such that :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` are defined
to live on cell faces. Our goal is to construct the inner product matrix :math:`\boldsymbol{M}` such that: 

.. math::
    (\vec{u}, \Sigma \vec{w}) = \int_\Omega \vec{u} \cdot \Sigma \vec{w} \, dv \approx \boldsymbol{u^T \, M \, e}
    :label: inner_product_anisotropic_faces

We must respect the dot product and the tensor. For vectors defined on cell faces, we discretize such that the
x-component of the vectors live on the x-faces, the y-component lives y-faces and the z-component
lives on the z-faces. For a single cell, this is illustrated in 2D and 3D below. By decomposing the
domain into a set of finite cells, we assume the tensor properties are spacial invariant within each cell.

.. figure:: ../../images/face_discretization.png
    :align: center
    :width: 600

As we can see there are 2 faces for each component. Therefore, we need to project each component of the
vector from its faces to the cell centers and take their averages separately. We must also recognize that
different parameters :math:`\sigma_{pq}` for :math:`p,q=x,y,z` multiply different components of the vectors.

For a single cell :math:`i` with volume :math:`v` and tensor properties defined by
:math:`\sigma_{pq}` for :math:`p,q=x,y,z`,
the contribution towards the inner product is:

.. math::
    \begin{align}
    \mathbf{In \; 2D:} \; \int_{\Omega_i} \vec{u} \cdot \vec{w} \, dv \approx & \;\; \frac{v}{4}
    \sum_{p,q=x,y} \sigma_{pq} \Big ( u_p^{(1)} + u_p^{(2)} \Big ) \Big ( w_q^{(1)} + w_q^{(2)} \Big ) \\
    & \\
    \mathbf{In \; 3D:} \; \int_{\Omega_i} \vec{u} \cdot \vec{w} \, dv \approx & \;\; \frac{v}{4}
    \sum_{p,q=x,y,z} \sigma_{pq} \Big ( u_p^{(1)} + u_p^{(2)} \Big ) \Big ( w_q^{(1)} + w_q^{(2)} \Big )
    \end{align}
    :label: inner_product_anisotropic_faces_1

where superscripts :math:`(1)` and :math:`(2)` denote face 1 and face 2, respectively.
Using the contribution for each cell described in expression :eq:`inner_product_anisotropic_faces_1`,
we want to approximate the inner product in the form described by
equation :eq:`inner_product_anisotropic_faces`. To accomlish this, we construct a sparse matrix
:math:`\boldsymbol{P_f}` which projects quantities on the x, y and z faces separately to the
the cell centers.

For discretize vectors :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` whose x, y (and z) components
are organized on cell faces as follows:

.. math::
    \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_z} \\ \end{bmatrix}
    \;\;\;\; \textrm{and} \;\;\;\;
    \boldsymbol{w} = \begin{bmatrix} \boldsymbol{w_x} \\ \boldsymbol{w_y} \\ \boldsymbol{w_z} \\ \end{bmatrix}

the approximation to the inner product is given by:

.. math::
     (\vec{u}, \Sigma \vec{w}) = \int_\Omega \vec{u} \cdot \Sigma \vec{w} \, dv \approx \boldsymbol{\boldsymbol{u} \, M_{\Sigma f}} \, \boldsymbol{w}

The inner product matrix defined in the previous expression is given by:

.. math::
    \boldsymbol{M_{\Sigma f}} = \frac{1}{4} \boldsymbol{P_f^T Q_u^T} \textrm{diag} \boldsymbol{\big ( (e_k \otimes e_k \otimes v) \odot \sigma \big )} \boldsymbol{Q_w P_f}

where :math:`\boldsymbol{\sigma}` is a large vector that organizes vectors :math:`\boldsymbol{\sigma_{pq}}` for :math:`p,q=x,y,z` as:

.. math::
    \boldsymbol{\sigma} = \begin{bmatrix}
    \boldsymbol{\sigma_{xx}} , \; \boldsymbol{\sigma_{xy}} , \; \boldsymbol{\sigma_{xz}} , \;
    \boldsymbol{\sigma_{yx}} , \; \boldsymbol{\sigma_{yy}} , \; \boldsymbol{\sigma_{yz}} , \;
    \boldsymbol{\sigma_{zx}} , \; \boldsymbol{\sigma_{zy}} , \; \boldsymbol{\sigma_{zz}} \end{bmatrix}^T

and

    - :math:`k = 1,2,3` represent the dimension (1D, 2D or 3D)
    - :math:`\boldsymbol{e_k}` is now a vector of 1s of length :math:`k`
    - :math:`\odot` is the Hadamard product
    - :math:`\otimes` is the kronecker product
    - :math:`\boldsymbol{P_f}` is a projection matrix that maps quantities from faces to cell centers
    - :math:`\boldsymbol{v}` is a vector that stores all of the volumes of the cells
    - :math:`\boldsymbol{Q_u}` is a sparse replication matrix that augments a vector of the form :math:`[\boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}]^T` to create a vector of the form :math:`[\boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}, \; \boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}, \; \boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z} ]^T`
    - :math:`\boldsymbol{Q_w}` is a sparse replication matrix that augments a vector of the form :math:`[\boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}]^T` to create a vector of the form :math:`[\boldsymbol{u_x}, \; \boldsymbol{u_x}, \; \boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_y}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}, \; \boldsymbol{u_z} \; \boldsymbol{u_z} ]^T`


Vectors on Cell Edges
^^^^^^^^^^^^^^^^^^^^^

We want to approximate the inner product between a vector quantity :math:`\vec{u}` and the product of
:math:`\Sigma` and :math:`\vec{w}`, where :math:`\Sigma` given in expression :eq:`inner_product_tensor`.
Here, we discretize such that :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` are defined
to live on cell edges. Our goal is to construct the inner product matrix :math:`\boldsymbol{M}` such that: 

.. math::
    (\vec{u}, \Sigma \vec{w}) = \int_\Omega \vec{u} \cdot \Sigma \vec{w} \, dv \approx \boldsymbol{u^T \, M \, w}
    :label: inner_product_anisotropic_edges

where :math:`\Sigma` is defined in expression :eq:`inner_product_tensor`.
We must respect the dot product and the tensor. For vectors defined on cell edges, we discretize such that the
x-component of the vectors live on the x-edges, the y-component lives y-edges and the z-component
lives on the z-edges. This is illustrated in 2D and 3D below. By decomposing the
domain into a set of finite cells, we assume the tensor properties are spacial invariant within each cell.

.. figure:: ../../images/edge_discretization.png
    :align: center
    :width: 600

As we can see there are 2 edges for each component in 2D and 4 edges for each component in 3D.
Therefore, we need to project each component of the vector from its edges to the cell centers and take their averages separately.
Since the tensor is symmetric, it has 3 independent components in 2D and 6 independent components in 3D.
Using this, we can reduce the size of the computation.

For a single cell :math:`i` with volume :math:`v` and tensor properties defined by
:math:`\sigma_{pq}` for :math:`p,q=x,y,z`,
the contribution towards the inner product is:

.. math::
    \begin{align}
    \mathbf{In \; 2D:} \; \int_{\Omega_i} \vec{u} \cdot \vec{w} \, dv \approx & \;\; \frac{v}{4}
    \sum_{p,q=x,y} \sigma_{pq} \Big ( u_p^{(1)} + u_p^{(2)} \Big ) \Big ( w_q^{(1)} + w_q^{(2)} \Big ) \\
    & \\
    \mathbf{In \; 3D:} \; \int_{\Omega_i} \vec{u} \cdot \vec{w} \, dv \approx & \;\; \frac{v}{16}
    \sum_{p,q=x,y,z} \sigma_{pq} \Big ( u_p^{(1)} + u_p^{(2)} + u_p^{(3)} + u_p^{(4)} \Big )
    \Big ( w_q^{(1)} + w_q^{(2)} + w_q^{(3)} + w_q^{(4)} \Big )
    \end{align}
    :label: inner_product_anisotropic_edges_1

where the superscripts :math:`(1)` to :math:`(4)` denote a particular edges.
Using the contribution for each cell described in expression :eq:`inner_product_anisotropic_edges_1`,
we want to approximate the inner product in the form described by
equation :eq:`inner_product_anisotropic_edges`. To accomlish this, we construct a sparse matrix
:math:`\boldsymbol{P_e}` which projects quantities on the x, y and z edges separately to the
the cell centers.

For discretize vectors :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` whose x, y (and z) components
are organized on cell edges as follows:

.. math::
    \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_y} \\ \end{bmatrix}
    \;\;\;\; \textrm{and} \;\;\;\;
    \boldsymbol{w} = \begin{bmatrix} \boldsymbol{w_x} \\ \boldsymbol{w_y} \\ \boldsymbol{w_y} \\ \end{bmatrix}

the approximation to the inner product is given by:

.. math::
     (\vec{u}, \Sigma \vec{w}) = \int_\Omega \vec{u} \cdot \vec{w} \, dv \approx \boldsymbol{\boldsymbol{u} \, M_{\Sigma e} \, \boldsymbol{w}}

The inner product matrix defined in the previous expression is given by:

.. math::
    \boldsymbol{M_{\Sigma e}} = \frac{1}{4^{k-1}} \boldsymbol{P_e^T Q_u^T} \textrm{diag} \boldsymbol{\big ( (e_k \otimes e_k \otimes v) \odot \sigma \big )} \boldsymbol{Q_w P_e}

where :math:`\boldsymbol{\sigma}` is a large vector that organizes vectors :math:`\boldsymbol{\sigma_{pq}}` for :math:`p,q=x,y,z` as:

.. math::
    \boldsymbol{\sigma} = \begin{bmatrix}
    \boldsymbol{\sigma_{xx}} , \; \boldsymbol{\sigma_{xy}} , \; \boldsymbol{\sigma_{xz}} , \;
    \boldsymbol{\sigma_{yx}} , \; \boldsymbol{\sigma_{yy}} , \; \boldsymbol{\sigma_{yz}} , \;
    \boldsymbol{\sigma_{zx}} , \; \boldsymbol{\sigma_{zy}} , \; \boldsymbol{\sigma_{zz}} \end{bmatrix}^T

and

    - :math:`k = 1,2,3` represent the dimension (1D, 2D or 3D)
    - :math:`\boldsymbol{e_k}` is a vector of 1s of length :math:`k`
    - :math:`\odot` is the Hadamard product
    - :math:`\otimes` is the kronecker product
    - :math:`\boldsymbol{v}` is a vector that stores all of the volumes of the cells
    - :math:`\boldsymbol{P_e}` is a projection matrix that maps quantities from edges to cell centers
    - :math:`\boldsymbol{Q_u}` is a sparse replication matrix that augments a vector of the form :math:`[\boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}]^T` to create a vector of the form :math:`[\boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}, \; \boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}, \; \boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z} ]^T`
    - :math:`\boldsymbol{Q_w}` is a sparse replication matrix that augments a vector of the form :math:`[\boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}]^T` to create a vector of the form :math:`[\boldsymbol{u_x}, \; \boldsymbol{u_x}, \; \boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_y}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}, \; \boldsymbol{u_z} \; \boldsymbol{u_z} ]^T`


.. _inner_products_anisotropic_reciprocal:

Reciprocal Properties
---------------------

Let :math:`\vec{v}` and :math:`\vec{w}` be two physically related quantities.
If their relationship is anisotropic and defined in terms of the reciprocal of a property (defined by a tensor :math:`\Gamma`), the constitutive relation is of the form:

.. math::
    \vec{v} = \Gamma \vec{w}
    :label: inner_product_anisotropic_reciprocal

where

.. math::
    \mathbf{In \; 2D:} \; 
    \Gamma = \begin{bmatrix} \rho^{-1}_{xx} & \rho^{-1}_{xy} \\
    \rho^{-1}_{yx} & \rho^{-1}_{yy} \end{bmatrix}
    \;\;\;\;\;\;\;\; \mathbf{In \; 3D:} \; 
    \Sigma = \begin{bmatrix} \rho^{-1}_{xx} & \rho^{-1}_{xy} & \rho^{-1}_{xz} \\
    \rho^{-1}_{yx} & \rho^{-1}_{yy} & \rho^{-1}_{yz} \\
    \rho^{-1}_{zx} & \rho^{-1}_{zy} & \rho^{-1}_{zz} \end{bmatrix}

Note that for real materials, the tensor is symmetric and has 6 independent variables
(i.e. :math:`\rho_{pq}=\rho_{qp}` for :math:`p,q=x,y,z`).

**Diagonal Anisotropic:**

For the diagonal anisotropic case, the tensor characterzing the material properties
has the form:

.. math::
    \mathbf{In \; 2D:} \; 
    \Gamma = \begin{bmatrix} \rho^{-1}_{x} & 0 \\
    0 & \rho^{-1}_{y} \end{bmatrix}
    \;\;\;\;\;\;\;\; \mathbf{In \; 3D:} \; 
    \Sigma = \begin{bmatrix} \rho^{-1}_{x} & 0 & 0 \\
    0 & \rho^{-1}_{y} & 0 \\
    0 & 0 & \rho^{-1}_{z} \end{bmatrix}
    :label: inner_product_tensor_diagonal_reciprocal


The inner product matrix defined in expression :eq:`inner_product_anisotropic_general` is given by:

.. math::
    \textrm{Vectors on faces:} \; \boldsymbol{M_{\Gamma f}} &= \frac{1}{4} \boldsymbol{P_f^T } \textrm{diag} \boldsymbol{\big ( (e_k \otimes v) \odot \rho^{-1} \big )} \boldsymbol{P_f} \\
    \textrm{Vectors on edges:} \; \boldsymbol{M_{\Gamma e}} &= \frac{1}{4^{k-1}} \boldsymbol{P_e^T } \textrm{diag} \boldsymbol{\big ( (e_k \otimes v) \odot \rho^{-1} \big )} \boldsymbol{P_e}

where :math:`\boldsymbol{\rho^{-1}}` organizes vectors :math:`\boldsymbol{\rho^{-1}_x}`,
:math:`\boldsymbol{\rho^{-1}_y}` and :math:`\boldsymbol{\rho^{-1}_z}` as:

.. math::
    \boldsymbol{\rho^{-1}} = \begin{bmatrix} \boldsymbol{\rho^{-1}_x} \\ \boldsymbol{\rho^{-1}_y} \\ \boldsymbol{\rho^{-1}_z} \\ \end{bmatrix}

and

    - :math:`k = 1,2,3` represent the dimension (1D, 2D or 3D)
    - :math:`\boldsymbol{e_k}` is a vector of 1s of length :math:`k`
    - :math:`\odot` is the Hadamard product
    - :math:`\otimes` is the kronecker product
    - :math:`\boldsymbol{v}` is a vector that stores all of the volumes of the cells
    - :math:`\boldsymbol{P_f}` and :math:`\boldsymbol{P_e}` are projection matricies that map quantities from faces and edges to cell centers, respectively

**Fully Anisotropic:**

For a fully anisotropic case, the tensor characterizing the material properties
has the form is given by:

.. math::
    \mathbf{In \; 2D:} \; 
    \Gamma = \begin{bmatrix} \rho^{-1}_{xx} & \rho^{-1}_{xy} \\
    \rho^{-1}_{yx} & \rho^{-1}_{yy} \end{bmatrix}
    \;\;\;\;\;\;\;\; \mathbf{In \; 3D:} \; 
    \Gamma = \begin{bmatrix} \rho^{-1}_{xx} & \rho^{-1}_{xy} & \rho^{-1}_{xz} \\
    \rho^{-1}_{yx} & \rho^{-1}_{yy} & \rho^{-1}_{yz} \\
    \rho^{-1}_{zx} & \rho^{-1}_{zy} & \rho^{-1}_{zz} \end{bmatrix}
    :label: inner_product_tensor_reciprocal

The inner product matrix defined in expression :eq:`inner_product_anisotropic_general` is given by:

.. math::
    \textrm{Vectors on faces:} \; \boldsymbol{M_{\Gamma f}} &= \frac{1}{4} \boldsymbol{P_f^T Q_u^T} \textrm{diag} \boldsymbol{\big ( (e_k \otimes v) \odot \rho^{-1} \big )} \boldsymbol{Q_w P_f} \\
    \textrm{Vectors on edges:} \; \boldsymbol{M_{\Gamma e}} &= \frac{1}{4^{k-1}} \boldsymbol{P_e^T Q_u^T} \textrm{diag} \boldsymbol{\big ( (e_k \otimes v) \odot \rho^{-1} \big )} \boldsymbol{Q_w P_e}

where :math:`\boldsymbol{\rho^{-1}}` is a large vector that organizes vectors :math:`\boldsymbol{\rho^{-1}_{pq}}` for :math:`p,q=x,y,z` as:

.. math::
    \boldsymbol{\Gamma} = \begin{bmatrix}
    \boldsymbol{\rho^{-1}_{xx}} , \; \boldsymbol{\rho^{-1}_{xy}} , \; \boldsymbol{\rho^{-1}_{xz}} , \;
    \boldsymbol{\rho^{-1}_{yx}} , \; \boldsymbol{\rho^{-1}_{yy}} , \; \boldsymbol{\rho^{-1}_{yz}} , \;
    \boldsymbol{\rho^{-1}_{zx}} , \; \boldsymbol{\rho^{-1}_{zy}} , \; \boldsymbol{\rho^{-1}_{zz}} \end{bmatrix}^T

and

    - :math:`k = 1,2,3` represent the dimension (1D, 2D or 3D)
    - :math:`\boldsymbol{e_k}` is a vector of 1s of length :math:`k`
    - :math:`\odot` is the Hadamard product
    - :math:`\otimes` is the kronecker product
    - :math:`\boldsymbol{v}` is a vector that stores all of the volumes of the cells
    - :math:`\boldsymbol{P_f}` and :math:`\boldsymbol{P_e}` are projection matricies that map quantities from faces and edges to cell centers, respectively
    - :math:`\boldsymbol{Q_u}` is a sparse replication matrix that augments a vector of the form :math:`[\boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}]^T` to create a vector of the form :math:`[\boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}, \; \boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}, \; \boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z} ]^T`
    - :math:`\boldsymbol{Q_w}` is a sparse replication matrix that augments a vector of the form :math:`[\boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}]^T` to create a vector of the form :math:`[\boldsymbol{u_x}, \; \boldsymbol{u_x}, \; \boldsymbol{u_x}, \; \boldsymbol{u_y}, \; \boldsymbol{u_y}, \; \boldsymbol{u_y}, \; \boldsymbol{u_z}, \; \boldsymbol{u_z} \; \boldsymbol{u_z} ]^T`