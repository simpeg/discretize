.. _inner_products_basic:

Basic Inner Products
********************

Summary
-------

Inner products between two scalar or vector quantities represents the most
basic class of inner products. For scalar quantities :math:`\psi` and :math:`\phi`,
we will show that a disrete approximation to the inner product is given by:

.. math::
    (\psi , \phi ) = \int_\Omega \psi \, \phi \, dv \approx \boldsymbol{\psi^T \, M \, \phi}
    :label: inner_product_basic_scalar

And for vector quantities :math:`\vec{u}` and :math:`\vec{w}`, a discrete approximation
to the inner product is given by:

.. math::
    (\vec{u}, \vec{w}) = \int_\Omega \vec{u} \cdot \vec{w} \, dv \approx \mathbf{u^T \, M \, w}
    :label: inner_product_basic_vector

where :math:`\mathbf{M}` in either equation represents an
*inner-product matrix*, and :math:`\boldsymbol{\psi}`, :math:`\boldsymbol{\phi}`,
:math:`\mathbf{u}` and :math:`\mathbf{w}` are discrete variables that live
on the mesh. It is important to note a few things about the
inner-product matrix in this case:

    1. It depends on the dimensions and discretization of the mesh
    2. It depends on where the discrete variables live; e.g. edges, faces, nodes, centers

For this simple class of inner products, the corresponding form of the inner product matricies for
discrete quantities living on various parts of the mesh are shown below:

.. math::
    \textrm{Scalars at centers:} \; \mathbf{M_c} &= \textrm{diag} (\mathbf{v} ) \\
    \textrm{Scalars at nodes:} \; \mathbf{M_n} &= \frac{1}{2^{2k}} \mathbf{P_n^T } \textrm{diag} (\mathbf{v} ) \mathbf{P_n} \\
    \textrm{Vectors on faces:} \; \mathbf{M_f} &= \frac{1}{4} \mathbf{P_f^T } \textrm{diag} (\mathbf{e_k \otimes v} ) \mathbf{P_f} \\
    \textrm{Vectors on edges:} \; \mathbf{M_e} &= \frac{1}{4^{k-1}} \mathbf{P_e^T } \textrm{diag} (\mathbf{e_k \otimes v}) \mathbf{P_e}

where

    - :math:`k = 1,2,3` represent the dimension (1D, 2D or 3D)
    - :math:`\mathbf{e_k}` is a vector of 1s of length :math:`k`
    - :math:`\otimes` is the kronecker product
    - :math:`\mathbf{P}` are projection matricies that map quantities from one part of the cell (nodes, faces, edges) to cell centers
    - :math:`\mathbf{v}` is a vector that stores all of the volumes of the cells

.. note:: To construct the inner product matrix and/or approximate inner products of this type, see the :ref:`tutorial on basic inner products <sphx_glr_tutorials_inner_products_1_basic.py>`


Scalars at Cell Centers
-----------------------

We want to approximate the inner product of two scalar quantities :math:`\psi` and :math:`\phi`
where the discrete quantities :math:`\boldsymbol{\psi}` and :math:`\boldsymbol{\phi}` are defined
to live at cell centers. Assuming we know the values of :math:`\psi` and :math:`\phi` at cell centers,
our goal is to construct the inner product matrix :math:`\mathbf{M}` such that: 

.. math::
    (\psi , \phi ) = \int_\Omega \psi \, \phi \, dv \approx \mathbf{\psi^T \, M \, \phi}


.. image:: ../images/center_discretization.png
    :align: center
    :width: 600


For a single cell (see above), the contribution towards the inner product is obtained by multiplying
:math:`\psi` and :math:`\phi` at the cell centers (represented by :math:`\psi_i and \phi_i`)
by the cell's volume (:math:`v_i`), i.e.:

.. math::
    \int_{\Omega_i} \psi \, \phi \, dv \approx \psi_i \phi_i v_i

Therefore a simple approximation to the inner product is obtained by summing the above
approximation over all cells. Where :math:`nc` refers to the number of cells in the mesh:

.. math::
     \int_\Omega \psi \, \phi \, dv = \approx \sum_i^{nc} \psi_i \phi_i v_i

Expressing the sum in terms of linear equations, we obtain:

.. math::
     (\psi , \phi ) = \int_\Omega \psi \, \phi \, dv  \approx \mathbf{\psi^T \, M_c \, \phi}

where the mass matrix for cell centered quantities is just a diagonal matrix containing
the cell volumes (:math:`\mathbf{v}`), i.e.:

.. math::
    \mathbf{M_c} = diag(\mathbf{v})


.. note:: To construct the inner product matrix and/or approximate inner products of this type, see the :ref:`tutorial on basic inner products <sphx_glr_tutorials_inner_products_1_basic.py>`


Scalars at Nodes
----------------

We want to approximate the inner product of two scalar quantities :math:`\psi` and :math:`\phi`
where the discrete quantities :math:`\boldsymbol{\psi}` and :math:`\boldsymbol{\phi}` are defined
to live on cell nodes. Assuming we know the values of :math:`\psi` and :math:`\phi` at the nodes,
our goal is to construct the inner product matrix :math:`\mathbf{M}` such that: 

.. math::
    (\psi , \phi ) = \int_\Omega \psi \, \phi \, dv \approx \mathbf{\psi^T \, M \, \phi}
    :label: inner_product_basic_nodes

Whereas :math:`\boldsymbol{\psi}` and :math:`\boldsymbol{\phi}` are defined
to live on cell nodes, it makes more sense for cell volumes to be considered a property
which lives at cell centers. This makes evaluating the inner product more complicated as
discrete quantities do not live at the same place.

.. image:: ../images/node_discretization.png
    :align: center
    :width: 600

For a single cell :math:`i`, the contribution towards the inner product is approximated by
mapping the values at the nodes to cell centers, taking the average, then multiplying
by the cell volume. For 2D cells there are 4 nodes. And for 3D cells there are 8 nodes
Thus:

.. math::
    \begin{align}
    \mathbf{In \; 2D:} \; \int_{\Omega_i} \psi \, \phi \, dv \approx & \;\;
    \frac{v_i}{16} \Bigg ( \psi_i^{(1)} \! + \psi_i^{(2)} \! + \psi_i^{(3)} \! + \psi_i^{(4)} \Bigg )
    \Bigg ( \phi_i^{(n1)} \! + \phi_i^{(n2)} \! + \phi_i^{(n3)} \! + \phi_i^{(n4)} \Bigg ) \\
    & \\
    \mathbf{In \; 3D:} \; \int_{\Omega_i} \psi \, \phi \, dv \approx & \;\; 
    \frac{v_i}{16} \Bigg ( \sum_{n=1}^8 \psi_i^{(n)} \Bigg ) \Bigg ( \sum_{n=1}^8 \psi_i^{(n)} \Bigg )
    \end{align}
    :label: inner_product_basic_nodes_1

where the superscript :math:`(n)` is used to point to a specific node.
Using the contribution for each cell described in expression :eq:`inner_product_basic_nodes_1`,
we want to approximate the inner product in the form described by
equation :eq:`inner_product_basic_nodes`. To accomlish this, we construct a sparse matrix
:math:`\mathbf{P_n}` which projects quantities on the nodes to the
the cell centers.

Our final approximation for the inner product is therefore:

.. math::
     (\psi , \phi ) = \int_\Omega \psi \, \phi \, dv  \approx \mathbf{\psi^T \, M_n \, \phi}

where the mass matrix for nodal quantities has the form:

.. math::
    \mathbf{M_n} = \frac{1}{2^{2k}} \mathbf{P_n^T } \textrm{diag} (\mathbf{v} ) \mathbf{P_n}

where

    - :math:`k = 1,2,3` represent the dimension (1D, 2D or 3D)
    - :math:`\mathbf{P_n}` is a projection matrix that maps quantities from nodes to cell centers
    - :math:`\mathbf{v}` is a vector that stores all of the volumes of the cells

.. note:: To construct the inner product matrix and/or approximate inner products of this type, see the :ref:`tutorial on basic inner products <sphx_glr_tutorials_inner_products_1_basic.py>`


Vectors on Cell Faces
---------------------

For the mimetic finite volume approach, fluxes are generally defined on cell faces;
as it allows cells to share faces while preserving natural boundary conditions.

We want to approximate the inner product of two vector quantities :math:`\vec{u}` and :math:`\vec{w}`
where the discrete quantities :math:`\mathbf{u}` and :math:`\mathbf{w}` are defined
to live on cell faces. Assuming we know the values of :math:`\vec{u}` and :math:`\vec{w}` on the faces,
our goal is to construct the inner product matrix :math:`\mathbf{M}` in the expression below: 

.. math::
    (\vec{u}, \vec{w}) = \int_\Omega \vec{u} \cdot \vec{w} \, dv \approx \mathbf{u^T \, M \, w}
    :label: inner_product_basic_faces

We must respect the dot product. For vectors defined on cell faces, we discretize such that the
x-component of the vectors live on the x-faces, the y-component lives y-faces and the z-component
lives on the z-faces. For a single cell, this is illustrated in 2D and 3D below.

.. image:: ../images/face_discretization.png
    :align: center
    :width: 600


As we can see there are 2 faces for each component. Therefore, we need to project each component of the
vector from its faces to the cell centers and take their averages separately. For a single cell with volume :math:`v_i`,
the contribution towards the inner product is:

.. math::
    \begin{align}
    \mathbf{In \; 2D:} \; \int_{\Omega_i} \vec{u} \cdot \vec{w} \, dv \approx & \;\; \frac{v_i}{4} \Big ( u_x^{(1)} + u_x^{(2)} \Big ) \Big ( w_x^{(1)} + w_x^{(2)} \Big ) \\
    & + \frac{v_i}{4} \Big ( u_y^{(1)} + u_y^{(2)} \Big ) \Big ( w_y^{(1)} + w_y^{(2)} \Big ) \\
    & \\
    \mathbf{In \; 3D:} \; \int_{\Omega_i} \vec{u} \cdot \vec{w} \, dv \approx & \;\; \frac{v_i}{4} \Big ( u_x^{(1)} + u_x^{(2)} \Big ) \Big ( w_x^{(1)} + w_x^{(2)} \Big ) \\
    & + \frac{v_i}{4} \Big ( u_y^{(1)} + u_y^{(2)} \Big ) \Big ( w_y^{(1)} + w_y^{(2)} \Big ) \\
    & + \frac{v_i}{4} \Big ( u_z^{(1)} + u_z^{(2)} \Big ) \Big ( w_z^{(1)} + w_z^{(2)} \Big )
    \end{align}
    :label: inner_product_basic_faces_1

where superscripts :math:`(1)` and :math:`(2)` denote face 1 and face 2, respectively.
Using the contribution for each cell described in expression :eq:`inner_product_basic_faces_1`,
we want to approximate the inner product in the form described by
equation :eq:`inner_product_basic_faces`. To accomlish this, we construct a sparse matrix
:math:`\mathbf{P_f}` which projects quantities on the x, y and z faces separately to the
the cell centers.

For discretize vectors :math:`\mathbf{u}` and :math:`\mathbf{w}` whose x, y (and z) components
are organized on cell faces as follows:

.. math::
    \mathbf{u} = \begin{bmatrix} \mathbf{u_x} \\ \mathbf{u_y} \\ \mathbf{u_y} \\ \end{bmatrix}
    \;\;\;\; \textrm{and} \;\;\;\;
    \mathbf{w} = \begin{bmatrix} \mathbf{w_x} \\ \mathbf{w_y} \\ \mathbf{w_y} \\ \end{bmatrix}

the approximation to the inner product is given by:

.. math::
     (\vec{u}, \vec{w}) = \int_\Omega \vec{u} \cdot \vec{w} \, dv \approx \mathbf{\mathbf{u} \, M_f \, \mathbf{w}}

where the mass matrix for face quantities has the form:

.. math::
    \mathbf{M_f} = \frac{1}{4} \mathbf{P_f^T } \textrm{diag} (\mathbf{e_k \otimes v} ) \mathbf{P_f}

and

    - :math:`k = 1,2,3` represents the dimension (1D, 2D or 3D)
    - :math:`\mathbf{e_k}` is a vector of 1s of length :math:`k`
    - :math:`\otimes` is the kronecker product
    - :math:`\mathbf{P_f}` is the projection matrix that maps quantities from faces to cell centers
    - :math:`\mathbf{v}` is a vector that stores all of the volumes of the cells

.. note:: To construct the inner product matrix and/or approximate inner products of this type, see the :ref:`tutorial on basic inner products <sphx_glr_tutorials_inner_products_1_basic.py>`


Vectors on Cell Edges
---------------------

For the mimetic finite volume approach, fields are generally defined on cell edges;
as it allows cells to share edges while preserving natural boundary conditions.
We want to approximate the inner product of two vector quantities :math:`\vec{u}` and :math:`\vec{w}`
where the discrete quantities :math:`\mathbf{u}` and :math:`\mathbf{w}` are defined
to live at cell edges. Assuming we know the values of :math:`\vec{u}` and :math:`\vec{w}` at the edges,
our goal is to construct the inner product matrix :math:`\mathbf{M}` in the expression below: 

.. math::
    (\vec{u}, \vec{w}) = \int_\Omega \vec{u} \cdot \vec{w} \, dv \approx \mathbf{u^T \, M \, w}
    :label: inner_product_basic_edges

We must respect the dot product. For vectors defined on cell edges, we discretize such that the
x-component of the vectors live on the x-edges, the y-component lives y-edges and the z-component
lives on the z-edges. This is illustrated in 2D and 3D below.

.. image:: ../images/edge_discretization.png
    :align: center
    :width: 600


As we can see there are 2 edges for each component in 2D and 4 edges for each component in 3D.
Therefore, we need to project each component of the
vector from its edges to the cell centers and take their averages separately. For a single cell with volume :math:`v_i`,
the contribution towards the inner product is:

.. math::
    \begin{align}
    \mathbf{In \; 2D:} \; \int_{\Omega_i} \vec{u} \cdot \vec{w} \, dv \approx & \;\; \frac{v_i}{4} \Big ( u_x^{(1)} + u_x^{(2)} \Big ) \Big ( w_x^{(1)} + w_x^{(2)} \Big ) \\
    & + \frac{v_i}{4} \Big ( u_y^{(1)} + u_y^{(2)} \Big ) \Big ( w_y^{(1)} + w_y^{(2)} \Big ) \\
    & \\
    \mathbf{In \; 3D:} \; \int_{\Omega_i} \vec{u} \cdot \vec{w} \, dv \approx & \;\; \frac{v_i}{16} \Bigg ( \sum_{n=1}^4 u_x^{(n)} \Bigg ) \Bigg ( \sum_{n=1}^4 w_x^{(n)} \Bigg ) \\
    & + \frac{v_i}{16} \Bigg ( \sum_{n=1}^4 u_y^{(n)} \Bigg ) \Bigg ( \sum_{n=1}^4 w_y^{(n)} \Bigg ) \\
    & + \frac{v_i}{16} \Bigg ( \sum_{n=1}^4 u_z^{(n)} \Bigg ) \Bigg ( \sum_{n=1}^4 w_z^{(n)} \Bigg )
    \end{align}
    :label: inner_product_basic_edges_1

where the superscript :math:`(n)` denotes a particular edge.
Using the contribution for each cell described in expression :eq:`inner_product_basic_edges_1`,
we want to approximate the inner product in the form described by
equation :eq:`inner_product_basic_edges`. To accomlish this, we construct a sparse matrix
:math:`\mathbf{P_e}` which projects quantities on the x, y and z edges separately to the
the cell centers.

For discretize vectors :math:`\mathbf{u}` and :math:`\mathbf{w}` whose x, y (and z) components
are organized on cell edges as follows:

.. math::
    \mathbf{u} = \begin{bmatrix} \mathbf{u_x} \\ \mathbf{u_y} \\ \mathbf{u_y} \\ \end{bmatrix}
    \;\;\;\; \textrm{and} \;\;\;\;
    \mathbf{w} = \begin{bmatrix} \mathbf{w_x} \\ \mathbf{w_y} \\ \mathbf{w_y} \\ \end{bmatrix}

the approximation to the inner product is given by:

.. math::
     (\vec{u}, \vec{w}) = \int_\Omega \vec{u} \cdot \vec{w} \, dv \approx \mathbf{\mathbf{u} \, M_e \, \mathbf{w}}

where the mass matrix for face quantities has the form:

.. math::
    \mathbf{M_e} = \frac{1}{4^{k-1}} \mathbf{P_e^T } \textrm{diag} (\mathbf{e_k \otimes v}) \mathbf{P_e}

and

    - :math:`k = 1,2,3` represents the dimension (1D, 2D or 3D)
    - :math:`\mathbf{e_k}` is a vector of 1s of length :math:`k`
    - :math:`\otimes` is the kronecker product
    - :math:`\mathbf{P_e}` is the projection matrix that maps quantities from edges to cell centers
    - :math:`\mathbf{v}` is a vector that stores all of the volumes of the cells

.. note:: To construct the inner product matrix and/or approximate inner products of this type, see the :ref:`tutorial on basic inner products <sphx_glr_tutorials_inner_products_1_basic.py>`








