.. _inner_products_differential:

Differential Operators
**********************

Summary
-------

For practical applications of the finite volume method,
we may need to take the inner product of expressions containing differential operators.
These operators include the:

    - the gradient (:math:`\nabla \phi` )
    - the divergence (:math:`\nabla \cdot \vec{w}` )
    - the curl (:math:`\nabla \times \vec{w}` )

For scalar quantities :math:`\psi` and :math:`\phi` and for vector quantities :math:`\vec{u}` and :math:`\vec{w}`
we are interested in approximating the following inner products:

.. math::
    \begin{align}
    (\vec{u}, \nabla \phi ) &= \int_\Omega \vec{u} \cdot \nabla \phi \, dv\\
    (\psi, \nabla \cdot \vec{w} ) &= \int_\Omega \psi \; (\nabla \cdot \vec{w}) \, dv \\
    (\vec{u}, \nabla \times \vec{w} ) &= \int_\Omega \vec{u} \cdot (\nabla \times \vec{w} ) \, dv
    \end{align}

.. important:: To construct differential operators and/or approximate inner products of this type, see the :ref:`tutorial on inner products with differential operators <sphx_glr_tutorials_inner_products_3_calculus.py>`

Gradient
^^^^^^^^

For the inner product between a vector (:math:`\vec{u}`) and the gradient of a scalar (:math:`\phi`),
there are two options for where the variables should live. For :math:`\boldsymbol{\phi}` defined on the nodes
and :math:`\boldsymbol{u}` defined on cell edges:

.. math::
    \int_\Omega \vec{u} \cdot \nabla \phi \, dv \approx \boldsymbol{u^T M_e G_n \, \phi + B.C}

And for :math:`\boldsymbol{\phi}` defined at cell centers and :math:`\boldsymbol{u}` defined on cell faces:

.. math::
    \int_\Omega \vec{u} \cdot \nabla \phi \, dv \approx - \boldsymbol{u^T D^T M_c \, \phi + B.C.}

where

    - :math:`\boldsymbol{G_n}` is a :ref:`discrete gradient operator <operators_differential_gradient>`
    - :math:`\boldsymbol{D}` is a :ref:`discrete divergence operator <operators_differential_divergence>`
    - :math:`\boldsymbol{M_c}` is the :ref:`basic inner product matrix for vectors at cell centers <inner_products_basic>`
    - :math:`\boldsymbol{M_e}` is the :ref:`basic inner product matrix for vectors at edges <inner_products_basic>`
    - :math:`\boldsymbol{BC}` represents an additional contribution accounting for the boundary conditions

Divergence
^^^^^^^^^^

For the inner product between a scalar (:math:`\psi`) and the divergence of a vector (:math:`\vec{w}`),
there are two options for where the variables should live. For :math:`\boldsymbol{\psi}` defined at cell centers
and :math:`\boldsymbol{w}` defined on cell faces:

.. math::
    \int_\Omega \psi \; (\nabla \cdot \vec{w}) \, dv \approx \boldsymbol{\psi^T M_c D \, w + B.C.}

And for :math:`\boldsymbol{\psi}` defined on the nodes and :math:`\boldsymbol{w}` defined on cell edges:

.. math::
    \int_\Omega \psi \; (\nabla \cdot \vec{w}) \, dv \approx - \boldsymbol{\psi^T G_n^T M_e \, w + B.C.}

where

    - :math:`\boldsymbol{G_n}` is a :ref:`discrete gradient operator <operators_differential_gradient>`
    - :math:`\boldsymbol{D}` is a :ref:`discrete divergence operator <operators_differential_divergence>`
    - :math:`\boldsymbol{M_c}` is the :ref:`basic inner product matrix for vectors at cell centers <inner_products_basic>`
    - :math:`\boldsymbol{M_e}` is the :ref:`basic inner product matrix for vectors at edges <inner_products_basic>`
    - :math:`\boldsymbol{BC}` represents an additional contribution accounting for the boundary conditions

Curl
^^^^

For the inner product between a vector (:math:`\vec{u}`) and the curl of another vector (:math:`\vec{w}`),
there are two options for where the variables should live. For :math:`\boldsymbol{u}` defined on the faces
and :math:`\boldsymbol{w}` defined on cell edges:

.. math::
    \int_\Omega \vec{u} \cdot (\nabla \times \vec{w} ) \, dv \approx \boldsymbol{u^T M_f C \, w + B.C.}

And for :math:`\boldsymbol{u}` defined on the edges and :math:`\boldsymbol{w}` defined on cell faces:

.. math::
    \int_\Omega \vec{u} \cdot (\nabla \times \vec{w} ) \, dv \approx \boldsymbol{u^T C^T \! M_f \, w + B.C.}

where

    - :math:`\boldsymbol{C}` is a :ref:`discrete curl operator <operators_differential_curl>`
    - :math:`\boldsymbol{M_f}` is the :ref:`basic inner product matrix for vectors on cell faces <inner_products_basic>`
    - :math:`\boldsymbol{BC}` represents an additional contribution accounting for the boundary conditions

Vector and the Gradient of a Scalar
-----------------------------------

Let :math:`\phi` be a scalar quantity and let :math:`\vec{u}` be a vector quantity.
We are interested in approximating the following inner product:

.. math::
    (\vec{u}, \nabla \phi ) = \int_\Omega \vec{u} \cdot \nabla \phi \, dv
    :label: inner_products_differential_gradient

Inner Product on Edges
^^^^^^^^^^^^^^^^^^^^^^

Here, the discrete representation :math:`\boldsymbol{\phi}` lives on the nodes and
the discrete representation :math:`\boldsymbol{u}` lives on the edges.
Since the :ref:`discrete gradient operator <operators_differential_gradient>` maps
a discrete scalar quantity from nodes to edges, we can approximate the inner product
between two discrete quantities living on the edge. Thus:

.. math::
    \int_\Omega \vec{u} \cdot \nabla \phi \, dv \approx \boldsymbol{u^T M_e G_n \, \phi + B.C}

where

    - :math:`\boldsymbol{G_n}` is the :ref:`discrete gradient operator <operators_differential_gradient>`
    - :math:`\boldsymbol{M_e}` is the :ref:`basic inner product matrix for vectors at edges <inner_products_basic>`
    - :math:`\boldsymbol{BC}` represents an additional contribution accounting for the boundary conditions


For this inner product, the natural boundary condition is a Dirichlet condition such that :math:`\phi = 0` on the boundary.


Inner Product on Faces
^^^^^^^^^^^^^^^^^^^^^^

Here, the discrete representation :math:`\boldsymbol{\phi}` lives at cell centers and
the discrete representation :math:`\boldsymbol{u}` lives on the faces.
We cannot simply use a discrete gradient operator, as a mapping from cell centers
to faces would require knowledge of the scalar at locations outside the mesh.

To evaluate the inner product we use the identity
:math:`\vec{u} \cdot \nabla \phi = \nabla \cdot \phi\vec{u} - \phi \nabla \cdot \vec{u}`
and apply the divergence theorem to equation :eq:`inner_products_differential_gradient`:

.. math::
    \begin{align}
    \int_\Omega \vec{u} \cdot \nabla \phi \, dv &= - \int_\Omega \phi \nabla \cdot \vec{u} \, dv + \int_\Omega \nabla \cdot \phi\vec{u} \, dv \\
    &= - \int_\Omega \phi \nabla \cdot \vec{u} \, dv + \oint_{\partial \Omega} \hat{n} \cdot \phi\vec{u} \, da
    \end{align}
    :label: inner_products_differential_gradient_centers

Where boundary conditions are implemented in the surface integral. The approximate to the inner product is given by:

.. math::
    \int_\Omega \vec{u} \cdot \nabla \phi \, dv \approx - \boldsymbol{u^T D^T M_c \, \phi + B.C}

where

    - :math:`\boldsymbol{D}` is the :ref:`discrete divergence operator <operators_differential_divergence>`
    - :math:`\boldsymbol{M_c}` is the :ref:`basic inner product matrix for vectors at cell centers <inner_products_basic>`
    - :math:`\boldsymbol{BC}` represents an additional contribution accounting for the boundary conditions


For this inner product, the natural boundary condition is a Dirichlet condition such that :math:`\frac{\partial \phi}{\partial n} = 0` on the boundary.


Scalar and the Divergence of a Vector
-------------------------------------

Let :math:`\psi` be a scalar quantity and let :math:`\vec{w}` be a vector quantity.
We are interested in approximating the following inner product:

.. math::
    (\psi, \nabla \cdot \vec{w} ) = \int_\Omega \psi \; (\nabla \cdot \vec{w}) \, dv \\
    :label: inner_products_differential_divergence


Inner Product on Faces
^^^^^^^^^^^^^^^^^^^^^^

Here, the discrete representation :math:`\boldsymbol{\psi}` lives at cell centers and
the discrete representation :math:`\boldsymbol{w}` lives on the faces.
Since the :ref:`discrete divergence operator <operators_differential_divergence>` maps
a discrete vector quantity from faces to cell centers, we can approximate the inner product
between two discrete quantities living at the centers. Thus:

.. math::
    \int_\Omega \psi \; (\nabla \cdot \vec{w}) \, dv \approx \boldsymbol{\psi^T M_c D \, w + B.C}

where

    - :math:`\boldsymbol{D}` is the :ref:`discrete divergence operator <operators_differential_divergence>` 
    - :math:`\boldsymbol{M_c}` is the :ref:`basic inner product matrix for vectors at cell centers <inner_products_basic>`
    - :math:`\boldsymbol{BC}` represents an additional contribution accounting for the boundary conditions



Inner Product on Edges
^^^^^^^^^^^^^^^^^^^^^^

Here, the discrete representation :math:`\boldsymbol{\psi}` lives on the nodes and
the discrete representation :math:`\boldsymbol{w}` lives on the edges.
We cannot simply use a discrete divergence operator, as a mapping from edges
to nodes would require knowledge of the scalar at locations outside the mesh.

To evaluate the inner product we use the identity
:math:`\psi \nabla \cdot \vec{w} = \nabla \cdot \psi\vec{w} - \vec{w} \cdot \nabla \psi`
and apply the divergence theorem to equation :eq:`inner_products_differential_gradient`:

.. math::
    \begin{align}
    \int_\Omega \psi \; (\nabla \cdot \vec{w}) \, dv &= - \int_\Omega \vec{w} \cdot \nabla \psi \, dv + \int_\Omega \nabla \cdot \psi\vec{w} \, dv \\
    &= - \int_\Omega \phi \nabla \cdot \vec{u} \, dv + \oint_{\partial \Omega} \hat{n} \cdot \phi\vec{u} \, da
    \end{align}
    :label: inner_products_differential_divergence_edges

Where boundary conditions are implemented in the surface integral. The approximate to the inner product is given by:

.. math::
    \int_\Omega \psi \; (\nabla \cdot \vec{w}) \, dv \approx - \boldsymbol{\psi^T G_n^T M_e \, w + B.C}

where

    - :math:`\boldsymbol{G_n}` is the :ref:`discrete gradient operator <operators_differential_divergence>`
    - :math:`\boldsymbol{M_e}` is the :ref:`basic inner product matrix for vectors at the edges <inner_products_basic>`
    - :math:`\boldsymbol{BC}` represents an additional contribution accounting for the boundary conditions




Vector and the Curl of a Vector
-------------------------------

Let :math:`\vec{u}` and :math:`\vec{w}` be vector quantities.
We are interested in approximating the following inner product:

.. math::
    (\vec{u}, \nabla \times \vec{w} ) = \int_\Omega \vec{u} \cdot (\nabla \times \vec{w} ) \, dv
    :label: inner_products_differential_curl


Inner Product at Faces
^^^^^^^^^^^^^^^^^^^^^^

Here, the discrete representation :math:`\boldsymbol{u}` lives on the faces and
the discrete representation :math:`\boldsymbol{w}` lives on the edges.
Since the :ref:`discrete curl operator <operators_differential_curl>` maps
a discrete vector quantity from edges to faces, we can approximate the inner product
between two discrete quantities living on the faces. Thus:

.. math::
    \int_\Omega \vec{u} \cdot (\nabla \times \vec{w}) \, dv \approx \boldsymbol{u^T M_f C w + B.C}

where

    - :math:`\boldsymbol{C}` is the :ref:`discrete curl operator <operators_differential_curl>` 
    - :math:`\boldsymbol{M_f}` is the :ref:`basic inner product matrix for vectors on faces <inner_products_basic>`
    - :math:`\boldsymbol{BC}` represents an additional contribution accounting for the boundary conditions



Inner Product at Edges
^^^^^^^^^^^^^^^^^^^^^^

Here, the discrete representation :math:`\boldsymbol{u}` lives on the edges and
the discrete representation :math:`\boldsymbol{w}` lives on the faces.
We cannot simply use the discrete curl operator, as a mapping from faces
to edges would require knowledge of :math:`\boldsymbol{w}` at locations outside the mesh.

To evaluate the inner product we use the identity
:math:`\vec{u} \cdot (\nabla \times \vec{w}) = \vec{w} \cdot (\nabla \times \vec{u}) - \nabla \cdot (\vec{u} \times \vec{w})`
and apply the divergence theorem to equation :eq:`inner_products_differential_curl`:

.. math::
    \begin{align}
    \int_\Omega \vec{u} \cdot (\nabla \times \vec{w}) \, dv &= \int_\Omega \vec{w} \cdot (\nabla \times \vec{u}) \, dv - \int_\Omega \nabla \cdot (\vec{u} \times \vec{w}) \, dv \\
    &= \int_\Omega \vec{w} \cdot (\nabla \times \vec{u}) \, dv + \oint_{\partial \Omega} \hat{n} \cdot (\vec{u} \times \vec{w}) \, da
    \end{align}
    :label: inner_products_differential_curl_edges

Where boundary conditions are implemented in the surface integral. The approximate to the inner product is given by:

.. math::
    \int_\Omega \vec{u} \cdot (\nabla \times \vec{w}) \, dv \approx \boldsymbol{u^T C^T M_f \, w + B.C}

where

    - :math:`\boldsymbol{C}` is the :ref:`discrete curl operator <operators_differential_curl>` 
    - :math:`\boldsymbol{M_f}` is the :ref:`basic inner product matrix for vectors on faces <inner_products_basic>`
    - :math:`\boldsymbol{BC}` represents an additional contribution accounting for the boundary conditions
