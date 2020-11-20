.. _inner_products_differential:

Differential Operators
**********************

Summary
-------

For practical applications of the finite volume method,
we may need to take the inner product of expressions containing differential operators.
These operators include:

    - the gradient (:math:`\nabla`)
    - the divergence (:math:`\nabla \cdot \;`)
    - the curl (:math:`\nabla \times \;`)

For scalar quantities :math:`\psi` and :math:`\phi` and for vector quantities :math:`\vec{u}` and :math:`\vec{w}`
we are interested in approximating the following inner products:

.. math::
    \begin{align}
    (\vec{u}, \nabla \phi ) &= \int_\Omega \vec{u} \cdot \nabla \phi \, dv\\
    (\psi, \nabla \cdot \vec{w} ) &= \int_\Omega \psi \; (\nabla \cdot \vec{w}) \, dv \\
    (\vec{u}, \nabla \times \vec{w} ) &= \int_\Omega \vec{u} \cdot (\nabla \times \vec{w} ) \, dv
    \end{align}

.. note:: To construct differential operators and/or approximate inner products of this type, see the :ref:`tutorial on inner products with differential operators <sphx_glr_tutorials_inner_products_3_calculus.py>`

**Gradient:**

For the inner product between a vector (:math:`\vec{u}`) and the gradient of a scalar (:math:`\phi`),
there are two options for where the variables should live. For :math:`\boldsymbol{\phi}` defined on the nodes
and :math:`\boldsymbol{u}` defined on cell edges:

.. math::
    \int_\Omega \vec{u} \cdot \nabla \phi \, dv \approx \boldsymbol{u^T M_e G_n \, \phi + B.C}

And for :math:`\boldsymbol{\phi}` defined at cell centers and :math:`\boldsymbol{u}` defined on cell faces:

.. math::
    \int_\Omega \vec{u} \cdot \nabla \phi \, dv \approx - \boldsymbol{u^T D^T M_c \, \phi + B.C.}

where

    - :math:`\boldsymbol{G_n}` is a discrete gradient operator that maps from nodes to edges
    - :math:`\boldsymbol{D}` is a discrete divergence operator whose transpose acts as a gradient operator from faces to cell centers
    - :math:`\boldsymbol{M_c}` is the :ref:`basic inner product matrix for vectors at cell centers <inner_products_basic>`
    - :math:`\boldsymbol{M_e}` is the :ref:`basic inner product matrix for vectors at edges <inner_products_basic>`
    - :math:`\boldsymbol{BC}` represents an additional contribution accounting for the boundary conditions

**Divergence:**

For the inner product between a scalar (:math:`\psi`) and the divergence of a vector (:math:`\vec{w}`),
there are two options for where the variables should live. For :math:`\boldsymbol{\psi}` defined at cell centers
and :math:`\boldsymbol{w}` defined on cell faces:

.. math::
    \int_\Omega \psi \; (\nabla \cdot \vec{w}) \, dv \approx \boldsymbol{\psi^T M_c D \, w + B.C.}

And for :math:`\boldsymbol{\psi}` defined on the nodes and :math:`\boldsymbol{w}` defined on cell edges:

.. math::
    \int_\Omega \psi \; (\nabla \cdot \vec{w}) \, dv \approx - \boldsymbol{\psi^T G_n^T M_e \, w + B.C.}

where

    - :math:`\boldsymbol{D}` is a discrete divergence operator from faces to cell centers
    - :math:`\boldsymbol{G_n}` is a discrete gradient operator whose transpose acts as a divergence operator from edges to nodes
    - :math:`\boldsymbol{M_c}` is the :ref:`basic inner product matrix for vectors at cell centers <inner_products_basic>`
    - :math:`\boldsymbol{M_e}` is the :ref:`basic inner product matrix for vectors at edges <inner_products_basic>`
    - :math:`\boldsymbol{BC}` represents an additional contribution accounting for the boundary conditions

**Curl:**

For the inner product between a vector (:math:`\vec{u}`) and the curl of another vector (:math:`\vec{w}`),
there are two options for where the variables should live. For :math:`\boldsymbol{u}` defined on the faces
and :math:`\boldsymbol{w}` defined on cell edges:

.. math::
    \int_\Omega \vec{u} \cdot (\nabla \times \vec{w} ) \, dv \approx \boldsymbol{u^T M_f C \, w + B.C.}

And for :math:`\boldsymbol{u}` defined on the edges and :math:`\boldsymbol{w}` defined on cell faces:

.. math::
    \int_\Omega \vec{u} \cdot (\nabla \times \vec{w} ) \, dv \approx \boldsymbol{u^T C^T \! M_f \, w + B.C.}

where

    - :math:`\boldsymbol{C}` is a discrete curl operator from edges to faces, whose transpose acts as a curl operator from faces to edges 
    - :math:`\boldsymbol{M_f}` is the :ref:`basic inner product matrix for vectors on cell faces <inner_products_basic>`
    - :math:`\boldsymbol{BC}` represents an additional contribution accounting for the boundary conditions

Vector and the Gradient of a Scalar
-----------------------------------




Scalar and the Divergence of a Vector
-------------------------------------





Vector and the Curl of a Vector
-------------------------------