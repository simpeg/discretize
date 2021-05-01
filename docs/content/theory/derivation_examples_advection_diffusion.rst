.. _derivation_examples_advection_diffusion:

Advection and Diffusion with Zero Neumann Boundary Condition
************************************************************

Here we provide the derivation for solving the advection-diffusion equation using the finite volume method.
We assume the fluid is incompressible. Key lessons include:

    - Implementing boundary conditions
    - Solving time-dependent PDEs
    - Strategies for applying finite volume to 2nd order PDEs

**Tutorial:** :ref:`Advection-Diffusion Equation <sphx_glr_tutorials_pde_2_advection_diffusion.py>`

Setup
-----

If we assume the fluid is incompressible (i.e. :math:`\nabla \cdot \vec{u} = 0`),
the advection-diffusion equation with zero Neumann boundary conditions is given by:

.. math::
	\begin{align}
    & p_t = \nabla \cdot \alpha \nabla p - \vec{u} \cdot \nabla p + s \\
    & \textrm{s.t.} \;\;\; \frac{\partial p}{\partial n} \Bigg |_{\partial \Omega} = 0 \\
    & \textrm{and} \;\;\; p(t=0) = 0
    \end{align}
    :label: derivation_examples_advection_diffusion_1

where 

	- :math:`p` is the unknown variable
	- :math:`p_t` is its time-derivative
	- :math:`\alpha` defines the diffusivity within the domain
	- :math:`\vec{u}` is the velocity field
	- :math:`s` is the source term

We will consider the case where there is a single point source within our domain.
Where :math:`s_0` is a constant:

.. math::
    s = s_0 \delta ( \vec{r} )

Direct implementation of the finite volume method is more challenging on higher order PDEs.
Therefore, we redefine the problem as a set of first order PDEs as follows:

.. math::
	\begin{align}
    p_t = \nabla \cdot \vec{j} - \vec{u} \cdot \vec{w} + s \;\;\;\; &(1)\\
    \vec{w} = \nabla p \;\;\;\; &(2)\\
    \alpha^{-1} \vec{j} = \vec{w} \;\;\;\; &(3)
    \end{align}
    :label: derivation_examples_advection_diffusion_2

We then take the inner products between the expressions in equation :eq:`derivation_examples_advection_diffusion_2` and an appropriate test function.

Discretization in Space
-----------------------

Because this is a time-dependent problem, we must consider discretization in both space and time.
We generally begin by discretizing in space. Where :math:`\boldsymbol{p}` is the discrete representation of :math:`p`,
we will discretize such that :math:`\boldsymbol{p}` lives on the cell centers. If :math:`\boldsymbol{p}` lives
at cell centers, it makes sense for the discrete representations  :math:`\vec{w}`, :math:`\vec{j}` and :math:`\vec{u}`
in expression :eq:`derivation_examples_advection_diffusion_2` to live on the faces.

First Equation
^^^^^^^^^^^^^^

Let :math:`\psi` be a scalar test function whose discrete representation :math:`\boldsymbol{\psi}` lives at cell centers.
The inner product between :math:`\psi` and the first equation
in :eq:`derivation_examples_advection_diffusion_2` is given by:

.. math::
	\int_\Omega \psi p_t \, dv = \int_\Omega \psi \nabla \cdot \vec{j} \, dv + \int_\Omega \psi (\vec{u} \cdot \vec{w}) \, dv + \int_\Omega \psi s \, dv
	:label: derivation_examples_advection_diffusion_3

According to :ref:`basic inner products <inner_products_basic>`, the term to the left of the equals sign is approximated by:

.. math::
	\int_\Omega \psi p_t \, dv \approx \boldsymbol{\psi^T M_c \, p_t}
	:label: derivation_examples_advection_diffusion_4a

where :math:`\boldsymbol{M_c}` is the :ref:`inner product matrix for quantities at cell centers <inner_products_basic>`.

Since :math:`\boldsymbol{\psi}` lives at cell centers, then so must the divergence of :math:`\vec{j}`.
This implies the discrete vector :math:`\boldsymbol{j}` must live on the faces.
And according to :ref:`inner products with differential operators <inner_products_differential>`:

.. math::
	\int_\Omega \psi \nabla \cdot \vec{j} \, dv \approx \boldsymbol{\psi^T M_c D \, j}
	:label: derivation_examples_advection_diffusion_4b

where :math:`\boldsymbol{D}` is the :ref:`discrete divergence matrix <operators_differential_divergence>`.

For the next term in equation :eq:`derivation_examples_advection_diffusion_3`, we must take the
dot product of vectors :math:`\vec{u}` and :math:`\vec{w}`. Here, we let the corresponding
discrete representations :math:`\boldsymbol{u}` and  :math:`\boldsymbol{w}` live on the faces;
this is because we need the gradient of :math:`\boldsymbol{p}` to live on the faces. In doing so, we need a set of operations which
multiplies the components of the dot product and sums them at the cell center.
Where :math:`\boldsymbol{A_{fc}}` is the :ref:`scalar averaging matrix from faces to cell centers <operators_averaging>`
and :math:`c` = 1, 2 or 3 is the dimension of the problem, the inner product is approximated by:

.. math::
	\int_\Omega \psi (\vec{u} \cdot \vec{w}) \, dv \approx c \, \boldsymbol{\psi^T M_c A_{fc}} diag(\boldsymbol{u}) \, \boldsymbol{w} 
	:label: derivation_examples_advection_diffusion_4c

For the final term in :eq:`derivation_examples_advection_diffusion_3`, our inner product contains the delta function.
As a result:

.. math::
	\int_\Omega \psi q \, dv \approx \boldsymbol{\psi^T q} 
	:label: derivation_examples_advection_diffusion_4d

where :math:`\boldsymbol{q}` is a discrete representation for the integrated source term in each cell.
In this case, :math:`\boldsymbol{q_i}=s_0` at the center of the cell containing the source
It is zero for every other cell.

If we substitute the inner product approximations from expressions
:eq:`derivation_examples_advection_diffusion_4a`, :eq:`derivation_examples_advection_diffusion_4b`,
:eq:`derivation_examples_advection_diffusion_4c` and :eq:`derivation_examples_advection_diffusion_4d`
into equation :eq:`derivation_examples_advection_diffusion_3`, we obtain:

.. math::
	\boldsymbol{\psi^T M_c \, p_t} = \boldsymbol{\psi^T M_c D \, j} -
	c\, \boldsymbol{\psi^T M_c A_{fc}} \, diag(\boldsymbol{u}) \, \boldsymbol{w} + \boldsymbol{\psi^T q}
	:label: derivation_examples_advection_diffusion_5

Second Equation
^^^^^^^^^^^^^^^

In the second equation of :eq:`derivation_examples_advection_diffusion_2`, we use what we learned in
:ref:`inner products with differential operators <inner_products_differential>`.
Let :math:`\vec{f}` be a vector test function whose discrete representation :math:`\boldsymbol{f}` lives on the faces.
After using the vector identity :math:`\vec{f} \cdot \nabla p = \nabla \cdot p\vec{f} - p \nabla \cdot \vec{f}`
and applying the divergence theorem:

.. math::
    \int_\Omega \vec{f} \cdot \vec{w} = - \int_\Omega p \nabla \cdot \vec{f} \, dv + \oint_{\partial \Omega} p \hat{n} \cdot \vec{f} \, da
    :label: derivation_examples_advection_diffusion_6

The discrete approximation is given by:

.. math::
	\boldsymbol{f^T M_f \, w} = - \boldsymbol{f^T D^T M_c \, p + f^T B \, p}
	:label: derivation_examples_advection_diffusion_7

where :math:`\boldsymbol{B}` is a sparse matrix that imposes boundary conditions correctly on :math:`p`.

Third Equation
^^^^^^^^^^^^^^

In the third equation of :eq:`derivation_examples_advection_diffusion_2`, we use what we learned in
:ref:`inner products with contitutive relationships <inner_products_isotropic>`; assume the diffusibility :math:`\alpha` is linear isotropic.
The inner product with a vector test function :math:`\vec{f}` whose discrete representation :math:`\boldsymbol{f}` lives on the faces
is given by:

.. math::
	\int_\Omega \vec{f} \cdot \alpha^{\! -1} \vec{j} \, dv = \int_\Omega \vec{f} \cdot \vec{w} \, dv
	:label: derivation_examples_advection_diffusion_8

Our formulation defines the diffusivity in terms of its inverse. As a result, the approximation of the inner
products is given by:

.. math::
	\boldsymbol{f^T M_\alpha \, j} = \boldsymbol{f^T M_f w}
	:label: derivation_examples_advection_diffusion_9

where :math:`\boldsymbol{M_\alpha}` is the :ref:`inner product matrix at faces for the reciprocal of the diffusivity <inner_products_isotropic_reciprocal>`.

Combining the Expressions
^^^^^^^^^^^^^^^^^^^^^^^^^

Here, we substitute the discrete representations of the inner products from expressions
:eq:`derivation_examples_advection_diffusion_7` and :eq:`derivation_examples_advection_diffusion_9`
into :eq:`derivation_examples_advection_diffusion_5` and factor like-terms.

Let :math:`\boldsymbol{\tilde{G}}` represent a modified gradient operator with the boundary conditions implemented:

.. math::
	\boldsymbol{\tilde{G}} = \boldsymbol{-D^T M_c + B}
	:label: derivation_examples_advection_diffusion_10

The system of equations discretized in space is given by:

.. math::
	\boldsymbol{p_t} = \boldsymbol{\big [ D \, M_\alpha^{-1} \tilde{G}} -
	c\, \boldsymbol{A_{fc}} diag(\boldsymbol{u}) \, \boldsymbol{M_f^{-1} \tilde{G} \big ] \, p} + \boldsymbol{M_c^{-1} \, q}
	:label: derivation_examples_advection_diffusion_11

Discretization in Time
----------------------

To discretize in time, let us re-express equations :eq:`derivation_examples_advection_diffusion_11` as:

.. math::
	\boldsymbol{p_t} = \boldsymbol{- M \, p + s}
	:label: derivation_examples_advection_diffusion_12

where

.. math::
	\boldsymbol{M} = - \boldsymbol{D \, M_\alpha^{-1} \tilde{G}} +  
	c\, \boldsymbol{A_{fc}} diag(\boldsymbol{u}) \, \boldsymbol{M_f^{-1} \tilde{G}}
	:label: derivation_examples_advection_diffusion_13

and

.. math::
	\boldsymbol{s} = \boldsymbol{M_c^{-1} \, q}
	:label: derivation_examples_advection_diffusion_14

There are a multitude of ways in which discretization in time can be implemented.
A stable and easy method to implement is the backward Euler.
By implementing the backward Euler, we must solve the following linear system
at each time step :math:`k`:

.. math::
	\big [ \boldsymbol{I} + \Delta t \, \boldsymbol{M} \big ] \, \boldsymbol{p}^{k+1} = \boldsymbol{p}^k + \Delta t \, \boldsymbol{s}
	:label: derivation_examples_advection_diffusion_14

where :math:`\boldsymbol{I}` is the identity matrix and :math:`\Delta t` is the step length.