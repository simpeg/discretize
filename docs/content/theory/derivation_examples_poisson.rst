.. _derivation_examples_poisson:

Poisson Equation with Zero Neumann Boundary Condition
*****************************************************

Here we provide the derivation for solving the Poisson equation with zero Neumann boundary conditions using the finite volume method.
Derivations are provided for discretization of the solution on both the nodes and at cell centers.
Key lessons include:

	- Differences between solving the problem on the nodes and the cell centers
	- Basic discretization of point sources
	- Choosing the right approach for natural boundary conditions
	- Implementing the zero Neumann condition when the discrete boundary condition term is not zero

For our example, we consider Gauss's law of electrostatics.
Our goal is to compute the electric potential (:math:`\phi`) and electric fields (:math:`\boldsymbol{e}`) that result from
a positive and a negative point charge separated by some distance.

**Tutorial:** :ref:`Poisson Equation with Zero Neumann Boundary Condition <sphx_glr_tutorials_pde_1_poisson.py>`

Setup
-----

Starting with Gauss's law and Faraday's law:
    
.. math::
    &\nabla \cdot \vec{e} = \frac{\rho}{\epsilon_0} \\
    &\nabla \times \vec{e} = \boldsymbol{0} \;\;\; \Rightarrow \;\;\; \vec{e} = -\nabla \phi \\
    &\textrm{s.t.} \;\;\; \hat{n} \cdot \vec{e} \, \Big |_{\partial \Omega} = -\frac{\partial \phi}{\partial n} \, \Big |_{\partial \Omega} = 0
    :label: derivation_examples_electrostatics_1
    
where :math:`\rho` is the charge density and :math:`\epsilon_0` is the permittivity of free space.
The Neumann boundary condition on :math:`\phi` implies no electric flux leaves the system.
For 2 point charges of equal and opposite sign, the charge density is given by:

.. math::
    \rho = \rho_0 \big [ \delta ( \boldsymbol{r_+}) - \delta (\boldsymbol{r_-} ) \big ]
    :label: derivation_examples_electrostatics_2

To solve this problem numerically, we first
take the inner product of each equation with an appropriate test function.
Where :math:`\psi` is a scalar test function and :math:`\vec{u}` is a
vector test function:

.. math::
    \int_\Omega \psi (\nabla \cdot \vec{e}) \, dv = \frac{1}{\epsilon_0} \int_\Omega \psi \rho \, dv
    :label: derivation_examples_electrostatics_3

and

.. math::
    \int_\Omega \vec{u} \cdot \vec{e} \, dv = - \int_\Omega \vec{u} \cdot (\nabla \phi ) \, dv
    :label: derivation_examples_electrostatics_4


In the case of Gauss' law, we have a volume integral containing the Dirac delta function.
Thus expression :eq:`derivation_examples_electrostatics_3` becomes:

.. math::
    \int_\Omega \psi (\nabla \cdot \vec{e}) \, dv = \frac{1}{\epsilon_0} \psi \, q
    :label: derivation_examples_electrostatics_5

where :math:`q` represents an integrated charge density.
To apply the finite volume method, we must choose whether to discretize the scalar quantity :math:`\phi` at the nodes or cell centers.

Electic Potential on the Nodes
------------------------------

Here, we let :math:`\boldsymbol{\phi}` be the discrete representation of the electic potential :math:`\phi` on the nodes
and :math:`\boldsymbol{e}` be the discrete representation of the electic field :math:`\vec{e}` on the edges.

**First Expression:**

To implement the finite volume approach, we begin by approximating the inner products in expression :eq:`derivation_examples_electrostatics_4`.
The left-hand side can be approximated according to :ref:`basic inner products <inner_products_basic>` .
And in :ref:`inner products with differential operators <inner_products_differential>`, we learned how to approximate the right-hand side.
The discrete representation of expression :eq:`derivation_examples_electrostatics_4` is therefore given by:

.. math::
	\boldsymbol{u^T M_e \, e} = - \boldsymbol{u^T M_e G \, \phi}
	:label: derivation_examples_electrostatics_6

where

	- :math:`\boldsymbol{M_e}` is the :ref:`inner product matrix at edges <inner_products_basic>`
	- :math:`\boldsymbol{G}` is the :ref:`discrete gradient operator <inner_products_differential>`

**Second Expression:**

Now we approximate the inner products in expression :eq:`derivation_examples_electrostatics_5`.
For the left-hand side, we must use the identity :math:`\psi \nabla \cdot \vec{e} = \nabla \cdot \psi\vec{e} - \vec{e} \cdot \nabla \psi`
and apply the divergence theorem such that expression :eq:`derivation_examples_electrostatics_5` becomes:

.. math::
    - \int_\Omega \vec{e} \cdot \nabla \psi \, dv + \oint_{\partial \Omega} \psi (\hat{n} \cdot \vec{e}) \, da = \frac{1}{\epsilon_0} \psi \, q
    :label: derivation_examples_electrostatics_7

Since :math:`\hat{n} \cdot \vec{e}` is zero on the boundary, the surface integral is equal to zero.
The left-hand side can be approximated according to :ref:`inner products with differential operators <inner_products_differential>`.
:math:`\boldsymbol{\psi}` and :math:`\boldsymbol{q}` are defined such that their discrete representations :math:`\psi` and :math:`\rho`
must live on the nodes. The discrete approximation to expression :eq:`derivation_examples_electrostatics_7` is given by:

.. math::
	- \boldsymbol{\psi^T G^T M_e \, e} = \frac{1}{\epsilon_0} \boldsymbol{\psi^T q}
	:label: derivation_examples_electrostatics_8

where :math:`\boldsymbol{q}` is a discrete representation of the integrated charge density.

The easiest way to discretize the source is to let :math:`\boldsymbol{q_i}=\rho_0` at the nearest node to the positive charge and
let :math:`\boldsymbol{q_i}=-\rho_0` at the nearest node to the negative charge.
The value is zero for all other nodes.

**Discretized System:**

By combining the discrete representations from expressions :eq:`derivation_examples_electrostatics_6` and :eq:`derivation_examples_electrostatics_8`
we obtain:

.. math::
	\boldsymbol{G^T M_e G \, \phi} = \frac{1}{\epsilon_0} \boldsymbol{q}
	:label: derivation_examples_electrostatics_9

Let :math:`\boldsymbol{A} = \boldsymbol{G^T M_e G}`.
The linear system has a single null vector.
To remedy this, we set a reference potential on the boundary
by setting :math:`A_{0,0} = 1` and by setting all other values in the row to 0.
Once the electric potential at nodes has been computed, the electric field on the edges can be computed using expression :eq:`derivation_examples_electrostatics_6`:

.. math::
	\boldsymbol{e} = - \boldsymbol{G \, \phi}


Electic Potential at Cell Centers
---------------------------------

Here, we let :math:`\boldsymbol{\phi}` be the discrete representation of the electic potential :math:`\phi` at cell centers
and :math:`\boldsymbol{e}` be the discrete representation of the electic field :math:`\vec{e}` on the faces.
It is acceptable to discretize the electric field on the faces in this case because the dielectric permittivity of the domain
is constant and the electric field at the faces is continuous.

**First Expression:**

To implement the finite volume approach, we begin by approximating the inner products in expression :eq:`derivation_examples_electrostatics_5`.
The left-hand side can be approximated according to :ref:`inner products with differential operators <inner_products_differential>`.
Where :math:`\boldsymbol{\psi}` and :math:`\boldsymbol{q}` are discrete representations of :math:`\psi` and :math:`\rho` living at cell centers:

.. math::
    \boldsymbol{\psi^T M_c D e} = \frac{1}{\epsilon_0} \boldsymbol{\psi^T q}
    :label: derivation_examples_electrostatics_10

where

	- :math:`\boldsymbol{M_c}` is the :ref:`inner product matrix at cell centers <inner_products_basic>`
	- :math:`\boldsymbol{D}` is the :ref:`discrete divergence operator <inner_products_differential>`
	- :math:`\boldsymbol{q}` is a discrete representation for the integrated charge density for each cell.

In this case, :math:`\boldsymbol{q_i}=\rho_0` at the center of the cell containing the positive charge and
:math:`\boldsymbol{q_i}=-\rho_0` at the center of the cell containing the negative charge.
It is zero for every other cell.



**Second Expression:**

We now approximate the inner products in expression :eq:`derivation_examples_electrostatics_4`.
The left-hand side can be approximated according to :ref:`basic inner products <inner_products_basic>` .
And in :ref:`inner products with differential operators <inner_products_differential>`, we learned how to approximate the right-hand side.
For the right-hand side, we must use the identity :math:`\vec{u} \cdot \nabla \phi = \nabla \cdot \phi\vec{u} - \phi \nabla \cdot \vec{u}`
and apply the divergence theorem such that expression :eq:`derivation_examples_electrostatics_4` becomes:

.. math::
    \int_\Omega \vec{u} \cdot \vec{e} \, dv = \int_\Omega \phi \nabla \cdot \vec{u} \, dv - \oint_{\partial \Omega} \phi \hat{n} \cdot \vec{u} \, da
    :label: derivation_examples_electrostatics_11

According to expression :eq:`derivation_examples_electrostatics_1`, :math:`\hat{n} \cdot \vec{e}`,
:math:`\frac{\partial \phi}{\partial n} = 0 on the boundaries.
To accurately compute the electric potentials at cell centers, we must implement the boundary conditions such that:

.. math::
	\boldsymbol{u^T M_f \, e} = \boldsymbol{u^T D^T M_c \, \phi} - \boldsymbol{u^T B \, \phi} = - \boldsymbol{\tilde{G} \, \phi}
	:label: derivation_examples_electrostatics_12

where

	- :math:`\boldsymbol{M_c}` is the :ref:`inner product matrix at cell centers <inner_products_basic>`
	- :math:`\boldsymbol{M_f}` is the :ref:`inner product matrix at faces <inner_products_basic>`
	- :math:`\boldsymbol{D}` is the :ref:`discrete divergence operator <inner_products_differential>`
	- :math:`\boldsymbol{B}` is a sparse matrix that imposes the Neumann boundary condition
	- :math:`\boldsymbol{\tilde{G}} = - \boldsymbol{D^T M_c} + \boldsymbol{B}` acts as a modified gradient operator with boundary conditions included

**Discretized System:**

By combining the discrete representations from expressions :eq:`derivation_examples_electrostatics_10` and :eq:`derivation_examples_electrostatics_12`
we obtain:

.. math::
	- \boldsymbol{M_c D M_f^{-1} \tilde{G} \, \phi} = \frac{1}{\epsilon_0} \boldsymbol{q}
	:label: derivation_examples_electrostatics_13

Once the electric potential at cell centers has been computed, the electric field on the faces can be computed using expression :eq:`derivation_examples_electrostatics_12`:

.. math::
	\boldsymbol{e} = - \boldsymbol{M_f^{-1} \tilde{G} \, \phi}


