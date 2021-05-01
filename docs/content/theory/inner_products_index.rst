.. _inner_products_index:

Inner Products
**************

Inner products provide the building blocks for discretizing and solving PDEs with the
finite volume method (:cite:`haber2014,HymanShashkov1999`). For scalar quantities :math:`\psi` and :math:`\phi`, the inner
product is given by:

.. math::
    (\psi , \phi ) = \int_\Omega \psi \, \phi \, dv

And for two vector quantities :math:`\vec{u}` and :math:`\vec{v}`, the inner product is
given by:

.. math::
    (\vec{u}, \vec{v}) = \int_\Omega \vec{u} \cdot \vec{v} \, dv

When implementing the finite volume method, we construct discrete approximations to inner products.
The approximations to inner products are combined and simplified to form a linear system in terms
of an unknown variable, then solved. The approximation to each inner product depends on the quantities
and differential operators present in the inner product. Here, we demonstrate how to formulate the
discrete approximation for several classes of inner products.

**Contents:**

.. toctree::
    :maxdepth: 1

    inner_products_basic
    inner_products_isotropic
    inner_products_anisotropic
    inner_products_differential
    inner_products_boundary_conditions

**Tutorials:**

- :ref:`Basic Inner Products <sphx_glr_tutorials_inner_products_1_basic.py>`
- :ref:`Inner Products with Constitutive Relationships <sphx_glr_tutorials_inner_products_2_physical_properties.py>`
- :ref:`Inner Products with Differential Operators <sphx_glr_tutorials_inner_products_3_calculus.py>`
- :ref:`Advanced Inner Product Examples <sphx_glr_tutorials_inner_products_4_advanced.py>`