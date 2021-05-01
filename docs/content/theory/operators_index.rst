.. _operators_index:

Operators
*********

To approximate numerical solutions to partial differential equations using the finite volume method,
we must be able to construct discrete approximations to the following mathematical operations:

    - Interpolation
    - Averaging
    - Differential (gradient, divergence, curl)

Let any one of the aforementioned operations be defined as a mapping such that :math:`w = \mathbb{M}[u]`.
Where :math:`\boldsymbol{u}` is a vector containing the discrete representation of :math:`u` on the mesh,
and :math:`\boldsymbol{w}` is a vector containing the discrete represention of :math:`w` on the mesh,
we can approximate the mapping as:  

.. math::
	\boldsymbol{w} \approx \boldsymbol{M \, u}

where :math:`\boldsymbol{M}` is a sparse matrix. Thus for each operator, the mapping is approximated by
constructing a sparse matrix and performing a matrix-vector product. Subsequent sections are devoted to
the general formation of these matrices.


**Contents:**

.. toctree::
    :maxdepth: 1

    operators_interpolation
    operators_averaging
    operators_differential

**Tutorials:**

- :ref:`Interpolation Operators <sphx_glr_tutorials_operators_0_interpolation.py>`
- :ref:`Averaging Operators <sphx_glr_tutorials_operators_1_averaging.py>`
- :ref:`Differential Operators <sphx_glr_tutorials_operators_2_differential.py>`