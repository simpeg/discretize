.. _operators_averaging:

Interpolation and Averaging
***************************

Here, we provide the background theory for how discrete interpolation and averaging matricies are formed.
Interpolation is required when a discrete quantity is known on the mesh (centers, nodes, edges or faces), but we would like to estimate its
value at a point within the continuous domain. Averaging is required when quantities that live on different
parts of the mesh need to be added, subtracted, multiplied or divided.

**Tutorials:** 

    - :ref:`tutorial for constructing and applying averaging operators <sphx_glr_tutorials_operators_1_averaging.py>`



Interpolation
=============

Presently there is an extensive set of interpolation methods (e.g. polynomial, spline, piecewise constant).
One of the most effective and widely used interpolation methods is linear interpolation.
The *discretize* package primarily uses linear interpolation because 1) it is very fast, and 2) higher order
interpolation methods require the construction of matricies which are less sparse.
Here, we discuss how a sparse matrix can be formed which interpolates the discrete values to
a set of points in continuous space.

The formulation for linear interpolation is adequately presented on Wikipedia, see:

	- `Linear Interpolation (1D) <https://en.wikipedia.org/wiki/Linear_interpolation>`__
	- `Bilinear Interpolation (2D) <https://en.wikipedia.org/wiki/Bilinear_interpolation>`__
	- `Trilinear Interpolation (3D) <https://en.wikipedia.org/wiki/Trilinear_interpolation>`__

Interpolation Matrix in 1D
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us define a 1D mesh that contains 8 cells of arbitrary width;
the mesh is shown in the figure below. The width of each cell is
given by :math:`h_i`. The location of each node is given by :math:`x_i`.

.. image:: ../../images/interpolation_1d.png
    :align: center
    :width: 600

Now let :math:`u(x)` be a function whose values are known at the nodes;
i.e. :math:`u_i = u(x_i)`.
The approximate value of the function at location :math:`x^*` 
using linear interpolation is given by:

.. math::
	u(x^*) \approx u_3 + \Bigg ( \frac{u_4 - u_3}{h_3} \Bigg ) (x^* - x_3)
	:label: operators_averaging_interpolation_1d


Suppose now that we organize the known values of :math:`u(x)` at the nodes
into a vector of the form:

.. math::
	\boldsymbol{u} = \begin{bmatrix} u_0 & u_1 & u_2 & u_3 & u_4 & u_5 & u_6 & u_7 & u_8 \end{bmatrix}^T

If we define a matrix consisting of a single row:

.. math::
	A_0 = \begin{bmatrix} 0 & 0 & 0 & a_3 & a_4 & 0 & 0 & 0 & 0 \end{bmatrix}

where

.. math::
	a_3 = 1 - \frac{x^* - x_3}{h_3} \;\;\;\;\; \textrm{and} \;\;\;\;\; a_4 = \frac{x^* - x_3}{h_3}

then

.. math::
	u(x^*) \approx A_0 \, \boldsymbol{u}

For a single location, we have just seen how a linear operator can be constructed to
compute the interpolation using a matrix vector-product.

Now consider the case where you would like to interpolate the function from the nodes to
*M* arbitrary locations within the boundaries of the mesh.
For each location, we simply construct the corresponding row in the interpolation matrix.
Where :math:`\boldsymbol{u^*}` is a vector containing the approximations of :math:`u(x)` at :math:`M`
locations:

.. math::
	(\boldsymbol{u^*}) \approx \boldsymbol{A\, u} \;\;\;\; \textrm{where} \;\;\;\; \begin{bmatrix} A_0 \\ A_1 \\ \vdots \\ A_{M-1} \end{bmatrix}
	:label: operators_averaging_interpolation_matrix

where :math:`\boldsymbol{A}` is a sparse matrix whose rows contain a maximum of 2 non-zero elements.
The size of :math:`\boldsymbol{A}` is the number of locations by the number of nodes.
For seven locations (:math:`x^* = 3,1,9,2,5,2,8`) and our mesh (9 nodes),
the non-zero elements of the interpolation matrix are illustrated below.

.. image:: ../../images/interpolation_1d_sparse.png
    :align: center
    :width: 250


**What if the function is defined at cell centers?**

Here we let :math:`\bar{x}_i` define the center locations
for cells 0 through 7, and we let :math:`\bar{u}_i = u(\bar{x}_i)`.
In this case, the approximation defined in expression :eq:`operators_averaging_interpolation_1d` is replaced by:

.. math::
	u(x^*) \approx \bar{u}_3 + 2 \Bigg ( \frac{\bar{u}_4 - \bar{u}_3}{h_3 + h_4} \Bigg ) (x^* - \bar{x}_3)

For an arbitrary number of locations, we can construct an interpolation matrix similar to that shown
in expression :eq:`operators_averaging_interpolation_1d`. In this case however, the size of
:math:`\boldsymbol{A}` is the number of locations by the number of cells. Note that we **cannot**
interpolate at locations between the first or last cell center and the boundaries of the mesh
for quantities defined at cell centers.


Interpolation Matrix in 2D and 3D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In 1D, the location of the interpolated quantity lies between 2 nodes or cell centers.
In 2D however, the location of the interpolated quantity lies within 4 nodes or cell centers.

.. image:: ../../images/interpolation_2d.png
    :align: center
    :width: 300

Let :math:`(x^*, y^*)` be within a cell whose nodes are located at
:math:`(x_1, y_1)`, :math:`(x_2, y_1)`, :math:`(x_1, y_2)` and :math:`(x_2, y_2)`.
If we define :math:`u_0 = u(x_1, y_1)`, :math:`u_1 = u(x_2, y_1)`, :math:`u_2 = u(x_1, y_2)` and
:math:`u_3 = u(x_2, y_2)`, then

.. math::
	u(x^*, y^*) \approx a_0 u_0 + a_1 u_1 + a_2 u_2 + a_3 u_3

where :math:`a_0`, :math:`a_1`, :math:`a_2` and :math:`a_3` are coefficients determined from equations
governing `bilinear interpolation <https://en.wikipedia.org/wiki/Bilinear_interpolation>`__ .
These coefficients represent the 4 non-zero values within the corresponding row of the interpolation matrix :math:`\boldsymbol{A}`.

Where the values of :math:`u(x,y)` at all nodes are organized into a single vector :math:`\boldsymbol{u}`,
and :math:`\boldsymbol{u^*}` is a vector containing the approximations of :math:`u(x,y)` at an arbitrary number of locations:

.. math::
	\boldsymbol{u^*} \approx \boldsymbol{A\, u}

In each row, the position of the non-zero elements :math:`a_0`, :math:`a_1`, :math:`a_2` and :math:`a_3`
corresponds to the indecies of the 4 nodes comprising a specific cell.
Once again the shape of :math:`\boldsymbol{A}` is the number of locations by the number of nodes.

**What if the function is defined at cell centers?**

A similar result can be obtained by interpolating a function define at cell centers.
In this case, we let :math:`(x^*, y^*)` lie within 4 cell centers located at
:math:`(\bar{x}_1, \bar{y}_1)`, :math:`(\bar{x}_2, \bar{y}_1)`, :math:`(\bar{x}_1, \bar{y}_2)` and :math:`(\bar{x}_2, \bar{y}_2)`.

.. math::
	u(x^*, y^*) \approx a_0 \bar{u}_0 + a_1 \bar{u}_1 + a_2 \bar{u}_2 + a_3 \bar{u}_3

The size of the resulting interpolation matrix is the number of locations by number of cells.

**What about the 3D case?**

The derivation for the 3D case is effectively the same, except each cell is defined by 8 nodes and
thus:

.. math::
	u(x^*, y^*, z^*) \approx \sum_{k=0}^7 a_k u_k

This creates an interpolation matrix with 8 non-zero entries per row.

Scalars vs. Vectors
^^^^^^^^^^^^^^^^^^^



Averaging
=========





