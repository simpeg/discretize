.. _operators_interpolation:

Interpolation
*************

Interpolation is required when a discrete quantity is known on the mesh (centers, nodes, edges or faces),
but we would like to estimate its value at locations within the continuous domain.
Here, we discuss how a sparse matrix :math:`\boldsymbol{P}` can be formed which interpolates the discrete values to
a set of locations in continuous space. Where :math:`\boldsymbol{u}` is vector that stores
the values of a discrete quantity on the mesh (centers, nodes, faces or edges) and
:math:`\boldsymbol{w}` is a vector containing the interpolated quantity at a set of locations,
we look to construct an interpolation matrix such that:

.. math::
	\boldsymbol{w} = \boldsymbol{P \, u}

Presently there is an extensive set of interpolation methods (e.g. polynomial, spline, piecewise constant).
One of the most effective and widely used interpolation methods is linear interpolation.
The *discretize* package primarily uses linear interpolation because 1) it is very fast, and 2) higher order
interpolation methods require the construction of matricies which are less sparse.
The formulation for linear interpolation is adequately presented on Wikipedia, see:

	- `Linear Interpolation (1D) <https://en.wikipedia.org/wiki/Linear_interpolation>`__
	- `Bilinear Interpolation (2D) <https://en.wikipedia.org/wiki/Bilinear_interpolation>`__
	- `Trilinear Interpolation (3D) <https://en.wikipedia.org/wiki/Trilinear_interpolation>`__

**Tutorial:** :ref:`tutorial for constructing and applying interpolation operators <sphx_glr_tutorials_operators_0_interpolation.py>`

Interpolation Matrix in 1D
==========================

Let us define a 1D mesh that contains 8 cells of arbitrary width.
The mesh is illustrated in the figure below. The width of each cell is
defined as :math:`h_i`. The location of each node is defined as :math:`x_i`.

.. figure:: ../../images/interpolation_1d.png
    :align: center
    :width: 600
    :name: operators_interpolation_1d

    Tensor mesh in 1D.

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

If we define a row:

.. math::
	\boldsymbol{p_0} = \begin{bmatrix} 0 & 0 & 0 & a_3 & a_4 & 0 & 0 & 0 & 0 \end{bmatrix}

where

.. math::
	a_3 = 1 - \frac{x^* - x_3}{h_3} \;\;\;\;\; \textrm{and} \;\;\;\;\; a_4 = \frac{x^* - x_3}{h_3}

then

.. math::
	u(x^*) \approx \boldsymbol{p_0 \, u}

For a single location, we have just seen how a linear operator can be constructed to
compute the interpolation using a matrix vector-product.

Now consider the case where you would like to interpolate the function from the nodes to
an arbitrary number of locations within the boundaries of the mesh.
For each location, we simply construct the corresponding row in the interpolation matrix.
Where :math:`\boldsymbol{u^*}` is a vector containing the approximations of :math:`u(x)` at :math:`M`
locations:

.. math::
	\boldsymbol{u^*} \approx \boldsymbol{P\, u} \;\;\;\;\;\; \textrm{where} \;\;\;\;\;\;
	\boldsymbol{P} = \begin{bmatrix} \cdots \;\; \boldsymbol{p_0} \;\; \cdots \\
	\cdots \;\; \boldsymbol{p_1} \;\; \cdots \\ \vdots \\
	\cdots \, \boldsymbol{p_{M-1}} \, \cdots \end{bmatrix}
	:label: operators_averaging_interpolation_matrix

:math:`\boldsymbol{P}` is a sparse matrix whose rows contain a maximum of 2 non-zero elements.
The size of :math:`\boldsymbol{P}` is the number of locations by the number of nodes.
For seven locations (:math:`x^* = 3,1,9,2,5,2,8`) and our mesh (9 nodes),
the non-zero elements of the interpolation matrix are illustrated below.

.. figure:: ../../images/interpolation_1d_sparse.png
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
:math:`\boldsymbol{P}` is the number of locations by the number of cells. Note that we **cannot**
interpolate at locations between the first or last cell center and the boundaries of the mesh
for quantities defined at cell centers.


Interpolation Matrix in 2D and 3D
=================================

In 1D, the location of the interpolated quantity lies between 2 nodes or cell centers.
In 2D however, the location of the interpolated quantity lies within 4 nodes or cell centers.

.. figure:: ../../images/interpolation_2d.png
    :align: center
    :width: 300

    A tensor mesh in 2D denoting interpolation from nodes (blue) and cell centers (red).

Let :math:`(x^*, y^*)` be within a cell whose nodes are located at
:math:`(x_1, y_1)`, :math:`(x_2, y_1)`, :math:`(x_1, y_2)` and :math:`(x_2, y_2)`.
If we define :math:`u_0 = u(x_1, y_1)`, :math:`u_1 = u(x_2, y_1)`, :math:`u_2 = u(x_1, y_2)` and
:math:`u_3 = u(x_2, y_2)`, then

.. math::
	u(x^*, y^*) \approx a_0 u_0 + a_1 u_1 + a_2 u_2 + a_3 u_3

where :math:`a_0`, :math:`a_1`, :math:`a_2` and :math:`a_3` are coefficients determined from equations
governing `bilinear interpolation <https://en.wikipedia.org/wiki/Bilinear_interpolation>`__ .
These coefficients represent the 4 non-zero values within the corresponding row of the interpolation matrix :math:`\boldsymbol{P}`.

Where the values of :math:`u(x,y)` at all nodes are organized into a single vector :math:`\boldsymbol{u}`,
and :math:`\boldsymbol{u^*}` is a vector containing the approximations of :math:`u(x,y)` at an arbitrary number of locations:

.. math::
	\boldsymbol{u^*} \approx \boldsymbol{P\, u}
	:label: operators_interpolation_general

In each row of :math:`\boldsymbol{P}`, the position of the non-zero elements :math:`a_0`, :math:`a_1`, :math:`a_2` and :math:`a_3`
corresponds to the indecies of the 4 nodes comprising a specific cell.
Once again the shape of :math:`\boldsymbol{P}` is the number of locations by the number of nodes.

**What if the function is defined at cell centers?**

A similar result can be obtained by interpolating a function define at cell centers.
In this case, we let :math:`(x^*, y^*)` lie within 4 cell centers located at
:math:`(\bar{x}_1, \bar{y}_1)`, :math:`(\bar{x}_2, \bar{y}_1)`, :math:`(\bar{x}_1, \bar{y}_2)` and :math:`(\bar{x}_2, \bar{y}_2)`.

.. math::
	u(x^*, y^*) \approx a_0 \bar{u}_0 + a_1 \bar{u}_1 + a_2 \bar{u}_2 + a_3 \bar{u}_3

The resulting interpolation is defined similar to expression :eq:`operators_interpolation_general`.
However the size of the resulting interpolation matrix is the number of locations by number of cells.

**What about for 3D case?**

The derivation for the 3D case is effectively the same, except 8 node or center locations must
be used in the interpolation. Thus:

.. math::
	u(x^*, y^*, z^*) \approx \sum_{k=0}^7 a_k u_k

This creates an interpolation matrix :math:`\boldsymbol{P}` with 8 non-zero entries per row.
To learn how to compute the value of the coefficients :math:`a_k`,
see `trilinear interpolation (3D) <https://en.wikipedia.org/wiki/Trilinear_interpolation>`__

Interpolation of Vectors
========================

Scalar quantities are discretized to live at nodes or cell centers, whereas the
components of vectors are discretized to live on their respective faces or edges;
see :ref:`where quantities live <meshes_index_quantities>`. 

.. figure:: ../../images/interpolation_2d_vectors.png
    :align: center
    :width: 600

    A tensor mesh in 2D denoting interpolation from faces (left) and edges (right).

Let :math:`\vec{u} (x,y)` be a 2D vector function that is known on the faces of the mesh;
that is, :math:`u_x` lives on the x-faces and :math:`u_y` lives on the y-faces. 
Note that in the above figure, the x-faces and y-faces both form tensor grids.
If we want to approximate the components of the vector at a location :math:`(x^*,y^*)`,
we simply need to treat each component as a scalar function and interpolate it separately.

Where :math:`u_{x,i}` represents the x-component of :math:`\vec{u} (x,y)` on a face :math:`i` being used for the interpolation,
the approximation of the x-component at :math:`(x^*, y^*)` has the form:

.. math::
	u_x(x^*, y^*) \approx a_0 u_{x,0} + a_1 u_{x,1} + a_2 u_{x,2} + a_3 u_{x,3}
	:label: operators_interpolation_xvec_coef

For the the y-component, we have a similar representation:

.. math::
	u_y(x^*, y^*) \approx b_0 u_{y,0} + b_1 u_{y,1} + b_2 u_{y,2} + b_3 u_{y,3}

Where :math:`\boldsymbol{u}` is a vector that organizes the discrete components of :math:`\vec{u} (x,y)` on cell faces,
and :math:`\boldsymbol{u^*}` is a vector organizing the components of the approximations of :math:`\vec{u}(x,y)` at an arbitrary number of locations,
the interpolation matrix :math:`\boldsymbol{P}` is defined by:

.. math::
	\boldsymbol{u^*} \approx \boldsymbol{P \, u}
	:label: operators_interpolation_2d_sys

where

.. math::
	\boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \end{bmatrix}
	\;\;\textrm{,}\;\;\;\;
	\boldsymbol{u^*} = \begin{bmatrix} \boldsymbol{u_x^*} \\ \boldsymbol{u_y^*} \end{bmatrix}
	\;\;\;\;\textrm{and}\;\;\;\;
	\boldsymbol{P} = \begin{bmatrix} \boldsymbol{P_x} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{P_y} \end{bmatrix}

The interpolation matrix :math:`\boldsymbol{P}` is a sparse block-diagonal matrix.
The size of the interpolation matrix is the number of locations by the number of faces in the mesh.

**What if we want to interpolate from edges?**

In this case, the derivation is effectively the same.
However, the locations used for the interpolation are different and
:math:`\boldsymbol{u}` is now a vector that organizes the discrete components of :math:`\vec{u} (x,y)` on cell edges.


**What if we are interpolating a 3D vector?**

In this case, there are 8 face locations or 8 edge locations that are used to approximate
:math:`\vec{u}(x,y,z)` at each location :math:`(x^*, y^*, z^*)`.
Similar to expression :eq:`operators_interpolation_xvec_coef` we have:

.. math::
	\begin{align}
	u_x(x^*, y^*, z^*) & \approx \sum_{i=1}^7 a_i u_{x,i} \\
	u_y(x^*, y^*, z^*) & \approx \sum_{i=1}^7 b_i u_{y,i} \\
	u_z(x^*, y^*, z^*) & \approx \sum_{i=1}^7 c_i u_{z,i}
	\end{align}

The interpolation can be expressed similar to that in equation :eq:`operators_interpolation_2d_sys`,
however:

.. math::
	\boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_z} \end{bmatrix}
	\;\;\textrm{,}\;\;\;\;
	\boldsymbol{u^*} = \begin{bmatrix} \boldsymbol{u_x^*} \\ \boldsymbol{u_y^*} \\ \boldsymbol{u_z^*} \end{bmatrix}
	\;\;\;\;\textrm{and}\;\;\;\;
	\boldsymbol{P} = \begin{bmatrix} \boldsymbol{P_x} & \boldsymbol{0} & \boldsymbol{0} \\
	\boldsymbol{0} & \boldsymbol{P_y} & \boldsymbol{0} \\
	\boldsymbol{0} & \boldsymbol{0} & \boldsymbol{P_z} 
	\end{bmatrix}

