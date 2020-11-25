.. _operators_averaging:

Averaging
*********

Here, we provide the background theory for how discrete averaging matricies are formed.
Averaging matrices are required when quantities that live on different
parts of the mesh need to be added, subtracted, multiplied or divided.

**Tutorials:** 

    - :ref:`tutorial for constructing and applying averaging operators <sphx_glr_tutorials_operators_1_averaging.py>`


Averaging Matrices in 1D
========================

Nodes to Cell Centers
^^^^^^^^^^^^^^^^^^^^^

Let us define a 1D mesh where cell center locations are defined at :math:`x_i`
and node locations are defined at :math:`x_{i+\frac{1}{2}}`.
The widths of the cells are given by :math:`\Delta x_i`.

.. image:: ../../images/averaging_1d.png
    :align: center
    :width: 600

Now let :math:`u(x)` be a scalar function whose values are known at the nodes; i.e. :math:`u_{i+\frac{1}{2}} = u \big (x_{i+\frac{1}{2}} \big )`.
The average at the center of cell :math:`i` is given by:

.. math::
	\bar{u}_i = \frac{u_{i-\frac{1}{2}} + u_{i+\frac{1}{2}}}{2}

Our goal is to construct a matrix :math:`\boldsymbol{A}` that averages the values
at the nodes and places them at the cell centers:

.. math::
	\bar{\boldsymbol{u}} = \boldsymbol{A \, u}

where :math:`\boldsymbol{u}` is a vector that stores the known values of :math:`u(x)` at the nodes, 
and :math:`\bar{\boldsymbol{u}}` is a vector that stores the averages at cell centers.
For the entire mesh, the averaging matrix is given by

.. math::
	\boldsymbol{A} = \begin{bmatrix}
	\frac{1}{2} & \frac{1}{2} & 0 & 0 & \cdots & 0 & 0 \\
	0 & \frac{1}{2} & \frac{1}{2} & 0 & \cdots & 0 & 0 \\
	0 & 0 & \frac{1}{2} & \frac{1}{2} & \cdots & 0 & 0 \\
	\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
	0 & 0 & 0 & 0 & \cdots & \frac{1}{2} & \frac{1}{2}
	\end{bmatrix}
	:label: operators_averaging_n2c_1d

where :math:`\boldsymbol{A}` is a sparse matrix. Where :math:`N` is the number of cells in the mesh,
:math:`\boldsymbol{A}` is :math:`N` by :math:`N+1`.

Cell Centers to Nodes
^^^^^^^^^^^^^^^^^^^^^

Let us re-examine the figure illustrating the 1D mesh and assume we know the value of the function :math:`u(x)` at cell centers; i.e. :math:`u_i = u(x_i)`.
Note that the nodes are not equal distance from the cell centers on either side.
Therefore we cannot simply sum the values and divide by half. Instead we need to implement a weighted averaging.
At node :math:`i+\frac{1}{2}`, the average value is given by:

.. math::
	\bar{u}_{i+\frac{1}{2}} = \Bigg ( \frac{\Delta x_{i+1}}{\Delta x_i + \Delta x_{i+1}} \Bigg ) u_{i}
	+ \Bigg ( \frac{\Delta x_i}{\Delta x_i + \Delta x_{i+1}} \Bigg ) u_{i+1}

Our goal is to construct a matrix :math:`\bar{\boldsymbol{A}}` that averages the values
at the cell centers and places them at the nodes:

.. math::
	\bar{\boldsymbol{u}} = \bar{\boldsymbol{A}} \, \boldsymbol{u}

where :math:`\boldsymbol{u}` is a vector that stores the known values of :math:`u(x)` at cell centers, 
and :math:`\bar{\boldsymbol{u}}` is a vector that stores the averages at the nodes.
For the entire mesh, the averaging matrix is given by:

.. math::
	\bar{\boldsymbol{A}} = \begin{bmatrix}
	1 & 0 & 0 & 0 & \cdots & 0 & 0 \\
	a_1 & b_1 & 0 & 0 & \cdots & 0 & 0 \\
	0 & a_2 & b_2 & 0 & \cdots & 0 & 0 \\
	\vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots \\
	0 & 0 & 0 & 0 & \cdots & a_{N-1} & b_{N-1} \\
	0 & 0 & 0 & 0 & \cdots & 0 & 1
	\end{bmatrix} \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;
	\begin{split}
	a_i &= \frac{\Delta x_{i+1}}{\Delta x_i + \Delta x_{i+1}} \\
	& \\
	b_i &= \frac{\Delta x_i}{\Delta x_i + \Delta x_{i+1}}
	\end{split}
	:label: operators_averaging_c2n_1d

where :math:`\bar{\boldsymbol{A}}` is a sparse matrix. Where :math:`N` is the number of cells in the mesh,
:math:`\bar{\boldsymbol{A}}` is :math:`N+1` by :math:`N`. Note that :math:`\bar{A}_{0,0}` and :math:`\bar{A}_{N,N-1}`
are equal to 1. This is because cell center locations need to compute the average lie outside the mesh.
Therefore we take the nearest neighbour instead.


Averaging Matrices in 2D and 3D
===============================

Scalars from Nodes to Cell Centers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In 1D, the value of a function at 2 locations is used to compute the average.
In 2D however, the value of the function at 4 locations is needed.
Let us define a 2D mesh where cell center locations are defined by :math:`(x_i,y_j)`
and the node locations are defined by :math:`\Big ( x_{i+\frac{1}{2}}, y_{i+\frac{1}{2}} \Big )`.
The widths of the cells in :math:`x` and :math:`y` are given by :math:`\Delta x_i` and :math:`\Delta y_j`, respectively.

.. image:: ../../images/averaging_2d.png
    :align: center
    :width: 300

Let :math:`u(x,y)` be a scalar function whose values are known at the nodes.
Where :math:`\bar{u}_{i,j}` is the average at the center of cell :math:`i,j`:

.. math::
	\bar{u}_{i,j} = \frac{1}{4} \Big ( u_{i-\frac{1}{2},j-\frac{1}{2}} + u_{i+\frac{1}{2},j-\frac{1}{2}} + u_{i-\frac{1}{2},j+\frac{1}{2}} + u_{i+\frac{1}{2}, j+\frac{1}{2}} \Big )

Our goal is to construct a matrix :math:`\boldsymbol{A}` that averages the values
at the nodes and places them at the cell centers:

.. math::
	\bar{\boldsymbol{u}} = \boldsymbol{A \, u}

where :math:`\boldsymbol{u}` is a vector that stores the known values of :math:`u(x,y)` at the nodes, 
and :math:`\bar{\boldsymbol{u}}` is a vector that stores the averages at cell centers.

For tensor meshes, the averaging matrix is rather easy to construct. Where the number of nodes in the :math:`x` and :math:`y`
directions are known, we use expression :eq:`operators_averaging_n2c_1d` to construct 1D averaging matrices
:math:`\boldsymbol{A_x}` and :math:`\boldsymbol{A_y}`. The averaging matrix in 2D is 

.. math::
	\boldsymbol{A} = \boldsymbol{A_y} \otimes \boldsymbol{A_x}

And in 3D:

.. math::
	\boldsymbol{A} = \boldsymbol{A_z} \otimes (\boldsymbol{A_y} \otimes \boldsymbol{A_x})

where :math:`\otimes` is the kronecker product.


.. important:: For mesh with more complicated structures (e.g. tree mesh), the general theory is the same. However, the averaging matrix cannot be constructed directly using kronecker products.


Scalars from Cell Centers to Nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Kronecker products can also be used to construct the averaging matrix from cell centers to nodes.
In this case, expression :eq:`operators_averaging_c2n_1d` is used to construct the 1D averaging
matricies in the :math:`x`, :math:`y` (and :math:`z`) directions using the dimensions of the cells
along each axis. Once again, nearest neighbour is used to assign a value to cell centers which lie outside the mesh.

For 2D averaging from cell centers to nodes:

.. math::
	\bar{\boldsymbol{A}} = \bar{\boldsymbol{A_y}} \otimes \bar{\boldsymbol{A_x}}

And for 3D averaging from cell centers to nodes:

.. math::
	\bar{\boldsymbol{A}} = \bar{\boldsymbol{A_z}} \otimes (\bar{\boldsymbol{A_y}} \otimes \bar{\boldsymbol{A_x}})



Vectors from Faces to Cell Centers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us define a 2D mesh where cell center locations are defined by :math:`(x_i,y_j)`.
The widths of the cells in :math:`x` and :math:`y` are given by :math:`\Delta x_i` and :math:`\Delta y_j`, respectively.

.. image:: ../../images/averaging_2d_faces.png
    :align: center
    :width: 400




Vectors from Edges to Cell Centers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us define a 2D mesh where node locations are defined by :math:`(x_i,y_j)`.
The widths of the cells in :math:`x` and :math:`y` are given by :math:`\Delta x_i` and :math:`\Delta y_j`, respectively.

.. image:: ../../images/averaging_2d_edges.png
    :align: center
    :width: 400




