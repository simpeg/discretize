.. _operators_averaging:

Averaging
*********

Here, we provide the background theory for how discrete averaging matricies are formed.
Averaging matrices are required when quantities that live on different
parts of the mesh need to be added, subtracted, multiplied or divided.
Averaging matrices are built using the same principles that were discussed when forming :ref:`interpolation matrices <operators_interpolation>`;
except the locations of the original quantity and the interpolated quantity are organized on a structured grid.

Where :math:`\boldsymbol{u}` is a discrete representation of a vector living somewhere on the mesh (nodes, edges, faces),
and :math:`\bar{\boldsymbol{u}}` is the vector containing the averages mapped to another part of the mesh,
we look to constructs an averaging matrix :math:`\boldsymbol{A}` such that:

.. math::
	\bar{\boldsymbol{u}} = A \, \boldsymbol{u}
	:label: operators_averaging_general

**Tutorial:** :ref:`tutorial for constructing and applying averaging operators <sphx_glr_tutorials_operators_1_averaging.py>`


Averaging Matrices in 1D
========================

Nodes to Cell Centers
^^^^^^^^^^^^^^^^^^^^^

Let us define a 1D mesh where cell center and node locations are defined according to the figure below.
The widths of the cells are given by :math:`\Delta x_i`.

.. figure:: ../../images/averaging_1d.png
    :align: center
    :width: 600

    A 1D tensor mesh denoting the node and cell center locations.

If :math:`u(x)` is a scalar function whose values are known at the nodes
and :math:`\bar{u}_i` is the average value at the center of cell :math:`i`,
then:

.. math::
	\bar{u}_i = \frac{u_{i-\tfrac{1}{2}} + u_{i+\tfrac{1}{2}}}{2}

Our goal is to construct a matrix :math:`A` that averages the values
at the nodes and places them at the cell centers, i.e.:

.. math::
	\bar{\boldsymbol{u}} = A \, \boldsymbol{u}

where :math:`\boldsymbol{u}` is a vector that stores the known values of :math:`u(x)` at the nodes, 
and :math:`\bar{\boldsymbol{u}}` is a vector that stores the averages at cell centers.

For the entire mesh, the averaging matrix is given by:

.. math::
	A = \frac{1}{2} \begin{bmatrix}
	1 & 1 & 0 & 0 & \cdots & 0 & 0 \\
	0 & 1 & 1 & 0 & \cdots & 0 & 0 \\
	0 & 0 & 1 &1 & \cdots & 0 & 0 \\
	\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
	0 & 0 & 0 & 0 & \cdots & 1 & 1
	\end{bmatrix}
	:label: operators_averaging_n2c_1d

where :math:`A` is a sparse matrix. Defining :math:`nc` as the number of cells in the mesh,
:math:`A` is has shape :math:`nc` by :math:`nc \! + \! 1`.

Cell Centers to Nodes
^^^^^^^^^^^^^^^^^^^^^

Let us re-examine the figure illustrating the 1D mesh and assume we know the value of the function :math:`u(x)` at cell centers.
Note that the nodes are not equal distance from the cell centers on either side.
Therefore we cannot simply sum the values and divide by half. Instead we need to implement a weighted averaging.

If :math:`u(x)` is a scalar function whose values are known at the cell centers
and :math:`\bar{u}_i` is the average value at the nodes,
then:

.. math::
	\bar{u}_{i+\frac{1}{2}} = \Bigg ( \frac{\Delta x_{i+1}}{\Delta x_i + \Delta x_{i+1}} \Bigg ) u_{i}
	+ \Bigg ( \frac{\Delta x_i}{\Delta x_i + \Delta x_{i+1}} \Bigg ) u_{i+1}

Our goal is to construct a matrix :math:`\bar{A}` that averages the values
at the cell centers and places them at the nodes, i.e.:

.. math::
	\bar{\boldsymbol{u}} = \bar{\boldsymbol{A}} \, \boldsymbol{u}

where :math:`\boldsymbol{u}` is a vector that stores the known values of :math:`u(x)` at cell centers, 
and :math:`\bar{\boldsymbol{u}}` is a vector that stores the averages at the nodes.

For the entire mesh, the averaging matrix is given by:

.. math::
	\bar{A} = \frac{1}{2} \begin{bmatrix}
	1 & 0 & 0 & 0 & \cdots & 0 & 0 \\
	a_1 & b_1 & 0 & 0 & \cdots & 0 & 0 \\
	0 & a_2 & b_2 & 0 & \cdots & 0 & 0 \\
	\vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots \\
	0 & 0 & 0 & 0 & \cdots & a_{nc-1} & b_{nc-1} \\
	0 & 0 & 0 & 0 & \cdots & 0 & 1
	\end{bmatrix} \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;
	\begin{split}
	a_i &= \frac{\Delta x_{i+1}}{\Delta x_i + \Delta x_{i+1}} \\
	& \\
	b_i &= \frac{\Delta x_i}{\Delta x_i + \Delta x_{i+1}}
	\end{split}
	:label: operators_averaging_c2n_1d

where :math:`\bar{A}` is a sparse matrix. Defining :math:`nc` as the number of cells in the mesh,
:math:`\bar{A}` has shape :math:`nc \! + \! 1` by :math:`nc`. Note that :math:`\bar{A}_{0,0}` and :math:`\bar{A}_{nc,nc-1}`
are equal to 1. This is because cell center locations needed to compute the average lie outside the mesh
and we must extrapolate using the nearest neighbour.


Averaging Scalars in 2D and 3D
==============================

Nodes to Cell Centers
^^^^^^^^^^^^^^^^^^^^^

For a 2D mesh, the value of the function at 4 locations is needed to average from nodes to cell centers.
Let us define a 2D mesh where cell center locations :math:`(x_i, y_j)` are represented using indices :math:`(i,j)`.
The widths of the cells in :math:`x` and :math:`y` are given by :math:`\Delta x_i` and :math:`\Delta y_j`, respectively.

.. figure:: ../../images/averaging_2d.png
    :align: center
    :width: 300

    A 2D tensor mesh which shows the indexing for node and cell center locations.

If :math:`u(x,y)` is a scalar function whose values are known at the nodes
and :math:`\bar{u} (i,j)` is the average at the center of cell :math:`i,j`,
then:

.. math::
	\bar{u}(i,j) = \frac{1}{4} \Big [
	u \Big ( i-\tfrac{1}{2}, j-\tfrac{1}{2} \Big ) +
	u \Big ( i+\tfrac{1}{2}, j-\tfrac{1}{2} \Big ) +
	u \Big ( i-\tfrac{1}{2}, j+\tfrac{1}{2} \Big ) +
	u \Big ( i+\tfrac{1}{2}, j+\tfrac{1}{2} \Big ) \Big ]

Our goal is to construct a matrix :math:`\boldsymbol{A}` that averages the values
at the nodes and places them at the cell centers, i.e.:

.. math::
	\bar{\boldsymbol{u}} = \boldsymbol{A \, u}

where :math:`\boldsymbol{u}` is a vector that stores the known values of :math:`u(x,y)` at the nodes, 
and :math:`\bar{\boldsymbol{u}}` is a vector that stores the averages at cell centers.

For tensor meshes, the averaging matrix is rather easy to construct.
Using equation :eq:`operators_averaging_n2c_1d`, the number of cells in the x-direction
can be used to construct a matrix :math:`A_x`. And the number of cells in the y-direction
can be used to construct a matrix :math:`A_y`. The averaging matrix in 2D is given by:

.. math::
	\boldsymbol{A} = A_y \otimes A_x

where :math:`\otimes` is the `Kronecker product <https://en.wikipedia.org/wiki/Kronecker_product>`__. For a 3D tensor mesh, the averaging matrix
from nodes to cell centers would be given by:

.. math::
	\boldsymbol{A} = A_z \otimes (A_y \otimes A_x)


Cell Centers to Nodes
^^^^^^^^^^^^^^^^^^^^^

A nearly identical approach can be implemented to average from cell centers to nodes.
In this case, expression :eq:`operators_averaging_c2n_1d` is used to construct the 1D averaging
matricies in the :math:`x`, :math:`y` (and :math:`z`) directions using the dimensions of the cells
along each axis. Once again, nearest neighbour is used to assign a value to cell centers which lie outside the mesh.

Our goal is to construct a matrix :math:`\boldsymbol{A}` that averages the values
at the cell centers and places them at the nodes, i.e.:

.. math::
	\bar{\boldsymbol{u}} = \boldsymbol{A \, u}

where :math:`\boldsymbol{u}` is a vector that stores the known values of :math:`u(x,y)` at the cell centers, 
and :math:`\bar{\boldsymbol{u}}` is a vector that stores the averages at nodes.

For 2D averaging from cell centers to nodes, we use equation :eq:`operators_averaging_c2n_1d` and the cell widths
in the x and y directions to construct 1D averaging matrices :math:`\bar{A}_x` and :math:`\bar{A}_y`, respectively.
The averaging operator for a 2D tensor mesh is given by:

.. math::
	\boldsymbol{A} = \bar{A}_y \otimes \bar{A}_x

And for 3D averaging from cell centers to nodes:

.. math::
	\boldsymbol{A} = \bar{A}_z \otimes (\bar{A}_y \otimes \bar{A}_x)


Faces to Cell Centers
^^^^^^^^^^^^^^^^^^^^^

Let us define a 2D mesh where cell center locations :math:`(x_i, y_j)` are represented using indices :math:`(i,j)`.
The widths of the cells in :math:`x` and :math:`y` are given by :math:`\Delta x_i` and :math:`\Delta y_j`, respectively.

.. figure:: ../../images/averaging_2d_faces.png
    :align: center
    :width: 350

    A 2D tensor mesh which shows the indexing for face and cell center locations.

If :math:`u(x,y)` is a scalar quantity whose values are known on the faces.
and :math:`\bar{u}(i,j)` is the average at the center of cell :math:`i,j`,
then:

.. math::
	\bar{u}(i,j) = \frac{1}{4} \Big [
	u \Big ( i-\tfrac{1}{2}, j \Big ) + 
	u \Big ( i+\tfrac{1}{2}, j \Big ) +
	u \Big ( i, j-\tfrac{1}{2} \Big ) +
	u \Big ( i, j+\tfrac{1}{2} \Big ) \Big ]

Our goal is to construct a matrix :math:`\boldsymbol{A}` that averages the values
on the faces and places them at the cell centers, i.e.:

.. math::
	\bar{\boldsymbol{u}} = \boldsymbol{A \, u}

where :math:`\boldsymbol{u}` is a vector that stores the known values of :math:`u(x,y)` on the x and y-faces, 
and :math:`\bar{\boldsymbol{u}}` is a vector that stores the averages at cell centers.

Let :math:`I_n` be an :math:`n` by :math:`n` identity matrix. 
And use equation :eq:`operators_averaging_n2c_1d` to construct 1D averaging matrices :math:`A_x` and :math:`A_y`.
Then for a 2D tensor mesh, the averaging matrix has the form:

.. math::
	\boldsymbol{A} = \frac{1}{2} \begin{bmatrix} \boldsymbol{A_x} & \boldsymbol{A_y} \end{bmatrix}

where

.. math::
	\begin{align}
	\boldsymbol{A_x} &= I_{ny} \otimes A_x \\
	\boldsymbol{A_y} &= A_y \otimes I_{nx}
	\end{align}
	:label: operators_averaging_Ai_f2c_2d

For a 3D tensor mesh, the averaging matrix takes the form:

.. math::
	\boldsymbol{A} = \frac{1}{3} \begin{bmatrix} \boldsymbol{A_x} & \boldsymbol{A_y} & \boldsymbol{A_z} \end{bmatrix}

where

.. math::
	\begin{align}
	\boldsymbol{A_x} &= I_{nz} \otimes ( I_{ny} \otimes A_x ) \\
	\boldsymbol{A_y} &= I_{nz} \otimes ( A_y \otimes I_{nx} ) \\
	\boldsymbol{A_z} &= A_z \otimes ( I_{ny} \otimes I_{nx} )
	\end{align}
	:label: operators_averaging_Ai_f2c_3d


Edges to Cell Centers
^^^^^^^^^^^^^^^^^^^^^

Let us define a 2D mesh where cell center locations :math:`(x_i, y_j)` are represented using indices :math:`(i,j)`.
The widths of the cells in :math:`x` and :math:`y` are given by :math:`\Delta x_i` and :math:`\Delta y_j`, respectively.

.. figure:: ../../images/averaging_2d_edges.png
    :align: center
    :width: 350

    A 2D tensor mesh which shows the indexing for edge and cell center locations.

If :math:`u(x,y)` is a scalar quantity whose values are known on the edges.
and :math:`\bar{u}(i,j)` is the average at the center of cell :math:`i,j`,
then:

.. math::
	\bar{u}(i,j) = \frac{1}{4} \Big [
	u \Big ( i, j-\tfrac{1}{2} \Big ) + 
	u \Big ( i, j+\tfrac{1}{2} \Big ) +
	u \Big ( i-\tfrac{1}{2}, j \Big ) +
	u \Big ( i+\tfrac{1}{2}, j \Big ) \Big ]

Our goal is to construct a matrix :math:`\boldsymbol{A}` that averages the values
on the edges and places them at the cell centers, i.e.:

.. math::
	\bar{\boldsymbol{u}} = \boldsymbol{A \, u}

where :math:`\boldsymbol{u}` is a vector that stores the known values of :math:`u(x,y)` on the x and y-edges, 
and :math:`\bar{\boldsymbol{u}}` is a vector that stores the averages at cell centers.

Let :math:`I_n` be an :math:`n` by :math:`n` identity matrix. 
And use equation :eq:`operators_averaging_n2c_1d` to construct 1D averaging matrices :math:`A_x` and :math:`A_y`.
Then for a 2D tensor mesh, the averaging matrix has the form:

.. math::
	\boldsymbol{A} = \frac{1}{2} \begin{bmatrix} \boldsymbol{A_x} & \boldsymbol{A_y} \end{bmatrix}

where

.. math::
	\begin{align}
	\boldsymbol{A_x} &= A_x \otimes I_{ny} \\
	\boldsymbol{A_y} &= I_{nx} \otimes A_y
	\end{align}
	:label: operators_averaging_Ai_e2c_2d

For a 3D tensor mesh, the averaging matrix takes the form:

.. math::
	\boldsymbol{A} = \frac{1}{3} \begin{bmatrix} \boldsymbol{A_x} & \boldsymbol{A_y} & \boldsymbol{A_z} \end{bmatrix}

where

.. math::
	\begin{align}
	\boldsymbol{A_x} &= I_{nz} \otimes ( A_y \otimes A_x ) \\
	\boldsymbol{A_y} &= A_z \otimes ( I_{ny} \otimes A_x ) \\
	\boldsymbol{A_z} &= A_z \otimes ( A_y \otimes I_{nx} )
	\end{align}
	:label: operators_averaging_Ai_e2c_3d


Averaging Vectors in 2D and 3D
==============================

Faces to Cell Centers
^^^^^^^^^^^^^^^^^^^^^

Let :math:`\vec{u}(x,y)` be a vector function that is known on the faces.
That is, :math:`u_x (x,y)` lives on x-faces and :math:`u_y(x,y)` lives on y-faces.
In this case, the x-faces are used to average the x-component
to cell centers and the y-faces are used to average the y-component to cell centers separately.

Let us define a 2D mesh where cell center locations :math:`(x_i, y_j)` are represented using indices :math:`(i,j)`.
The widths of the cells in :math:`x` and :math:`y` are given by :math:`\Delta x_i` and :math:`\Delta y_j`, respectively.

.. figure:: ../../images/averaging_2d_faces.png
    :align: center
    :width: 350

    A 2D tensor mesh which shows the indexing for face and cell center locations.

Where :math:`\bar{u}_x (i,j)` is the average x-component at the center of cell :math:`i,j`:

.. math::
	\bar{u}_x (i,j) = \frac{1}{2} \Big [ u_x \Big ( i-\tfrac{1}{2},j \Big ) + u_x \Big ( i+\tfrac{1}{2},j \Big ) \Big ] 

And where :math:`\bar{u}_y (i,j)` is the average at the center of cell :math:`i,j`:

.. math::
	\bar{u}_y (i,j) = \frac{1}{2} \Big [ u_y \Big ( i,j-\tfrac{1}{2} \Big ) + u_y \Big ( i,j+\tfrac{1}{2} \Big ) \Big ] 

Our goal is to construct a matrix :math:`\boldsymbol{A}` that averages the vector components living
on the faces separately and places them at the cell centers.
Here, the operation :math:`\bar{\boldsymbol{u}} = \boldsymbol{A \, u}` takes the form:

.. math::
	\begin{bmatrix} \bar{\boldsymbol{u}}_{\boldsymbol{x}} \\ \bar{\boldsymbol{u}}_{\boldsymbol{y}} \end{bmatrix} =
	\begin{bmatrix} \boldsymbol{A_x} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{A_y} \end{bmatrix}
	\begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \end{bmatrix}

where :math:`\boldsymbol{u}` is a vector that stores the known values of :math:`u_x(x,y)` and :math:`u_y(x,y)` on their respective faces, 
and :math:`\bar{\boldsymbol{u}}` is a vector that stores the x component and y component averages at cell centers.
Matrices :math:`\boldsymbol{A_x}` and :math:`\boldsymbol{A_y}` are defined in expression :eq:`operators_averaging_Ai_f2c_2d`.

In 3D, the corresponding averaging matrix is defined by:

.. math::
	\begin{bmatrix} \bar{\boldsymbol{u}}_{\boldsymbol{x}} \\ \bar{\boldsymbol{u}}_{\boldsymbol{y}} \\ \bar{\boldsymbol{u}}_{\boldsymbol{z}} \end{bmatrix} =
	\begin{bmatrix} \boldsymbol{A_x} & \boldsymbol{0} & \boldsymbol{0} \\
	\boldsymbol{0} & \boldsymbol{A_y} & \boldsymbol{0} \\
	\boldsymbol{0} & \boldsymbol{0} & \boldsymbol{A_z} \end{bmatrix}
	\begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_z} \end{bmatrix}

where matrices :math:`\boldsymbol{A_x}`, :math:`\boldsymbol{A_y}` and :math:`\boldsymbol{A_z}` are defined in expression :eq:`operators_averaging_Ai_f2c_3d`.


Edges to Cell Centers
^^^^^^^^^^^^^^^^^^^^^

Let :math:`\vec{u}(x,y)` be a vector function that is known on the edges.
That is, :math:`u_x (x,y)` lives on x-edges and :math:`u_y(x,y)` lives on y-edges.
In this case, the x-edges are used to average the x-component
to cell centers and the y-edges are used to average the y-component to cell centers separately.

Let us define a 2D mesh where cell center locations :math:`(x_i, y_j)` are represented using indices :math:`(i,j)`.
The widths of the cells in :math:`x` and :math:`y` are given by :math:`\Delta x_i` and :math:`\Delta y_j`, respectively.

.. figure:: ../../images/averaging_2d_edges.png
    :align: center
    :width: 350

    A 2D tensor mesh which shows the indexing for edge and cell center locations.

Where :math:`\bar{u}_x (i,j)` is the average x-component at the center of cell :math:`i,j`:

.. math::
	\bar{u}_x (i,j) = \frac{1}{2} \Big [ u_x \Big ( i,j-\tfrac{1}{2} \Big ) + u_x \Big ( i,j+\tfrac{1}{2} \Big ) \Big ] 

And where :math:`\bar{u}_y (i,j)` is the average at the center of cell :math:`i,j`:

.. math::
	\bar{u}_y (i,j) = \frac{1}{2} \Big [ u_y \Big ( i-\tfrac{1}{2},j \Big ) + u_y \Big ( i+\tfrac{1}{2},j \Big ) \Big ] 

Our goal is to construct a matrix :math:`\boldsymbol{A}` that averages the vector components living
on the edges separately and places them at the cell centers.
Here, the operation :math:`\bar{\boldsymbol{u}} = \boldsymbol{A \, u}` takes the form:

.. math::
	\begin{bmatrix} \bar{\boldsymbol{u}}_{\boldsymbol{x}} \\ \bar{\boldsymbol{u}}_{\boldsymbol{y}} \end{bmatrix} =
	\begin{bmatrix} \boldsymbol{A_x} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{A_y} \end{bmatrix}
	\begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \end{bmatrix}

where :math:`\boldsymbol{u}` is a vector that stores the known values of :math:`u_x(x,y)` and :math:`u_y(x,y)` on their respective edges, 
and :math:`\bar{\boldsymbol{u}}` is a vector that stores the x component and y component averages at cell centers.
Matrices :math:`\boldsymbol{A_x}` and :math:`\boldsymbol{A_y}` are defined in expression :eq:`operators_averaging_Ai_e2c_2d`.

In 3D, the corresponding averaging matrix is defined by:

.. math::
	\begin{bmatrix} \bar{\boldsymbol{u}}_{\boldsymbol{x}} \\ \bar{\boldsymbol{u}}_{\boldsymbol{y}} \\ \bar{\boldsymbol{u}}_{\boldsymbol{z}} \end{bmatrix} =
	\begin{bmatrix} \boldsymbol{A_x} & \boldsymbol{0} & \boldsymbol{0} \\
	\boldsymbol{0} & \boldsymbol{A_y} & \boldsymbol{0} \\
	\boldsymbol{0} & \boldsymbol{0} & \boldsymbol{A_z} \end{bmatrix}
	\begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_z} \end{bmatrix}

where matrices :math:`\boldsymbol{A_x}`, :math:`\boldsymbol{A_y}` and :math:`\boldsymbol{A_z}` are defined in expression :eq:`operators_averaging_Ai_e2c_3d`.




