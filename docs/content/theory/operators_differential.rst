.. _operators_differential:

Differential Operators
**********************

Here, we provide the background theory for how discrete differential operators are formed.
We follow the approach discussed in :cite:`haber2014,HymanShashkov1999`.
For geophysical problems, the relationship between two physical quantities may include one of several differential operators:

    - **Divergence:** :math:`\nabla \cdot \vec{u} = \dfrac{\partial u_x}{\partial x} + \dfrac{\partial u_y}{\partial y} + \dfrac{\partial u_y}{\partial y}`
    - **Gradient:** :math:`\nabla \phi = \dfrac{\partial \phi}{\partial x}\hat{x} + \dfrac{\partial \phi}{\partial y}\hat{y} + \dfrac{\partial \phi}{\partial z}\hat{z}`
    - **Curl:** :math:`\nabla \times \vec{u} = \Bigg ( \dfrac{\partial u_y}{\partial z} - \dfrac{\partial u_z}{\partial y} \Bigg )\hat{x} - \Bigg ( \dfrac{\partial u_x}{\partial z} - \dfrac{\partial u_z}{\partial x} \Bigg )\hat{y} + \Bigg ( \dfrac{\partial u_x}{\partial y} - \dfrac{\partial u_y}{\partial x} \Bigg )\hat{z}`

When implementing the finite volume method, continuous variables are discretized to live at the cell centers, nodes, edges or faces of a mesh.
Thus for each differential operator, we need a discrete approximation that acts on the discrete variables living on the mesh.

Our approximations are derived using numerical differentiation (figure below). So long as a function :math:`f(x)` is sufficiently smooth
within the interval :math:`[x-h/2, \; x+h/2]`, then the derivative of the function at :math:`x` is approximated by:

.. math::
    \frac{df(x)}{dx} \approx \frac{f(x+h/2) \; - \; f(x-h/2)}{h}

where the approximation becomes increasingly accurate as :math:`h \rightarrow 0`. In subsequent sections, we will show how
the gradient, divergence and curl can be computed for discrete variables.

.. figure:: ../../images/approximate_derivative.png
    :align: center
    :width: 300

    Approximating the derivative of :math:`f(x)` using numerical differentiation.


**Tutorial:** :ref:`tutorial for constructing and applying differential operators <sphx_glr_tutorials_operators_2_differential.py>`


.. _operators_differential_divergence:

Divergence
----------

Let us define a continuous scalar function :math:`\phi` and a continuous vector function :math:`\vec{u}` such that:

.. math::
    \phi = \nabla \cdot \vec{u}

And let :math:`\boldsymbol{\phi}` and :math:`\boldsymbol{u}` be the discrete representations of :math:`\phi` and :math:`\vec{u}`
that live on the mesh (centers, nodes, edges or faces), respectively. Provided we know the discrete values :math:`\boldsymbol{u}`,
our goal is to use discrete differentiation to approximate the values of :math:`\boldsymbol{\phi}`.
We begin by considering a single cell (2D or 3D). We let the indices :math:`i`, :math:`j` and :math:`k` 
denote positions along the x, y and z axes, respectively.

.. figure:: ../../images/divergence_discretization.png
    :align: center
    :width: 600

    Discretization for approximating the divergence at the center of a single 2D cell (left) and 3D cell (right).

+-------------+-------------------------------------------------+-----------------------------------------------------+
|             |                    **2D**                       |                       **3D**                        |
+-------------+-------------------------------------------------+-----------------------------------------------------+
| **center**  | :math:`(i,j)`                                   | :math:`(i,j,k)`                                     |
+-------------+-------------------------------------------------+-----------------------------------------------------+
| **x-faces** | :math:`(i-\frac{1}{2},j)\;\; (i+\frac{1}{2},j)` | :math:`(i-\frac{1}{2},j,k)\;\; (i+\frac{1}{2},j,k)` |
+-------------+-------------------------------------------------+-----------------------------------------------------+
| **y-faces** | :math:`(i,j-\frac{1}{2})\;\; (i,j+\frac{1}{2})` | :math:`(i,j-\frac{1}{2},k)\;\; (i,j+\frac{1}{2},k)` |
+-------------+-------------------------------------------------+-----------------------------------------------------+
| **z-faces** | N/A                                             | :math:`(i,j,k-\frac{1}{2})\;\; (i,j,k+\frac{1}{2})` |
+-------------+-------------------------------------------------+-----------------------------------------------------+

As we will see, it makes the most sense for :math:`\boldsymbol{\phi}` to live at the cell centers and
for the components of :math:`\boldsymbol{u}` to live on the faces. If :math:`u_x` lives on x-faces, then its discrete
derivative with respect to :math:`x` lives at the cell center. And if :math:`u_y` lives on y-faces its discrete
derivative with respect to :math:`y` lives at the cell center. Likewise for :math:`u_z`. Thus to approximate the
divergence of :math:`\vec{u}` at the cell center, we simply need to sum the discrete derivatives of :math:`u_x`, :math:`u_y`
and :math:`u_z` that are defined at the cell center. Where :math:`h_x`, :math:`h_y` and :math:`h_z` represent the dimension of the cell along the x, y and
z directions, respectively:

.. math::
    \begin{align}
    \mathbf{In \; 2D:} \;\; \phi(i,j) \approx \; & \frac{u_x(i,j+\frac{1}{2}) - u_x(i,j-\frac{1}{2})}{h_x} \\
    & + \frac{u_y(i+\frac{1}{2},j) - u_y(i-\frac{1}{2},j)}{h_y}
    \end{align}

|

.. math::
    \begin{align}
    \mathbf{In \; 3D:} \;\; \phi(i,j,k) \approx \; & \frac{u_x(i+\frac{1}{2},j,k) - u_x(i-\frac{1}{2},j,k)}{h_x} \\
    & + \frac{u_y(i,j+\frac{1}{2},k) - u_y(i,j-\frac{1}{2},k)}{h_y} \\
    & + \frac{u_z(i,j,k+\frac{1}{2}) - u_z(i,j,k-\frac{1}{2})}{h_z}
    \end{align}


Ultimately we are trying to approximate the divergence at the center of every cell in a mesh.
Adjacent cells share faces. If the components :math:`u_x`, :math:`u_y` and :math:`u_z` are
continuous across their respective faces, then :math:`\boldsymbol{\phi}` and :math:`\boldsymbol{u}`
can be related by a sparse matrix-vector product:

.. math::
    \boldsymbol{\phi} = \boldsymbol{D \, u}

where :math:`\boldsymbol{D}` is the divergence matrix from faces to cell centers,
:math:`\boldsymbol{\phi}` is a vector containing the discrete approximations of :math:`\phi` at all cell centers,
and :math:`\boldsymbol{u}` stores the components of :math:`\vec{u}` on cell faces as a vector of the form:

.. math::
    \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_z} \end{bmatrix}

.. _operators_differential_gradient:

Gradient
--------

Let us define a continuous scalar function :math:`\phi` and a continuous vector function :math:`\vec{u}` such that:

.. math::
    \vec{u} = \nabla \phi

And let :math:`\boldsymbol{\phi}` and :math:`\boldsymbol{u}` be the discrete representations of :math:`\phi` and :math:`\vec{u}`
that live on the mesh (centers, nodes, edges or faces), respectively. Provided we know the discrete values :math:`\boldsymbol{\phi}`,
our goal is to use discrete differentiation to approximate the vector components of :math:`\boldsymbol{u}`.
We begin by considering a single cell (2D or 3D). We let the indices :math:`i`, :math:`j` and :math:`k` 
denote positions along the x, y and z axes, respectively.

.. figure:: ../../images/gradient_discretization.png
    :align: center
    :width: 600

    Discretization for approximating the gradient on the edges of a single 2D cell (left) and 3D cell (right).

As we will see, it makes the most sense for :math:`\boldsymbol{\phi}` to live at the cell nodes and
for the components of :math:`\boldsymbol{u}` to live on corresponding edges. If :math:`\phi` lives on the nodes, then:

    - the partial derivative :math:`\dfrac{\partial \phi}{\partial x}\hat{x}` lives on x-edges,
    - the partial derivative :math:`\dfrac{\partial \phi}{\partial y}\hat{y}` lives on y-edges, and
    - the partial derivative :math:`\dfrac{\partial \phi}{\partial z}\hat{z}` lives on z-edges

Thus to approximate the gradient of :math:`\phi`, 
we simply need to take discrete derivatives of :math:`\phi` with respect to :math:`x`, :math:`y` and :math:`z`,
and organize the resulting vector components on the corresponding edges.
Let :math:`h_x`, :math:`h_y` and :math:`h_z` represent the dimension of the cell along the x, y and
z directions, respectively.

**In 2D**, the value of :math:`\phi` at 4 node locations is used to approximate the vector components of the
gradient at 4 edges locations (2 x-edges and 2 y-edges) as follows:

.. math::
    \begin{align}
    u_x \Big ( i+\frac{1}{2},j \Big ) \approx \; & \frac{\phi (i+1,j) - \phi (i,j)}{h_x} \\
    u_x \Big ( i+\frac{1}{2},j+1 \Big ) \approx \; & \frac{\phi (i+1,j+1) - \phi (i,j+1)}{h_x} \\
    u_y \Big ( i,j+\frac{1}{2} \Big ) \approx \; & \frac{\phi (i,j+1) - \phi (i,j)}{h_y} \\
    u_y \Big ( i+1,j+\frac{1}{2} \Big ) \approx \; & \frac{\phi (i+1,j+1) - \phi (i+1,j)}{h_y}
    \end{align}

**In 3D**, the value of :math:`\phi` at 8 node locations is used to approximate the vector components of the
gradient at 12 edges locations (4 x-edges, 4 y-edges and 4 z-edges). An example of the approximation
for each vector component is given below:

.. math::
    \begin{align}
    u_x \Big ( i+\frac{1}{2},j,k \Big ) \approx \; & \frac{\phi (i+1,j,k) - \phi (i,j,k)}{h_x} \\
    u_y \Big ( i,j+\frac{1}{2},k \Big ) \approx \; & \frac{\phi (i,j+1,k) - \phi (i,j,k)}{h_y} \\
    u_z \Big ( i,j,k+\frac{1}{2} \Big ) \approx \; & \frac{\phi (i,j,k+1) - \phi (i,j,k)}{h_z}
    \end{align}


Ultimately we are trying to approximate the vector components of the gradient at all edges of a mesh.
Adjacent cells share nodes. If :math:`\phi` is continuous at the nodes, then :math:`\boldsymbol{\phi}` and :math:`\boldsymbol{u}`
can be related by a sparse matrix-vector product:

.. math::
    \boldsymbol{u} = \boldsymbol{G \, \phi}

where :math:`\boldsymbol{G}` is the gradient matrix that maps from nodes to edges,
:math:`\boldsymbol{\phi}` is a vector containing :math:`\phi` at all nodes,
and :math:`\boldsymbol{u}` stores the components of :math:`\vec{u}` on cell edges as a vector of the form:

.. math::
    \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_z} \end{bmatrix}

.. _operators_differential_curl:

Curl
----

Let us define two continuous vector functions :math:`\vec{u}` and :math:`\vec{w}` such that:

.. math::
    \vec{w} = \nabla \times \vec{u}

And let :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` be the discrete representations of :math:`\vec{u}` and :math:`\vec{w}`
that live on the mesh (centers, nodes, edges or faces), respectively. Provided we know the discrete values :math:`\boldsymbol{u}`,
our goal is to use discrete differentiation to approximate the vector components of :math:`\boldsymbol{w}`.
We begin by considering a single 3D cell. We let the indices :math:`i`, :math:`j` and :math:`k` 
denote positions along the x, y and z axes, respectively.

.. figure:: ../../images/curl_discretization.png
    :align: center
    :width: 800

    Discretization for approximating the x, y and z components of the curl on the respective faces of a 3D cell.


As we will see, it makes the most sense for the vector components of :math:`\boldsymbol{u}` to live on the edges
for the vector components of :math:`\boldsymbol{w}` to live the faces. In this case, we need to approximate:


    - the partial derivatives :math:`\dfrac{\partial u_y}{\partial z}` and :math:`\dfrac{\partial u_z}{\partial y}` to compute :math:`w_x`,
    - the partial derivatives :math:`\dfrac{\partial u_x}{\partial z}` and :math:`\dfrac{\partial u_z}{\partial x}` to compute :math:`w_y`, and
    - the partial derivatives :math:`\dfrac{\partial u_x}{\partial y}` and :math:`\dfrac{\partial u_y}{\partial x}` to compute :math:`w_z`

**In 3D**, discrete values at 12 edge locations (4 x-edges, 4 y-edges and 4 z-edges) are used to
approximate the vector components of the curl at 6 face locations (2 x-faces, 2-faces and 2 z-faces).
An example of the approximation for each vector component is given below:

.. math::
    \begin{align}
    w_x \Big ( i,j \! +\!\!\frac{1}{2},k \! +\!\!\frac{1}{2} \Big ) \!\approx\! \; &
    \!\Bigg ( \! \frac{u_z (i,j \! +\!\!1,k \! +\!\!\frac{1}{2})  \! -\! u_z (i,j,k \! +\!\!\frac{1}{2})}{h_y} \Bigg) \!
    \! -\! \!\Bigg ( \! \frac{u_y (i,j \! +\!\!\frac{1}{2},k \! +\!\!1)  \! -\! u_y (i,j \! +\!\!\frac{1}{2},k)}{h_z} \Bigg) \! \\
    & \\
    w_y \Big ( i \! +\!\!\frac{1}{2},j,k \! +\!\!\frac{1}{2} \Big ) \!\approx\! \; &
    \!\Bigg ( \! \frac{u_x (i \! +\!\!\frac{1}{2},j,k \! +\!\!1)  \! -\! u_x (i \! +\!\!\frac{1}{2},j,k)}{h_z} \Bigg)
    \! -\! \!\Bigg ( \! \frac{u_z (i \! +\!\!1,j,k \! +\!\!\frac{1}{2})  \! -\! u_z (i,j,k \! +\!\!\frac{1}{2})}{h_x} \Bigg) \! \\
    & \\
    w_z \Big ( i \! +\!\!\frac{1}{2},j \! +\!\!\frac{1}{2},k \Big ) \!\approx\! \; &
    \!\Bigg ( \! \frac{u_y (i \! +\!\!1,j \! +\!\!\frac{1}{2},k)  \! -\! u_y (i,j \! +\!\!\frac{1}{2},k)}{h_x} \Bigg )
    \! -\! \!\Bigg ( \! \frac{u_x (i \! +\!\!\frac{1}{2},j \! +\!\!1,k)  \! -\! u_x (i \! +\!\!\frac{1}{2},j,k)}{h_y} \Bigg) \!
    \end{align}


Ultimately we are trying to approximate the curl on all the faces within a mesh.
Adjacent cells share edges. If the components :math:`u_x`, :math:`u_y` and :math:`u_z` are
continuous across at the edges, then :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}`
can be related by a sparse matrix-vector product:

.. math::
    \boldsymbol{w} = \boldsymbol{C \, u}

where :math:`\boldsymbol{C}` is the curl matrix from edges to faces,
:math:`\boldsymbol{u}` is a vector that stores the components of :math:`\vec{u}` on cell edges,
and :math:`\boldsymbol{w}` is a vector that stores the components of :math:`\vec{w}` on cell faces such that:

.. math::
    \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_z} \end{bmatrix}
    \;\;\;\; \textrm{and} \;\;\;\; \begin{bmatrix} \boldsymbol{w_x} \\ \boldsymbol{w_y} \\ \boldsymbol{w_z} \end{bmatrix}

