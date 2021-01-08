r"""
Differential Operators New
==========================

For geophysical problems, the relationship between two physical quantities may include one of several differential operators:

    - **Divergence:** :math:`\nabla \cdot \vec{u} = \dfrac{\partial u_x}{\partial x} + \dfrac{\partial u_y}{\partial y} + \dfrac{\partial u_y}{\partial y}`
    - **Gradient:** :math:`\nabla \phi = \dfrac{\partial \phi}{\partial x}\hat{x} + \dfrac{\partial \phi}{\partial y}\hat{y} + \dfrac{\partial \phi}{\partial z}\hat{z}`
    - **Curl:** :math:`\nabla \times \vec{u} = \Bigg ( \dfrac{\partial u_y}{\partial z} - \dfrac{\partial u_z}{\partial y} \Bigg )\hat{x} - \Bigg ( \dfrac{\partial u_x}{\partial z} - \dfrac{\partial u_z}{\partial x} \Bigg )\hat{y} + \Bigg ( \dfrac{\partial u_x}{\partial y} - \dfrac{\partial u_y}{\partial x} \Bigg )\hat{z}`

When implementing the finite volume method, continuous variables are discretized to live at the cell centers, nodes, edges or faces of a mesh.
Thus for each differential operator, we need a discrete approximation that acts on the discrete variables living on the mesh.
For discretized quantities living on a mesh, sparse matricies can be used to approximate the differential operators according to
:cite:`haber2014,HymanShashkov1999`.



Numerical differential operators exist for 1D, 2D and 3D meshes. For each mesh
class (*Tensor mesh*, *Tree mesh*, *Curvilinear mesh*), the set of numerical
differential operators are properties that are only constructed when called.

Here we demonstrate:

    - How to construct and apply numerical differential operators
    - Mapping and dimensions
    - Applications for the transpose

"""

###############################################
# Import Packages
# ---------------
#

from discretize import TensorMesh, TreeMesh
import matplotlib.pyplot as plt
import numpy as np

# sphinx_gallery_thumbnail_number = 2

#############################################
# Numerical Differentiation in 1D
# -------------------------------
#
# Discrete approximations for differential operators are derived using the principles
# of numerical differentiation. In 1D, so long as a function :math:`f(x)` is sufficiently smooth
# within the interval :math:`[x-h/2, \; x+h/2]`, the derivative of the function at :math:`x` is
# approximated by:
# 
# .. math::
#     \frac{df(x)}{dx} \approx \frac{f(x+h/2) \; - \; f(x-h/2)}{h}
# 
# where the approximation becomes increasingly accurate as :math:`h \rightarrow 0`.
# This principle can be applied to construct differential operators in 2D and 3D.
# 
# .. figure:: ../../images/approximate_derivative.png
#     :align: center
#     :width: 300
# 
#     Approximating the derivative of :math:`f(x)` using numerical differentiation.
# 
# 
# Below we compute a scalar function on cell nodes and differentiate with
# respect to x using a 1D differential operator. We then compute the analytic
# derivative of function to validate the numerical differentiation.
#

# Create a uniform grid
h = np.ones(20)
mesh = TensorMesh([h], "C")

# Get node and cell center locations
x_nodes = mesh.vectorNx
x_centers = mesh.vectorCCx

# Compute function on nodes and derivative at cell centers
v = np.exp(-(x_nodes ** 2) / 4 ** 2)
dvdx = -(2 * x_centers / 4 ** 2) * np.exp(-(x_centers ** 2) / 4 ** 2)

# Derivative in x (gradient in 1D) from nodes to cell centers
G = mesh.nodalGrad
dvdx_approx = G * v

# Compare
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_axes([0.05, 0.01, 0.3, 0.85])
ax1.spy(G, markersize=5)
ax1.set_title("Sparse representation of G", pad=10)

ax2 = fig.add_axes([0.4, 0.06, 0.55, 0.85])
ax2.plot(x_nodes, v, "b-", x_centers, dvdx, "r-", x_centers, dvdx_approx, "ko")
ax2.set_title("Comparison plot")
ax2.legend(("function", "analytic derivative", "numeric derivative"))

fig.show()

#######################################################
# Divergence
# ----------
#
# Let us define a continuous scalar function :math:`\phi` and a continuous
# vector function :math:`\vec{u}` such that:
# 
# .. math::
#     \phi = \nabla \cdot \vec{u}
# 
# And let :math:`\boldsymbol{\phi}` and :math:`\boldsymbol{u}` be the
# discrete representations of :math:`\phi` and :math:`\vec{u}`
# that live on the mesh (centers, nodes, edges or faces), respectively.
# Provided we know the discrete values :math:`\boldsymbol{u}`,
# our goal is to use discrete differentiation to approximate the values of
# :math:`\boldsymbol{\phi}`.
# We begin by considering a single cell (2D or 3D). We let the indices
# :math:`i`, :math:`j` and :math:`k` 
# denote positions along the x, y and z axes, respectively.
# 
# .. figure:: ../../images/divergence_discretization.png
#     :align: center
#     :width: 600
# 
#     Discretization for approximating the divergence at the center of a single 2D cell (left) and 3D cell (right).
# 
# +-------------+-------------------------------------------------+-----------------------------------------------------+
# |             |                    **2D**                       |                       **3D**                        |
# +-------------+-------------------------------------------------+-----------------------------------------------------+
# | **center**  | :math:`(i,j)`                                   | :math:`(i,j,k)`                                     |
# +-------------+-------------------------------------------------+-----------------------------------------------------+
# | **x-faces** | :math:`(i-\frac{1}{2},j)\;\; (i+\frac{1}{2},j)` | :math:`(i-\frac{1}{2},j,k)\;\; (i+\frac{1}{2},j,k)` |
# +-------------+-------------------------------------------------+-----------------------------------------------------+
# | **y-faces** | :math:`(i,j-\frac{1}{2})\;\; (i,j+\frac{1}{2})` | :math:`(i,j-\frac{1}{2},k)\;\; (i,j+\frac{1}{2},k)` |
# +-------------+-------------------------------------------------+-----------------------------------------------------+
# | **z-faces** | N/A                                             | :math:`(i,j,k-\frac{1}{2})\;\; (i,j,k+\frac{1}{2})` |
# +-------------+-------------------------------------------------+-----------------------------------------------------+
# 
# As we will see, it makes the most sense for :math:`\boldsymbol{\phi}` to
# live at the cell centers and
# for the components of :math:`\boldsymbol{u}` to live on the faces.
# If :math:`u_x` lives on x-faces, then its discrete
# derivative with respect to :math:`x` lives at the cell center.
# And if :math:`u_y` lives on y-faces its discrete
# derivative with respect to :math:`y` lives at the cell center.
# Likewise for :math:`u_z`. Thus to approximate the
# divergence of :math:`\vec{u}` at the cell center, we simply need to
# sum the discrete derivatives of :math:`u_x`, :math:`u_y`
# and :math:`u_z` that are defined at the cell center. Where :math:`h_x`,
# :math:`h_y` and :math:`h_z` represent the dimension of the cell along the x, y and
# z directions, respectively:
# 
# .. math::
#     \begin{align}
#     \mathbf{In \; 2D:} \;\; \phi(i,j) \approx \; & \frac{u_x(i,j+\frac{1}{2}) - u_x(i,j-\frac{1}{2})}{h_x} \\
#     & + \frac{u_y(i+\frac{1}{2},j) - u_y(i-\frac{1}{2},j)}{h_y}
#     \end{align}
# 
# |
# 
# .. math::
#     \begin{align}
#     \mathbf{In \; 3D:} \;\; \phi(i,j,k) \approx \; & \frac{u_x(i+\frac{1}{2},j,k) - u_x(i-\frac{1}{2},j,k)}{h_x} \\
#     & + \frac{u_y(i,j+\frac{1}{2},k) - u_y(i,j-\frac{1}{2},k)}{h_y} \\
#     & + \frac{u_z(i,j,k+\frac{1}{2}) - u_z(i,j,k-\frac{1}{2})}{h_z}
#     \end{align}
# 
# 
# Ultimately we are trying to approximate the divergence at the center of
# every cell in a mesh. Adjacent cells share faces.
# If the components :math:`u_x`, :math:`u_y` and :math:`u_z` are
# continuous across their respective faces, then :math:`\boldsymbol{\phi}`
# and :math:`\boldsymbol{u}` can be related by a sparse matrix-vector product:
# 
# .. math::
#     \boldsymbol{\phi} = \boldsymbol{D \, u}
# 
# where :math:`\boldsymbol{D}` is the divergence matrix from faces to cell centers,
# :math:`\boldsymbol{\phi}` is a vector containing the discrete approximations
# of :math:`\phi` at all cell centers, and :math:`\boldsymbol{u}` stores
# the components of :math:`\vec{u}` on cell faces as a vector of the form:
# 
# .. math::
#     \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_z} \end{bmatrix}
#

#############################################
# Divergence: Mapping and Dimensions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When discretizing and solving differential equations, it is
# natural for certain quantities to be defined at particular locations on the
# mesh; e.g.:
#
#    - Scalar quantities on nodes or at cell centers
#    - Vector quantities on cell edges or on cell faces
#
# As such, numerical differential operators frequently map from one part of
# the mesh to another. For example, the gradient acts on a scalar quantity
# an results in a vector quantity. As a result, the numerical gradient
# operator may map from nodes to edges or from cell centers to faces.
#
# Here we explore the dimensions of the divergence
# operator for a 3D tensor mesh. This can be extended to other mesh types.
#

# Create a uniform grid
h = np.ones(20)
mesh = TensorMesh([h, h, h], "CCC")

# Get differential operator
DIV = mesh.faceDiv  # Divergence from faces to cell centers

# Spy Plot
fig = plt.figure(figsize=(6, 3))
ax1 = fig.add_axes([0.1, 0.05, 0.8, 0.8])
ax1.spy(DIV, markersize=0.5)
ax1.set_title("Divergence (faces to centers)", pad=20)
fig.show()

# Print some properties
print("Divergence:")
print("- Number of faces:", str(mesh.nF))
print("- Number of cells:", str(mesh.nC))
print("- Dimensions of operator:", str(mesh.nC), "x", str(mesh.nF))
print("- Number of non-zero elements:", str(DIV.nnz), "\n")


#############################################
# 2D Divergence Example
# ^^^^^^^^^^^^^^^^^^^^^
#
# Here we apply the divergence operator to a function
# defined on a 2D tensor mesh. We then plot the results.
#

# Create a uniform grid
h = np.ones(20)
mesh = TensorMesh([h, h], "CC")

# Get differential operator
DIV = mesh.faceDiv  # Divergence from faces to cell centers

# Evaluate divergence of a vector function in x and y
faces_x = mesh.gridFx
faces_y = mesh.gridFy

vx = (faces_x[:, 0] / np.sqrt(np.sum(faces_x ** 2, axis=1))) * np.exp(
    -(faces_x[:, 0] ** 2 + faces_x[:, 1] ** 2) / 6 ** 2
)

vy = (faces_y[:, 1] / np.sqrt(np.sum(faces_y ** 2, axis=1))) * np.exp(
    -(faces_y[:, 0] ** 2 + faces_y[:, 1] ** 2) / 6 ** 2
)

v = np.r_[vx, vy]
div_v = DIV * v

# Plot divergence of v
fig = plt.figure(figsize=(10, 4.5))

ax1 = fig.add_subplot(121)
mesh.plotImage(
    v, ax=ax1, v_type="F", view="vec", stream_opts={"color": "w", "density": 1.0}
)
ax1.set_title("v at cell faces")

ax2 = fig.add_subplot(122)
mesh.plotImage(div_v, ax=ax2)
ax2.set_title("divergence of v at cell centers")

fig.show()

#########################################################
# Tree Mesh Divergence Example
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# For a tree mesh, there needs to be special attention taken for the hanging
# faces to achieve second order convergence for the divergence operator.
# Although the divergence cannot be constructed through Kronecker product
# operations, the initial steps are exactly the same for calculating the
# stencil, volumes, and areas. This yields a divergence defined for every
# cell in the mesh using all faces. There is, however, redundant information
# when hanging faces are included.
#

mesh = TreeMesh([[(1, 16)], [(1, 16)]], levels=4)
mesh.insert_cells(np.array([5.0, 5.0]), np.array([3]))
mesh.number()

fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_subplot(211)

mesh.plotGrid(centers=True, nodes=False, ax=ax1)
ax1.axis("off")
ax1.set_title("Simple QuadTree Mesh")
ax1.set_xlim([-1, 17])
ax1.set_ylim([-1, 17])

for ii, loc in zip(range(mesh.nC), mesh.gridCC):
    ax1.text(loc[0] + 0.2, loc[1], "{0:d}".format(ii), color="r")

ax1.plot(mesh.gridFx[:, 0], mesh.gridFx[:, 1], "g>")
for ii, loc in zip(range(mesh.nFx), mesh.gridFx):
    ax1.text(loc[0] + 0.2, loc[1], "{0:d}".format(ii), color="g")

ax1.plot(mesh.gridFy[:, 0], mesh.gridFy[:, 1], "m^")
for ii, loc in zip(range(mesh.nFy), mesh.gridFy):
    ax1.text(loc[0] + 0.2, loc[1] + 0.2, "{0:d}".format((ii + mesh.nFx)), color="m")

ax2 = fig.add_subplot(212)
ax2.spy(mesh.faceDiv)
ax2.set_title("Face Divergence")
ax2.set_ylabel("Cell Number")
ax2.set_xlabel("Face Number")


##################################################
# Gradient
# --------
# 
# Let us define a continuous scalar function :math:`\phi` and a continuous
# vector function :math:`\vec{u}` such that:
# 
# .. math::
#     \vec{u} = \nabla \phi
# 
# And let :math:`\boldsymbol{\phi}` and :math:`\boldsymbol{u}` be the
# discrete representations of :math:`\phi` and :math:`\vec{u}`
# that live on the mesh (centers, nodes, edges or faces), respectively.
# Provided we know the discrete values :math:`\boldsymbol{\phi}`,
# our goal is to use discrete differentiation to approximate the vector
# components of :math:`\boldsymbol{u}`.
# We begin by considering a single cell (2D or 3D). We let the indices
# :math:`i`, :math:`j` and :math:`k` 
# denote positions along the x, y and z axes, respectively.
# 
# .. figure:: ../../images/gradient_discretization.png
#     :align: center
#     :width: 600
# 
#     Discretization for approximating the gradient on the edges of a single 2D cell (left) and 3D cell (right).
# 
# As we will see, it makes the most sense for :math:`\boldsymbol{\phi}`
# to live at the cell nodes and for the components of
# :math:`\boldsymbol{u}` to live on corresponding edges. If :math:`\phi` lives on the nodes, then:
# 
#     - the partial derivative :math:`\dfrac{\partial \phi}{\partial x}\hat{x}` lives on x-edges,
#     - the partial derivative :math:`\dfrac{\partial \phi}{\partial y}\hat{y}` lives on y-edges, and
#     - the partial derivative :math:`\dfrac{\partial \phi}{\partial z}\hat{z}` lives on z-edges
# 
# Thus to approximate the gradient of :math:`\phi`, 
# we simply need to take discrete derivatives of :math:`\phi` with respect
# to :math:`x`, :math:`y` and :math:`z`,
# and organize the resulting vector components on the corresponding edges.
# Let :math:`h_x`, :math:`h_y` and :math:`h_z` represent the dimension of
# the cell along the x, y and z directions, respectively.
# 
# **In 2D**, the value of :math:`\phi` at 4 node locations is used to
# approximate the vector components of the
# gradient at 4 edges locations (2 x-edges and 2 y-edges) as follows:
# 
# .. math::
#     \begin{align}
#     u_x \Big ( i+\frac{1}{2},j \Big ) \approx \; & \frac{\phi (i+1,j) - \phi (i,j)}{h_x} \\
#     u_x \Big ( i+\frac{1}{2},j+1 \Big ) \approx \; & \frac{\phi (i+1,j+1) - \phi (i,j+1)}{h_x} \\
#     u_y \Big ( i,j+\frac{1}{2} \Big ) \approx \; & \frac{\phi (i,j+1) - \phi (i,j)}{h_y} \\
#     u_y \Big ( i+1,j+\frac{1}{2} \Big ) \approx \; & \frac{\phi (i+1,j+1) - \phi (i+1,j)}{h_y}
#     \end{align}
# 
# **In 3D**, the value of :math:`\phi` at 8 node locations is used to
# approximate the vector components of the
# gradient at 12 edges locations (4 x-edges, 4 y-edges and 4 z-edges).
# An example of the approximation for each vector component is given below:
# 
# .. math::
#     \begin{align}
#     u_x \Big ( i+\frac{1}{2},j,k \Big ) \approx \; & \frac{\phi (i+1,j,k) - \phi (i,j,k)}{h_x} \\
#     u_y \Big ( i,j+\frac{1}{2},k \Big ) \approx \; & \frac{\phi (i,j+1,k) - \phi (i,j,k)}{h_y} \\
#     u_z \Big ( i,j,k+\frac{1}{2} \Big ) \approx \; & \frac{\phi (i,j,k+1) - \phi (i,j,k)}{h_z}
#     \end{align}
# 
# 
# Ultimately we are trying to approximate the vector components of the
# gradient at all edges of a mesh.
# Adjacent cells share nodes. If :math:`\phi` is continuous at the nodes,
# then :math:`\boldsymbol{\phi}` and :math:`\boldsymbol{u}`
# can be related by a sparse matrix-vector product:
# 
# .. math::
#     \boldsymbol{u} = \boldsymbol{G \, \phi}
# 
# where :math:`\boldsymbol{G}` is the gradient matrix that maps from
# nodes to edges, :math:`\boldsymbol{\phi}` is a vector containing
# :math:`\phi` at all nodes,  and :math:`\boldsymbol{u}` stores the
# components of :math:`\vec{u}` on cell edges as a vector of the form:
# 
# .. math::
#     \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_z} \end{bmatrix}


#############################################
# Gradient: Mapping and Dimensions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Here we explore the dimensions of the gradient operator for a 3D tensor mesh.
# This can be extended to other mesh types.
#

# Create a uniform grid
h = np.ones(20)
mesh = TensorMesh([h, h, h], "CCC")

# Get differential operators
GRAD = mesh.nodalGrad  # Gradient from nodes to edges

# Spy Plot
fig = plt.figure(figsize=(3, 6))
ax1 = fig.add_axes([0.15, 0.05, 0.75, 0.8])
ax1.spy(GRAD, markersize=0.5)
ax1.set_title("Gradient (nodes to edges)")
fig.show()

# Print some properties
print("\n Gradient:")
print("- Number of nodes:", str(mesh.nN))
print("- Number of edges:", str(mesh.nE))
print("- Dimensions of operator:", str(mesh.nE), "x", str(mesh.nN))
print("- Number of non-zero elements:", str(GRAD.nnz), "\n")


#############################################
# 2D Gradient Example
# ^^^^^^^^^^^^^^^^^^^
#
# Here we apply the gradient operator to a
# function defined on a 2D tensor mesh. We then plot the results.
#

# Create a uniform grid
h = np.ones(20)
mesh = TensorMesh([h, h], "CC")

# Get differential operators
GRAD = mesh.nodalGrad  # Gradient from nodes to edges

# Evaluate gradient of a scalar function
nodes = mesh.gridN
u = np.exp(-(nodes[:, 0] ** 2 + nodes[:, 1] ** 2) / 4 ** 2)
grad_u = GRAD * u

# Plot Gradient of u
fig = plt.figure(figsize=(10, 4.5))

ax1 = fig.add_subplot(121)
mesh.plotImage(u, ax=ax1, v_type="N")
ax1.set_title("u at cell centers")

ax2 = fig.add_subplot(122)
mesh.plotImage(
    grad_u, ax=ax2, v_type="E", view="vec", stream_opts={"color": "w", "density": 1.0}
)
ax2.set_title("gradient of u on edges")

fig.show()


##############################################
# Curl
# ----
#
# Let us define two continuous vector functions :math:`\vec{u}` and
# :math:`\vec{w}` such that:
# 
# .. math::
#     \vec{w} = \nabla \times \vec{u}
# 
# And let :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` be the
# discrete representations of :math:`\vec{u}` and :math:`\vec{w}`
# that live on the mesh (centers, nodes, edges or faces), respectively.
# Provided we know the discrete values :math:`\boldsymbol{u}`,
# our goal is to use discrete differentiation to approximate the vector
# components of :math:`\boldsymbol{w}`.
# We begin by considering a single 3D cell. We let the indices :math:`i`,
# :math:`j` and :math:`k` denote positions along the x, y and z axes, respectively.
# 
# .. figure:: ../../images/curl_discretization.png
#     :align: center
#     :width: 800
# 
#     Discretization for approximating the x, y and z components of the curl on the respective faces of a 3D cell.
# 
# 
# As we will see, it makes the most sense for the vector components of
# :math:`\boldsymbol{u}` to live on the edges
# for the vector components of :math:`\boldsymbol{w}` to live the faces.
# In this case, we need to approximate:
# 
# 
#     - the partial derivatives :math:`\dfrac{\partial u_y}{\partial z}` and :math:`\dfrac{\partial u_z}{\partial y}` to compute :math:`w_x`,
#     - the partial derivatives :math:`\dfrac{\partial u_x}{\partial z}` and :math:`\dfrac{\partial u_z}{\partial x}` to compute :math:`w_y`, and
#     - the partial derivatives :math:`\dfrac{\partial u_x}{\partial y}` and :math:`\dfrac{\partial u_y}{\partial x}` to compute :math:`w_z`
# 
# **In 3D**, discrete values at 12 edge locations (4 x-edges, 4 y-edges and 4 z-edges) are used to
# approximate the vector components of the curl at 6 face locations (2 x-faces, 2-faces and 2 z-faces).
# An example of the approximation for each vector component is given below:
# 
# .. math::
#     \begin{align}
#     w_x \Big ( i,j \! +\!\!\frac{1}{2},k \! +\!\!\frac{1}{2} \Big ) \!\approx\! \; &
#     \!\Bigg ( \! \frac{u_z (i,j \! +\!\!1,k \! +\!\!\frac{1}{2})  \! -\! u_z (i,j,k \! +\!\!\frac{1}{2})}{h_y} \Bigg) \!
#     \! -\! \!\Bigg ( \! \frac{u_y (i,j \! +\!\!\frac{1}{2},k \! +\!\!1)  \! -\! u_y (i,j \! +\!\!\frac{1}{2},k)}{h_z} \Bigg) \! \\
#     & \\
#     w_y \Big ( i \! +\!\!\frac{1}{2},j,k \! +\!\!\frac{1}{2} \Big ) \!\approx\! \; &
#     \!\Bigg ( \! \frac{u_x (i \! +\!\!\frac{1}{2},j,k \! +\!\!1)  \! -\! u_x (i \! +\!\!\frac{1}{2},j,k)}{h_z} \Bigg)
#     \! -\! \!\Bigg ( \! \frac{u_z (i \! +\!\!1,j,k \! +\!\!\frac{1}{2})  \! -\! u_z (i,j,k \! +\!\!\frac{1}{2})}{h_x} \Bigg) \! \\
#     & \\
#     w_z \Big ( i \! +\!\!\frac{1}{2},j \! +\!\!\frac{1}{2},k \Big ) \!\approx\! \; &
#     \!\Bigg ( \! \frac{u_y (i \! +\!\!1,j \! +\!\!\frac{1}{2},k)  \! -\! u_y (i,j \! +\!\!\frac{1}{2},k)}{h_x} \Bigg )
#     \! -\! \!\Bigg ( \! \frac{u_x (i \! +\!\!\frac{1}{2},j \! +\!\!1,k)  \! -\! u_x (i \! +\!\!\frac{1}{2},j,k)}{h_y} \Bigg) \!
#     \end{align}
# 
# 
# Ultimately we are trying to approximate the curl on all the faces within a mesh.
# Adjacent cells share edges. If the components :math:`u_x`, :math:`u_y` and :math:`u_z` are
# continuous across at the edges, then :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}`
# can be related by a sparse matrix-vector product:
# 
# .. math::
#     \boldsymbol{w} = \boldsymbol{C \, u}
# 
# where :math:`\boldsymbol{C}` is the curl matrix from edges to faces,
# :math:`\boldsymbol{u}` is a vector that stores the components of :math:`\vec{u}` on cell edges,
# and :math:`\boldsymbol{w}` is a vector that stores the components of
# :math:`\vec{w}` on cell faces such that:
# 
# .. math::
#     \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_z} \end{bmatrix}
#     \;\;\;\; \textrm{and} \;\;\;\; \begin{bmatrix} \boldsymbol{w_x} \\ \boldsymbol{w_y} \\ \boldsymbol{w_z} \end{bmatrix}

#############################################
# Curl: Mapping and Dimensions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Here we explore the dimensions of the curl
# operator for a 3D tensor mesh. This can be extended to other mesh types.
#

# Create a uniform grid
h = np.ones(20)
mesh = TensorMesh([h, h, h], "CCC")

# Get differential operators
CURL = mesh.edgeCurl  # Curl edges to cell centers

# Spy Plot
fig = plt.figure(figsize=(4, 4))
ax1 = fig.add_axes([0.1, 0.05, 0.8, 0.8])
ax1.spy(CURL, markersize=0.5)
ax1.set_title("Curl (edges to faces)")

fig.show()

# Print some properties
print("Curl:")
print("- Number of faces:", str(mesh.nF))
print("- Number of edges:", str(mesh.nE))
print("- Dimensions of operator:", str(mesh.nE), "x", str(mesh.nF))
print("- Number of non-zero elements:", str(CURL.nnz))


#############################################
# 2D Curl Example
# ^^^^^^^^^^^^^^^
#
# Here we apply the curl operator to a
# function defined on a 2D tensor mesh. We then plot the results.
#

# Create a uniform grid
h = np.ones(20)
mesh = TensorMesh([h, h], "CC")

# Get differential operator
CURL = mesh.edgeCurl  # Curl edges to cell centers (goes to faces in 3D)

# Evaluate curl of a vector function in x and y
edges_x = mesh.gridEx
edges_y = mesh.gridEy

wx = (-edges_x[:, 1] / np.sqrt(np.sum(edges_x ** 2, axis=1))) * np.exp(
    -(edges_x[:, 0] ** 2 + edges_x[:, 1] ** 2) / 6 ** 2
)

wy = (edges_y[:, 0] / np.sqrt(np.sum(edges_y ** 2, axis=1))) * np.exp(
    -(edges_y[:, 0] ** 2 + edges_y[:, 1] ** 2) / 6 ** 2
)

w = np.r_[wx, wy]
curl_w = CURL * w

# Plot curl of w
fig = plt.figure(figsize=(10, 4.5))

ax1 = fig.add_subplot(121)
mesh.plotImage(
    w, ax=ax1, v_type="E", view="vec", stream_opts={"color": "w", "density": 1.0}
)
ax1.set_title("w at cell edges")

ax2 = fig.add_subplot(122)
mesh.plotImage(curl_w, ax=ax2)
ax2.set_title("curl of w at cell centers")

fig.show()

#########################################################
# Vector Calculus Identities
# --------------------------
#
# Here we show that vector calculus identities hold for the discrete
# differential operators. Namely that for a scalar quantity :math:`\phi` and
# a vector quantity :math:`\mathbf{v}`:
#
# .. math::
#     \begin{align}
#     &\nabla \times (\nabla \phi ) = 0 \\
#     &\nabla \cdot (\nabla \times \mathbf{v}) = 0
#     \end{align}
#
#
# We do this by computing the CURL*GRAD and DIV*CURL matricies. We then
# plot the sparse representations and show neither contain any non-zero
# entries; **e.g. each is just a matrix of zeros**.
#

# Create a mesh
h = 5 * np.ones(20)
mesh = TensorMesh([h, h, h], "CCC")

# Get operators
GRAD = mesh.nodalGrad  # nodes to edges
DIV = mesh.faceDiv  # faces to centers
CURL = mesh.edgeCurl  # edges to faces

# Plot
fig = plt.figure(figsize=(11, 7))

ax1 = fig.add_axes([0.12, 0.1, 0.2, 0.8])
ax1.spy(CURL * GRAD, markersize=0.5)
ax1.set_title("CURL*GRAD")

ax2 = fig.add_axes([0.35, 0.64, 0.6, 0.25])
ax2.spy(DIV * CURL, markersize=0.5)
ax2.set_title("DIV*CURL", pad=20)
