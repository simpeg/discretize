"""
Differential Operators
======================

For discretized quantities living on a mesh, sparse matricies can be used to
approximate the following differential operators:

    - gradient: :math:`\\nabla \phi`
    - divergence: :math:`\\nabla \cdot \mathbf{v}`
    - curl: :math:`\\nabla \\times \mathbf{v}`
    - scalar Laplacian: :math:`\Delta \mathbf{v}`

Numerical differential operators exist for 1D, 2D and 3D meshes. For each mesh
class (*Tensor mesh*, *Tree mesh*, *Curvilinear mesh*), the set of numerical
differential operators are properties that are only constructed when called.

Here we demonstrate:

    - How to construct and apply numerical differential operators
    - Mapping and dimensions
    - Applications for the transpose
    

"""

###############################################
#
# Import Packages
# ---------------
#
# Here we import the packages required for this tutorial.
#

from discretize import TensorMesh
import matplotlib.pyplot as plt
import numpy as np

# sphinx_gallery_thumbnail_number = 2


#############################################
# 1D Example
# ----------
#
# Here we compute a scalar function on cell nodes and differentiate with
# respect to x. We then compute the analytic derivative of function to validate
# the numerical differentiation.
#

# Create a uniform grid
h = np.ones(20)
mesh = TensorMesh([h], 'C')

# Get node and cell center locations
x_nodes = mesh.vectorNx
x_centers = mesh.vectorCCx

# Compute function on nodes and derivative at cell centers
v = np.exp(-(x_nodes**2) / 4**2)
dvdx = -(2*x_centers / 4**2)*np.exp(-(x_centers**2) / 4**2)

# Derivative in x (gradient in 1D) from nodes to cell centers
G = mesh.nodalGrad
dvdx_approx = G*v

# Compare
fig = plt.figure(figsize=(12, 4))
Ax1 = fig.add_axes([0.03, 0.01, 0.3, 0.92])
Ax1.spy(G, markersize=5)
Ax1.set_title('Sparse representation of G', pad=10)

Ax2 = fig.add_axes([0.4, 0.06, 0.55, 0.88])
Ax2.plot(x_nodes, v, 'b-',
         x_centers, dvdx, 'r-',
         x_centers, dvdx_approx, 'ko'
         )
Ax2.set_title('Comparison plot')
Ax2.legend(('function', 'analytic derivative', 'numeric derivative'))

fig.show()


#############################################
# Mapping and Dimensions
# ----------------------
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
# Here we explore the dimensions of the gradient, divergence and curl
# operators for a 3D tensor mesh. This can be extended to other mesh types.
#

# Create a uniform grid
h = np.ones(20)
mesh = TensorMesh([h, h, h], 'CCC')

# Get differential operators
GRAD = mesh.nodalGrad  # Gradient from nodes to edges
DIV = mesh.faceDiv     # Divergence from faces to cell centers
CURL = mesh.edgeCurl   # Curl edges to cell centers


fig = plt.figure(figsize=(9, 8))

Ax1 = fig.add_axes([0.07, 0, 0.20, .7])
Ax1.spy(GRAD, markersize=0.5)
Ax1.set_title('Gradient (nodes to edges)')

Ax2 = fig.add_axes([0.345, 0.73, 0.59, 0.185])
Ax2.spy(DIV, markersize=0.5)
Ax2.set_title('Divergence (faces to centers)', pad=20)

Ax3 = fig.add_axes([0.31, 0.05, 0.67, 0.60])
Ax3.spy(CURL, markersize=0.5)
Ax3.set_title('Curl (edges to faces)')

fig.show()

# Print some properties
print('\n Gradient:')
print('- Number of nodes:', str(mesh.nN))
print('- Number of edges:', str(mesh.nE))
print('- Dimensions of operator:', str(mesh.nE), 'x', str(mesh.nN))
print('- Number of non-zero elements:', str(GRAD.nnz), '\n')

print('Divergence:')
print('- Number of faces:', str(mesh.nF))
print('- Number of cells:', str(mesh.nC))
print('- Dimensions of operator:', str(mesh.nC), 'x', str(mesh.nF))
print('- Number of non-zero elements:', str(DIV.nnz), '\n')

print('Curl:')
print('- Number of faces:', str(mesh.nF))
print('- Number of edges:', str(mesh.nE))
print('- Dimensions of operator:', str(mesh.nE), 'x', str(mesh.nF))
print('- Number of non-zero elements:', str(CURL.nnz))


#############################################
# 2D Example
# ----------
#
# Here we apply the gradient, divergence and curl operators to a set of
# functions defined on a 2D tensor mesh. We then plot the results.
#

# Create a uniform grid
h = np.ones(20)
mesh = TensorMesh([h, h], 'CC')

# Get differential operators
GRAD = mesh.nodalGrad  # Gradient from nodes to edges
DIV = mesh.faceDiv     # Divergence from faces to cell centers
CURL = mesh.edgeCurl   # Curl edges to cell centers (goes to faces in 3D)

# Evaluate gradient of a scalar function
nodes = mesh.gridN
u = np.exp(-(nodes[:, 0]**2 + nodes[:, 1]**2) / 4**2)
grad_u = GRAD*u

# Evaluate divergence of a vector function in x and y
faces_x = mesh.gridFx
faces_y = mesh.gridFy

vx = (
      (faces_x[:, 0]/np.sqrt(np.sum(faces_x**2, axis=1))) *
      np.exp(-(faces_x[:, 0]**2 + faces_x[:, 1]**2) / 6**2)
      )

vy = (
      (faces_y[:, 1]/np.sqrt(np.sum(faces_y**2, axis=1))) *
      np.exp(-(faces_y[:, 0]**2 + faces_y[:, 1]**2) / 6**2)
      )

v = np.r_[vx, vy]
div_v = DIV*v

# Evaluate curl of a vector function in x and y
edges_x = mesh.gridEx
edges_y = mesh.gridEy

wx = (
      (-edges_x[:, 1]/np.sqrt(np.sum(edges_x**2, axis=1))) *
      np.exp(-(edges_x[:, 0]**2 + edges_x[:, 1]**2) / 6**2)
      )

wy = (
      (edges_y[:, 0]/np.sqrt(np.sum(edges_y**2, axis=1))) *
      np.exp(-(edges_y[:, 0]**2 + edges_y[:, 1]**2) / 6**2)
      )

w = np.r_[wx, wy]
curl_w = CURL*w

# Plot Gradient of u
fig = plt.figure(figsize=(10, 5))

Ax1 = fig.add_subplot(121)
mesh.plotImage(u, ax=Ax1, vType='N')
Ax1.set_title('u at cell centers')

Ax2 = fig.add_subplot(122)
mesh.plotImage(grad_u, ax=Ax2, vType='E', view='vec',
               streamOpts={'color': 'w', 'density': 1.0})
Ax2.set_title('gradient of u on edges')

fig.show()

# Plot divergence of v
fig = plt.figure(figsize=(10, 5))

Ax1 = fig.add_subplot(121)
mesh.plotImage(v, ax=Ax1, vType='F', view='vec',
               streamOpts={'color': 'w', 'density': 1.0})
Ax1.set_title('v at cell faces')

Ax2 = fig.add_subplot(122)
mesh.plotImage(div_v, ax=Ax2)
Ax2.set_title('divergence of v at cell centers')

fig.show()

# Plot curl of w
fig = plt.figure(figsize=(10, 5))

Ax1 = fig.add_subplot(121)
mesh.plotImage(w, ax=Ax1, vType='E', view='vec',
               streamOpts={'color': 'w', 'density': 1.0})
Ax1.set_title('w at cell edges')

Ax2 = fig.add_subplot(122)
mesh.plotImage(curl_w, ax=Ax2)
Ax2.set_title('curl of w at cell centers')

fig.show()
