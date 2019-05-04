"""
Differential Operators
======================

For variables living on a discretized mesh, sparse matricies can be used to
approximate the following differential operators:
    
    - gradient (:math:`\nabla \phi` )
    - divergence (:math:`\nabla \cdot \mathbf{v}` )
    - curl (:math:`\nabla \times \mathbf{v}` )
    - scalar Laplacian (:math:`\Delta` )

Numerical differential operators exist for 1D, 2D and 3D meshes. For each mesh
class (*Tensor mesh*, *Tree mesh*, *Curvilinear mesh*), the set of numerical
differential operators are properties that are only constructed when called.

Here we demonstrate:

    - How to construct and apply numerical differential operators to a vector
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
from IPython.display import display
import numpy as np

# sphinx_gallery_thumbnail_number = 3


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
fig = plt.figure(figsize=(8, 4))
Ax1 = fig.add_subplot(121)
Ax1.spy(G, markersize=5)
Ax1.set_title('Sparse representation of G', pad=10)

Ax2 = fig.add_subplot(122)
Ax2.plot(x_nodes, v)
Ax2.plot(x_centers, dvdx)
Ax2.plot(x_centers, dvdx_approx, 'o')
Ax2.set_title('Comparison plot')
Ax2.legend(('function','derivative','approx. deriv.'))

display(fig)
plt.close()


#############################################
# Mapping and Dimensions
# ----------------------
#
# When solving differential equations on a numerical grid (mesh), it is
# natural for certain quantities to be defined at particular locations on the
# mesh; e.g.:
#
#    - Scalar quantities on nodes or at cell centers
#    - Vector quantities on cell edges or on cell faces
#
# Here we explore the dimensions of the gradient, divergence and curl
# operators for a 3D tensor mesh. And we discuss the corresponding mapping.
#

# Create a uniform grid
h = np.ones(20)
mesh = TensorMesh([h, h, h], 'CCC')

# Get differential operators
GRAD = mesh.nodalGrad  # Gradient from nodes to edges
DIV = mesh.faceDiv     # Divergence from faces to cell centers
CURL = mesh.edgeCurl   # Curl edges to cell centers


fig = plt.figure(figsize=(8, 8))

Ax1 = fig.add_axes([0, 0, 0.23, .7])
Ax1.spy(GRAD, markersize=0.5)
Ax1.set_title('Gradient (nodes to edges)')

Ax2 = fig.add_axes([0.35, 0.7, 0.63, 0.25])
Ax2.spy(DIV, markersize=0.5)
Ax2.set_title('Divergence (faces to centers)', pad=20)

Ax3 = fig.add_axes([0.33, 0, 0.65, 0.65])
Ax3.spy(CURL, markersize=0.5)
Ax3.set_title('Curl (edges to faces)')

display(fig)
plt.close()

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
# functions defined on the 2D tensor mesh.
#

# Create a uniform grid
h = np.ones(20)
mesh = TensorMesh([h, h], 'CC')

# Get differential operators
GRAD = mesh.nodalGrad  # Gradient from nodes to edges
DIV = mesh.faceDiv     # Divergence from faces to cell centers
CURL = mesh.edgeCurl   # Curl edges to cell centers

# Evaluate gradient of a scalar function
nodes = mesh.gridN
u = np.exp(-(nodes[:, 0]**2 + nodes[:, 1]**2) / 4**2)
grad_u = GRAD*u

# Evaluate divergence of a vector function in x and y
faces_x = mesh.gridFx
faces_y = mesh.gridFy
vx = (faces_x[:,0]/np.sqrt(np.sum(faces_x**2, axis=1))) * np.exp(-(faces_x[:, 0]**2 + faces_x[:, 1]**2) / 6**2)
vy = (faces_y[:,1]/np.sqrt(np.sum(faces_y**2, axis=1))) * np.exp(-(faces_y[:, 0]**2 + faces_y[:, 1]**2) / 6**2)
v = np.r_[vx, vy]
div_v = DIV*v

# Evaluate curl of a vector function in x and y
edges_x = mesh.gridEx
edges_y = mesh.gridEy
wx = (-edges_x[:,1]/np.sqrt(np.sum(edges_x**2, axis=1))) * np.exp(-(edges_x[:, 0]**2 + edges_x[:, 1]**2) / 6**2)
wy = ( edges_y[:,0]/np.sqrt(np.sum(edges_y**2, axis=1))) * np.exp(-(edges_y[:, 0]**2 + edges_y[:, 1]**2) / 6**2)
w = np.r_[wx, wy]
curl_w = CURL*w

# Plot Gradient
fig = plt.figure(figsize=(10, 5))

Ax1 = fig.add_subplot(121)
mesh.plotImage(u, ax=Ax1, vType='N')
Ax1.set_title('Function at cell centers')

Ax2 = fig.add_subplot(122)
mesh.plotImage(grad_u, ax=Ax2, vType='E', view='vec', streamOpts={'color':'w', 'density':1.0})
Ax2.set_title('Stream plot of gradient')

display(fig)
plt.close()

# Plot divergence
fig = plt.figure(figsize=(10, 5))

Ax1 = fig.add_subplot(121)
mesh.plotImage(v, ax=Ax1, vType='F', view='vec', streamOpts={'color':'w', 'density':1.0})
Ax1.set_title('Function at cell faces')

Ax2 = fig.add_subplot(122)
mesh.plotImage(div_v, ax=Ax2)
Ax2.set_title('Divergence at cell centers')

display(fig)
plt.close()

# Plot curl
fig = plt.figure(figsize=(10, 5))

Ax1 = fig.add_subplot(121)
mesh.plotImage(w, ax=Ax1, vType='E', view='vec', streamOpts={'color':'w', 'density':1.0})
Ax1.set_title('Function at cell edges')

Ax2 = fig.add_subplot(122)
mesh.plotImage(curl_w, ax=Ax2)
Ax2.set_title('Curl at cell centers')

display(fig)
plt.close()










