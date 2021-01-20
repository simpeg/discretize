r"""
Constitutive Relations (Isotropic)
==================================

A constitutive relationship quantifies the response of a material to an external stimulus.
Examples include Ohm's law and Hooke's law.
When the constitutive relationship is isotropic, the relationship
is between quantities is may be defined by a constant.

When solving PDEs using the finite volume approach, inner products may
contain constitutive relations. In this tutorial, you will learn how to:

    - construct the inner-product matrix in the case of isotropic constitutive relations
    - construct the inverse of the inner-product matrix
    - work with constitutive relations defined by the reciprocal of a parameter

"""

#####################################################
# Background Theory
# -----------------
# 
# Let :math:`\vec{v}` and :math:`\vec{w}` be two physically related
# quantities. If their relationship is isotropic (defined by a constant
# :math:`\sigma`), then the constitutive relation is given by:
# 
# .. math::
#     \vec{v} = \sigma \vec{w}
# 
# The inner product between a vector :math:`\vec{u}` and the right-hand side
# of this expression is given by:
# 
# .. math::
#     (\vec{u}, \sigma \vec{w} ) = \int_\Omega \vec{v} \cdot \sigma \vec{w} \, dv
# 
# Assuming the constitutive relationship is spatially invariant within each
# cell of a mesh, the inner product can be approximated numerically using
# an *inner-product matrix* such that:
# 
# .. math::
#     (\vec{u}, \sigma \vec{w} ) \approx \mathbf{u^T M w}
# 
# where the inner product matrix :math:`\mathbf{M}` depends on:
# 
#     1. the dimensions and discretization of the mesh
#     2. where discrete variables :math:`\mathbf{u}` and :math:`\mathbf{w}` live
#     3. the spatial distribution of the property :math:`\sigma`
# 
# 

####################################################
# Import Packages
# ---------------
#

from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_number = 1

#####################################################
# Inner Product for a Single Cell in 2D and 3D
# --------------------------------------------
#
# Here we construct the inner product matrices for a single cell in 2D and in 3D.
# When approximating the inner product according to the finite volume approach,
# the constitutive parameters are defined at cell centers; even if the
# fields/fluxes live at cell edges/faces.
#

# Create a single 3D cell
h = np.ones(1)
mesh_2d = TensorMesh([h, h])
mesh_3d = TensorMesh([h, h, h])

# Constant defining the constitutive relation for the cell
sig = 10
sig = np.array([sig])

# Inner products for a single 2D cell
Mf_2d = mesh_2d.get_face_inner_product(sig)  # Faces inner product matrix
Me_2d = mesh_2d.get_edge_inner_product(sig)  # Edges inner product matrix

# Inner products for a single 2D cell
Mf_3d = mesh_3d.get_face_inner_product(sig)  # Faces inner product matrix
Me_3d = mesh_3d.get_edge_inner_product(sig)  # Edges inner product matrix

# Plotting matrix entries
fig = plt.figure(figsize=(9, 9))

ax11 = fig.add_subplot(221)
ax11.imshow(Mf_2d.todense())
ax11.set_title("Faces inner product (2D)")

ax12 = fig.add_subplot(222)
ax12.imshow(Me_2d.todense())
ax12.set_title("Edges inner product (2D)")

ax21 = fig.add_subplot(223)
ax21.imshow(Mf_3d.todense())
ax21.set_title("Faces inner product (3D)")

ax22 = fig.add_subplot(224)
ax22.imshow(Me_3d.todense())
ax22.set_title("Edges inner product (3D)")

#############################################################
# Spatially Variant Parameters
# ----------------------------
#
# In practice, the parameter :math:`\sigma` will vary spatially.
# In this case, we define the parameter :math:`\sigma` for each cell.
# When creating the inner product matrix, we enter these parameters as
# a numpy array. This is demonstrated below. Properties of the resulting
# inner product matrices are discussed.
#

# Create a small 3D mesh
h = np.ones(5)
mesh = TensorMesh([h, h, h])

# Define contitutive relation for each cell
sig = np.random.rand(mesh.nC)

# Define inner product matrices
Me = mesh.get_edge_inner_product(sig)  # Edges inner product matrix
Mf = mesh.get_face_inner_product(sig)  # Faces inner product matrix

# Properties of inner product matricies
print("\n EDGE INNER PRODUCT MATRIX")
print("- Number of edges              :", mesh.nE)
print("- Dimensions of operator       :", str(mesh.nE), "x", str(mesh.nE))
print("- Number non-zero (isotropic)  :", str(Me.nnz), "\n")

print("\n FACE INNER PRODUCT MATRIX")
print("- Number of faces              :", mesh.nF)
print("- Dimensions of operator       :", str(mesh.nF), "x", str(mesh.nF))
print("- Number non-zero (isotropic)  :", str(Mf.nnz), "\n")


#############################################################
# Inverse of the Inner Product Matrix
# -----------------------------------
#
# You may need to compute the inverse of the inner product matrix for 
# constitutive relationships. Here we show how to call this
# using the *invert_matrix* keyword argument.
# For the isotropic case, the inner product matrix is diagonal.
# As a result, its inverse can be easily formed. 
#
# We validate the accuracy for the inverse of the inner product matrix
# for edges and faces by computing the following L2-norm for each:
#
# .. math::
#     \| \mathbf{u - M^{-1} M u} \|^2
#
# which we expect to be small.
#

# Create a small 3D mesh
h = np.ones(5)
mesh = TensorMesh([h, h, h])

# Define the constitutive relationship for each cell
sig = np.random.rand(mesh.nC)

# Inner product and inverse at edges
Me = mesh.get_edge_inner_product(sig)
Me_inv = mesh.get_edge_inner_product(sig, invert_matrix=True)

# Inner product and inverse at faces
Mf = mesh.get_face_inner_product(sig)
Mf_inv = mesh.get_face_inner_product(sig, invert_matrix=True)

# Generate some random vectors
vec_e = np.random.rand(mesh.nE)
vec_f = np.random.rand(mesh.nF)

# Compute norms
norm_e = np.linalg.norm(vec_e - Me_inv * Me * vec_e)
norm_f = np.linalg.norm(vec_f - Mf_inv * Mf * vec_f)

# Verify accuracy
print("ACCURACY")
print("Norm for edges:  ", norm_e)
print("Norm for faces:  ", norm_f)
