"""
Tree Meshes
===========

Compared to tensor meshes, tree meshes are able to provide higher levels
of discretization in certain regions while reducing the total number of
cells. Tree meshes belong to the class (:class:`~discretize.TreeMesh`).
Tree meshes can be defined in 2 or 3 dimensions. Here we demonstrate:

    - How to create basic tree meshes in 2D and 3D
    - Strategies for local mesh refinement
    - How to plot tree meshes
    - How to extract properties from tree meshes

To create a tree mesh, we first define the base tensor mesh (a mesh
comprised entirely of the smallest cells). Next we choose the level of
discretization around certain points or within certain regions. When
creating tree meshes, we must remember certain rules:

    - The number of base mesh cells in x, y and z must all be powers of 2
    - We cannot refine the mesh to create cells smaller than those defining the
    base mesh
    - The range of cell sizes in the tree mesh depends on the number of base
    mesh cells in x, y and z


"""

###############################################
#
# Import Packages
# ---------------
#
# Here we import the packages required for this tutorial.
#

from discretize import TreeMesh
from discretize.utils import matutils, meshutils
import numpy as np

# sphinx_gallery_thumbnail_number = 3

###############################################
# Basic Example
# -------------
#
# Here we demonstrate the basic two step process for creating a 2D tree mesh
# (QuadTree mesh). The region of highest discretization if defined within a
# rectangular box.
#

dh = 5    # minimum cell width (base mesh cell width)
nbc = 64  # number of base mesh cells in x

# Define base mesh
h = dh*np.ones(nbc)
mesh = TreeMesh([h, h])

xp, yp = np.meshgrid([120., 240.], [80., 160.])
xy = np.c_[matutils.mkvc(xp), matutils.mkvc(yp)]

# Discretize to finest cell size within rectangular region
mesh2 = meshutils.refine_tree_xyz(mesh, xy, octree_levels=[2, 2],
                                 method='box', finalize=False)

mesh2.finalize()  # Must finalize tree mesh before use

mesh.plotGrid(showIt=True)


###############################################
# Intermediate Example
# --------------------
#
# The widths of the base mesh cells do not need to be the same in x and y but
# the number of base mesh cells in x and y does need to be a power of 2. Here
# we show a method for mesh refinement which discretizes more finely around a
# set of points. This is usedful for discretizing surface topography.

dx = 5    # minimum cell width (base mesh cell width) in x
dy = 5    # minimum cell width (base mesh cell width) in y

x_length = 300.    # domain width in x
y_length = 300.    # domain width in y

# Compute number of base mesh cells required in x and y
nbcx = 2**int(np.round(np.log(x_length/dx)/np.log(2.)))
nbcy = 2**int(np.round(np.log(y_length/dy)/np.log(2.)))

# Define the base mesh
hx = [(dx, nbcx)]
hy = [(dy, nbcy)]
mesh = TreeMesh([hx, hy], x0='CC')

# Refine surface topography
xx = mesh.vectorNx
yy = -3*np.exp((xx**2) / 100**2) + 50.
pts = np.c_[matutils.mkvc(xx), matutils.mkvc(yy)]
mesh = meshutils.refine_tree_xyz(mesh, pts, octree_levels=[2,2],
                                  method='radial', finalize=False)

# Refine polygon
xx = np.array([0., 10., 0., -10.])
yy = np.array([-20., -10., 0., -10])
pts = np.c_[matutils.mkvc(xx), matutils.mkvc(yy)]
mesh = meshutils.refine_tree_xyz(mesh, pts, octree_levels=[2, 2],
                                  method='radial', finalize=False)

mesh.finalize()
mesh.plotGrid(showIt=True)

####################################################
# Extracting Mesh Properties
# --------------------------
#
# Once the mesh is created, you may want to extract certain properties. Here,
# we show some properties that can be extracted from a QuadTree mesh.

dx = 5    # minimum cell width (base mesh cell width) in x
dy = 5    # minimum cell width (base mesh cell width) in y

x_length = 300.    # domain width in x
y_length = 300.     # domain width in y

# Compute number of base mesh cells required in x and y
nbcx = 2**int(np.round(np.log(x_length/dx)/np.log(2.)))
nbcy = 2**int(np.round(np.log(y_length/dy)/np.log(2.)))

# Define the base mesh
hx = [(dx, nbcx)]
hy = [(dy, nbcy)]
mesh = TreeMesh([hx, hy], x0='CC')

# Refine surface topography
xx = mesh.vectorNx
yy = -3*np.exp((xx**2) / 100**2) + 50.
pts = np.c_[matutils.mkvc(xx), matutils.mkvc(yy)]
mesh = meshutils.refine_tree_xyz(mesh, pts, octree_levels=[2,2],
                                 method='radial', finalize=False)

# Refine polygon
xx = np.array([0., 10., 0., -10.])
yy = np.array([-20., -10., 0., -10])
pts = np.c_[matutils.mkvc(xx), matutils.mkvc(yy)]
mesh = meshutils.refine_tree_xyz(mesh, pts, octree_levels=[2, 2],
                                 method='radial', finalize=False)

mesh.finalize()

# The bottom west corner
x0 = mesh.x0

# The total number of cells
nC = mesh.nC

# An (nC, 2) array containing the cell-center locations
cc = mesh.gridCC

# A boolean array specifying which cells lie on the boundary
bInd = mesh.cellBoundaryInd

# The cell areas (2D "volume")
s = mesh.vol
mesh.plotImage(s, grid=True)


###############################################
# 3D Example
# ----------
#
# Here we show how the same approach can be used to create and extract
# properties from a 3D tensor mesh.

dx = 5    # minimum cell width (base mesh cell width) in x
dy = 5    # minimum cell width (base mesh cell width) in y
dz = 5    # minimum cell width (base mesh cell width) in z

x_length = 300.     # domain width in x
y_length = 300.     # domain width in y
z_length = 300.     # domain width in y

# Compute number of base mesh cells required in x and y
nbcx = 2**int(np.round(np.log(x_length/dx)/np.log(2.)))
nbcy = 2**int(np.round(np.log(y_length/dy)/np.log(2.)))
nbcz = 2**int(np.round(np.log(z_length/dz)/np.log(2.)))

# Define the base mesh
hx = [(dx, nbcx)]
hy = [(dy, nbcy)]
hz = [(dz, nbcz)]
mesh = TreeMesh([hx, hy, hz], x0='CCC')

# Refine surface topography
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = -3*np.exp((xx**2 + yy**2) / 100**2) + 50.
pts = np.c_[matutils.mkvc(xx), matutils.mkvc(yy), matutils.mkvc(zz)]
mesh = meshutils.refine_tree_xyz(mesh, pts, octree_levels=[2, 2],
                                 method='radial', finalize=False)

# Refine box
xp, yp, zp = np.meshgrid([-40., 40.], [-40., 40.], [-60., 0.])
xyz = np.c_[matutils.mkvc(xp), matutils.mkvc(yp), matutils.mkvc(zp)]

mesh = meshutils.refine_tree_xyz(mesh, xyz, octree_levels=[2, 2],
                                 method='box', finalize=False)

mesh.finalize()

# The bottom west corner
x0 = mesh.x0

# The total number of cells
nC = mesh.nC

# An (nC, 2) array containing the cell-center locations
cc = mesh.gridCC

# A boolean array specifying which cells lie on the boundary
bInd = mesh.cellBoundaryInd

# The cell areas (2D "volume")
s = mesh.vol
mesh.plotSlice(s, normal='Y', ind=int(mesh.hy.size/2), grid=True)
