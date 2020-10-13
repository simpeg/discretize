.. _api:

API Reference
=============

.. automodule:: discretize
    :show-inheritance:

.. currentmodule:: discretize


Meshes
------

.. currentmodule:: discretize

.. autosummary::
    :toctree: generated

    TensorMesh
    CylindricalMesh
    TreeMesh
    tree_mesh.TreeCell
    CurvilinearMesh



Numerical Operators
-------------------

.. automodule:: discretize.operators.InnerProducts

.. currentmodule:: discretize

.. autosummary::
    :toctree: generated

    operators.DiffOperators
    operators.InnerProducts


Mesh IO
-------

.. autosummary::
    :toctree: generated

    load_mesh
    base.mesh_io.TensorMeshIO
    base.mesh_io.TreeMeshIO


Visualization
-------------

.. autosummary::
    :toctree: generated

    mixins.mpl_mod.InterfaceMPL
    mixins.mpl_mod.Slicer
    mixins.vtk_mod.InterfaceVTK



Testing
-------

.. autosummary::
    :toctree: generated

    tests.OrderTest
    tests.checkDerivative
    tests.getQuadratic
    tests.Rosenbrock


Utilities
---------

.. automodule:: discretize.utils

.. currentmodule:: discretize

General Utilities
*****************

.. autosummary::
    :toctree: generated

    utils.download

Interpolation Operations
************************

.. autosummary::
    :toctree: generated

    utils.interpolation_matrix
    utils.interpmat
    utils.volume_average


Mesh Utilities
**************

.. autosummary::
    :toctree: generated

    utils.exampleLrmGrid
    utils.meshTensor
    utils.closestPoints
    utils.ExtractCoreMesh
    utils.random_model
    utils.mesh_builder_xyz
    utils.refine_tree_xyz
    utils.active_from_xyz


Matrix Utilities
****************

.. autosummary::
    :toctree: generated

    utils.mkvc
    utils.sdiag
    utils.sdInv
    utils.speye
    utils.kron3
    utils.spzeros
    utils.ddx
    utils.av
    utils.av_extrap
    utils.ndgrid
    utils.ind2sub
    utils.sub2ind
    utils.getSubArray
    utils.inv3X3BlockDiagonal
    utils.inv2X2BlockDiagonal
    utils.TensorType
    utils.makePropertyTensor
    utils.invPropertyTensor
    utils.Zero
    utils.Identity


Coordinate Utilities
********************

.. autosummary::
    :toctree: generated

    utils.cylindrical_to_cartesian
    utils.cyl2cart
    utils.cartesian_to_cylindrical
    utils.cart2cyl
    utils.rotate_points_from_normals
    utils.rotatePointsFromNormals
    utils.rotation_matrix_from_normals
    utils.rotationMatrixFromNormals
    utils.isScalar
    utils.asArray_N_x_Dim



Curvilinear Mesh Utilities
**************************

.. autosummary::
    :toctree: generated

    utils.volTetra
    utils.faceInfo
    utils.indexCube



Base Mesh
---------

.. automodule:: discretize.base

.. currentmodule:: discretize

.. autosummary::
    :toctree: generated

    base.BaseMesh
    base.BaseRectangularMesh
    base.BaseTensorMesh
