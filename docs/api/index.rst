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
    CylMesh
    TreeMesh
    TreeMesh.TreeCell
    CurvilinearMesh



Numerical Operators
-------------------

.. automodule:: discretize.InnerProducts

.. currentmodule:: discretize

.. autosummary::
    :toctree: generated

    DiffOperators.DiffOperators
    InnerProducts.InnerProducts


Mesh IO
-------

.. autosummary::
    :toctree: generated

    load_mesh
    MeshIO.TensorMeshIO
    MeshIO.TreeMeshIO


Visualization
-------------

.. autosummary::
    :toctree: generated

    View.TensorView
    View.CylView
    View.CurviView
    View.Slicer
    mixins.vtkModule



Testing
-------

.. autosummary::
    :toctree: generated

    Tests.OrderTest
    Tests.checkDerivative
    Tests.getQuadratic
    Tests.Rosenbrock


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


Mathematical Operations
***********************

.. autosummary::
    :toctree: generated

    utils.rotatePointsFromNormals
    utils.rotationMatrixFromNormals
    utils.cyl2cart
    utils.cart2cyl
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
