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
    CurvilinearMesh
    TreeMesh
    tree_mesh.TreeCell


Numerical Operators
-------------------

.. automodule:: discretize.operators

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
    tests.check_derivative
    tests.get_quadratic
    tests.rosenbrock


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
    utils.volume_average


Mesh Utilities
**************

.. autosummary::
    :toctree: generated

    utils.example_curvilinear_grid
    utils.unpack_widths
    utils.closest_points_index
    utils.extract_core_mesh
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
    utils.sdinv
    utils.speye
    utils.kron3
    utils.spzeros
    utils.ddx
    utils.av
    utils.av_extrap
    utils.ndgrid
    utils.ind2sub
    utils.sub2ind
    utils.get_subarray
    utils.inverse_3x3_block_diagonal
    utils.inverse_2x2_block_diagonal
    utils.make_property_tensor
    utils.inverse_property_tensor
    utils.TensorType
    utils.Zero
    utils.Identity


Mathematical Operations
***********************

.. autosummary::
    :toctree: generated

    utils.rotate_points_from_normals
    utils.rotation_matrix_from_normals
    utils.cylindrical_to_cartesian
    utils.cartesian_to_cylindrical
    utils.is_scalar
    utils.as_array_n_by_dim
    utils.cyl2cart
    utils.cart2cyl



Curvilinear Mesh Utilities
**************************

.. autosummary::
    :toctree: generated

    utils.volume_tetrahedron
    utils.face_info
    utils.index_cube



Base Mesh
---------

.. automodule:: discretize.base

.. currentmodule:: discretize

.. autosummary::
    :toctree: generated

    base.BaseMesh
    base.BaseRectangularMesh
    base.BaseTensorMesh
