"""
==================================
Base Mesh (:mod:`discretize.base`)
==================================
.. currentmodule:: discretize.base

The ``base`` sub-package houses the fundamental classes for all meshes in ``discretize``.

Base Mesh Class
---------------
.. autosummary::
  :toctree: generated/

  BaseMesh
  BaseRegularMesh
  BaseRectangularMesh
  BaseTensorMesh
"""
from discretize.base.base_mesh import BaseMesh
from discretize.base.base_regular_mesh import BaseRegularMesh, BaseRectangularMesh
from discretize.base.base_tensor_mesh import BaseTensorMesh
