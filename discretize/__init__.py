"""
=====================================
Discretize Meshes (:mod:`discretize`)
=====================================
.. currentmodule:: discretize

The ``discretize`` package contains four types of meshes for soliving partial differential
equations using the finite volume method.

Mesh Classes
============
.. autosummary::
  :toctree: generated/

  TensorMesh
  CylindricalMesh
  CurvilinearMesh
  TreeMesh

Tree Mesh Cells
===============
The :class:`~discretize.tree_mesh.TreeCell` class was designed specificialy to define the cells within tree meshes.
Instances of :class:`~discretize.tree_mesh.TreeCell` are not meant to be created on there own.
However, they can be returned directly by indexing a particular cell within a tree mesh.

.. autosummary::
  :toctree: generated/

  tree_mesh.TreeCell
"""

from discretize.tensor_mesh import TensorMesh
from discretize.cylindrical_mesh import CylMesh, CylindricalMesh
from discretize.curvilinear_mesh import CurvilinearMesh
from discretize.utils.io_utils import load_mesh

try:
    from discretize.tree_mesh import TreeMesh
except ImportError as err:
    print(err)
    import os

    # Check if being called from non-standard location (i.e. a git repository)
    # is tree_ext.pyx here? will not be in the folder if installed to site-packages...
    file_test = os.path.dirname(os.path.abspath(__file__)) + "/_extensions/tree_ext.pyx"
    if os.path.isfile(file_test):
        # Then we are being run from a repository
        raise ImportError(
            """
            Unable to import TreeMesh.

            It would appear that discretize is being imported from its repository.
            If this is intentional, you need to run:

            python setup.py build_ext --inplace

            to compile the cython code.
            """
        )
from discretize import tests

__version__ = "0.7.1"
__author__ = "SimPEG Team"
__license__ = "MIT"
__copyright__ = "2013, SimPEG Developers, http://simpeg.xyz"
