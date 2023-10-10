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
  SimplexMesh

Mesh Cells
==========
The :class:`~discretize.tensor_cell.TensorCell` and
:class:`~discretize.tree_mesh.TreeCell` classes were designed specifically to
define the cells within tensor and tree meshes, respectively.
Instances of :class:`~discretize.tree_mesh.TreeCell` and
:class:`~discretize.tensor_cell.TensorCell` are not meant to be created on
their own.
However, they can be returned directly by indexing a particular cell within
a tensor or tree mesh.

.. autosummary::
  :toctree: generated/

  tensor_cell.TensorCell
  tree_mesh.TreeCell
"""

from discretize.tensor_mesh import TensorMesh
from discretize.cylindrical_mesh import CylMesh, CylindricalMesh
from discretize.curvilinear_mesh import CurvilinearMesh
from discretize.unstructured_mesh import SimplexMesh
from discretize.utils.io_utils import load_mesh
from .tensor_cell import TensorCell

try:
    from discretize.tree_mesh import TreeMesh
except ImportError as err:
    import os

    # Check if being called from non-standard location (i.e. a git repository)
    # is tree_ext.pyx here? will not be in the folder if installed to site-packages...
    file_test = os.path.dirname(os.path.abspath(__file__)) + "/_extensions/tree_ext.pyx"
    if os.path.isfile(file_test):
        # Then we are being run from a repository
        raise ImportError(
            """
            It would appear that discretize is being imported from its source code
            directory and is unable to load its compiled extension modules. Try changing
            your directory and re-launching your python interpreter.

            If this was intentional, you need to install discretize in an editable mode.
            """
        )
    else:
        raise err
from discretize import tests

__author__ = "SimPEG Team"
__license__ = "MIT"
__copyright__ = "2013 - 2023, SimPEG Developers, https://simpeg.xyz"

from importlib.metadata import version, PackageNotFoundError

# Version
try:
    # - Released versions just tags:       0.8.0
    # - GitHub commits add .dev#+hash:     0.8.1.dev4+g2785721
    # - Uncommitted changes add timestamp: 0.8.1.dev4+g2785721.d20191022
    __version__ = version("discretize")
except PackageNotFoundError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. discretize should be
    # installed properly!
    from datetime import datetime

    __version__ = "unknown-" + datetime.today().strftime("%Y%m%d")
