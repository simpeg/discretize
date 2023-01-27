"""
==================================
Mixins (:mod:`discretize.mixins`)
==================================
.. currentmodule:: discretize.mixins

The ``mixins`` module provides a set of tools for interfacing ``discretize``
with external libraries such as VTK, OMF, and matplotlib. These modules are only
imported if those external packages are available in the active Python environment and
provide extra functionality that different finite volume meshes can inherit.

Mixin Classes
-------------
.. autosummary::
  :toctree: generated/

  TensorMeshIO
  TreeMeshIO
  InterfaceMPL
  InterfaceVTK
  InterfaceOMF

Other Optional Classes
----------------------
.. autosummary::
  :toctree: generated/

  Slicer
"""
import importlib.util
from .mesh_io import TensorMeshIO, TreeMeshIO, SimplexMeshIO

AVAILABLE_MIXIN_CLASSES = []
SIMPLEX_MIXIN_CLASSES = []

if importlib.util.find_spec("vtk"):
    from .vtk_mod import InterfaceVTK

    AVAILABLE_MIXIN_CLASSES.append(InterfaceVTK)

if importlib.util.find_spec("omf"):
    from .omf_mod import InterfaceOMF

    AVAILABLE_MIXIN_CLASSES.append(InterfaceOMF)

# keep this one last in defaults in case anything else wants to overwrite
# plot commands
if importlib.util.find_spec("matplotlib"):
    from .mpl_mod import Slicer, InterfaceMPL

    AVAILABLE_MIXIN_CLASSES.append(InterfaceMPL)


# # Python 3 friendly
class InterfaceMixins(*AVAILABLE_MIXIN_CLASSES):
    """Class to handle all the avaialble mixins that can be inherrited
    directly onto ``discretize.base.BaseMesh``
    """

    pass
