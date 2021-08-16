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
from .mesh_io import TensorMeshIO, TreeMeshIO

AVAILABLE_MIXIN_CLASSES = []

try:
    from .vtk_mod import InterfaceVTK

    AVAILABLE_MIXIN_CLASSES.append(InterfaceVTK)
except ImportError as err:
    pass

try:
    from .omf_mod import InterfaceOMF

    AVAILABLE_MIXIN_CLASSES.append(InterfaceOMF)
except ImportError as err:
    pass

# keep this one last in defaults in case anything else wants to overwrite
# plot commands
try:
    from .mpl_mod import Slicer, InterfaceMPL

    AVAILABLE_MIXIN_CLASSES.append(InterfaceMPL)
except ImportError as err:
    pass

# # Python 3 friendly
class InterfaceMixins(*AVAILABLE_MIXIN_CLASSES):
    """This class handles all the avaialble mixins that can be inherrited
    directly onto ``discretize.base.BaseMesh``
    """

    pass
