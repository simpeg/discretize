"""
The ``mixins`` module provides a set of tools for interfacing ``discretize``
with external libraries such as VTK and OMF. These modules are only imported if
those external packages are available in the active Python environment and
provide extra functionality that different finite volume meshes can inherrit.
"""

AVAILABLE_MIXIN_CLASSES = []

try:
    from .vtkModule import vtkInterface, vtkTensorRead
    AVAILABLE_MIXIN_CLASSES.append(vtkInterface)
except ImportError as err:
    pass

try:
    from .omfModule import omfInterface
    AVAILABLE_MIXIN_CLASSES.append(omfInterface)
except ImportError as err:
    pass


class InterfaceMixins(*AVAILABLE_MIXIN_CLASSES):
    """This class handles all the avaialble mixins that can be inherrited
    directly onto ``discretize.BaseMesh``
    """
    pass
