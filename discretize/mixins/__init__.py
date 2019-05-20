"""
The ``mixins`` module provides a set of tools for interfacing ``discretize``
with external libraries such as VTK and OMF. These modules are only imported if
those external packages are available in the active Python environment and
provide extra functionality that different finite volume meshes can inherrit.
"""

AVAILABLE_MIXIN_CLASSES = []

try:
    from .vtkModule import InterfaceVTK, InterfaceTensorread_vtk
    AVAILABLE_MIXIN_CLASSES.append(InterfaceVTK)
except ImportError as err:
    pass

try:
    from .omfModule import InterfaceOMF
    AVAILABLE_MIXIN_CLASSES.append(InterfaceOMF)
except ImportError as err:
    pass


# NOTE: this is what we need to use when Python 2 support is dropped
#       This throws a syntax error on Python 2
# # Python 3 friendly
# class InterfaceMixins(*AVAILABLE_MIXIN_CLASSES):
#     """This class handles all the avaialble mixins that can be inherrited
#     directly onto ``discretize.BaseMesh``
#     """
#     pass

# Python 2 is tedious
# NOTE: this will have to be updated anytime a mixin is added
# NOTE: this should be deleted when Python 2 support is dropped
if len(AVAILABLE_MIXIN_CLASSES) == 0:
    class InterfaceMixins():
        """This class handles all the avaialble mixins that can be inherrited
        directly onto ``discretize.BaseMesh``
        """
        pass
elif len(AVAILABLE_MIXIN_CLASSES) == 1:
    class InterfaceMixins(AVAILABLE_MIXIN_CLASSES[0]):
        """This class handles all the avaialble mixins that can be inherrited
        directly onto ``discretize.BaseMesh``
        """
        pass
elif len(AVAILABLE_MIXIN_CLASSES) == 2:
    class InterfaceMixins(AVAILABLE_MIXIN_CLASSES[0],
                          AVAILABLE_MIXIN_CLASSES[1]):
        """This class handles all the avaialble mixins that can be inherrited
        directly onto ``discretize.BaseMesh``
        """
        pass
