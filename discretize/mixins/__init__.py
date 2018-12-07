"""
The ``mixins`` module provides a set of tools for interfacing ``discretize``
with external libraries such as VTK. These modules are only imported if those
external packages are available in the active Python environment and provide
extra functionality that different finite volume meshes can inherrit.
"""

try:
    from .vtkModule import vtkInterface, vtkTensorRead
except ImportError as err:
    pass
