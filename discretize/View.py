from .utils.code_utils import deprecate_module
deprecate_module("View", "mixins.mpl_mod", removal_version="1.0.0")
try:
    from .mixins.mpl_mod import Slicer
except ImportError:
    pass
