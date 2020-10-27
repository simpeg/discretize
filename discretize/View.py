from discretize.utils.code_utils import deprecate_module

deprecate_module(
    "discretize.View", "discretize.mixins.mpl_mod", removal_version="1.0.0"
)
try:
    from discretize.mixins.mpl_mod import Slicer
except ImportError:
    pass
