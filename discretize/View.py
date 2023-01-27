"""Deprecated view module."""
from discretize.utils.code_utils import deprecate_module

deprecate_module(
    "discretize.View",
    "discretize.mixins.mpl_mod",
    removal_version="1.0.0",
    future_warn=True,
)
try:
    from discretize.mixins.mpl_mod import Slicer  # NOQA F401
except ImportError:
    pass
