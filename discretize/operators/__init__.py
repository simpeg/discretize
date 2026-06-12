"""
================================================
Discrete Operators (:mod:`discretize.operators`)
================================================
.. currentmodule:: discretize.operators

The ``operators`` package contains the classes discretize meshes
use to construct discrete versions of the differential operators.

Operator Classes
----------------
.. autosummary::
  :toctree: generated/

  DiffOperators
  InnerProducts
  UnstructuredInnerProducts
"""

from discretize.operators.differential_operators import DiffOperators
from discretize.operators.inner_products import InnerProducts, UnstructuredInnerProducts
