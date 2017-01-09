Why discretize?
===============

Inverse problems are common across the geosciences: imaging in geophysics,
history matching, parameter estimation, and many of these require constrained
optimization using partial differential equations (PDEs) where the derivative
of mesh variables are sought. Finite difference, finite element and finite
volume techniques allow subdivision of continuous differential equations into
discrete domains. The knowledge and appropriate application of these methods
is fundamental to simulating physical processes. Many inverse problems in the
geosciences are solved using stochastic techniques or external finite
difference based tools (e.g. PEST); these are robust to
local minima and the programmatic implementation, respectively, however these
methods do not scale to millions of parameters to be estimated. This sort of
scale is necessary for solving many of the inverse problems in geophysics and
increasingly hydrogeology (e.g. electromagnetics, gravity, and fluid flow
problems).

In the context of the inverse problem, when the physical properties, the
domain, and the boundary conditions are not necessarily known, the simplicity
and efficiency in mesh generation are important criteria. Complex mesh
geometries, such as body fitted grids, commonly used when the domain is
explicitly given, are less appropriate. Additionally, when considering the
inverse problem, it is important that operators and their derivatives are
accessible to interrogation and extension. The goal of this work is to
provide a high level background to finite volume techniques abstracted across
four mesh types:

    1) tensor product mesh  :class:`discretize.TensorMesh`
    2) cylindrically symmetric mesh :class:`discretize.CylMesh`
    3) curvilinear mesh :class:`discretize.CurvilinearMesh`
    4) octree and quadtree meshes :class:`discretize.TreeMesh`

:code:`discretize` contributes an overview of finite volume techniques in the
context of geoscience inverse problems, which are treated in a consistent way
across various mesh types, highlighting similarities and differences.

.. include:: ../../CITATION.rst

Authors
-------

.. include:: ../../AUTHORS.rst

License
-------

.. include:: ../../LICENSE
