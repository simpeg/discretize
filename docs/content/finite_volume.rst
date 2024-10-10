.. _api_FiniteVolume:

Finite Volume
*************

Any numerical implementation requires the discretization of continuous
functions into discrete approximations. These approximations are typically
organized in a mesh, which defines boundaries, locations, and connectivity. Of
specific interest to geophysical simulations, we require that averaging,
interpolation and differential operators be defined for any mesh. In SimPEG,
we have implemented a staggered mimetic finite volume approach (`Hyman and
Shashkov, 1999 <https://doi.org/10.1006/jcph.1999.6225>`_). This
approach requires the definitions of variables at either cell-centers, nodes,
faces, or edges as seen in the figure below.

.. image:: ../images/finitevolrealestate.png
   :width: 400 px
   :alt: FiniteVolume
   :align: center
