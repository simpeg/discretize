.. _operators_averaging:

Interpolation and Averaging
***************************

Here, we provide the background theory for how discrete interpolation and averaging matricies are formed.
Interpolation is required when a discrete quantity is known on the mesh (centers, nodes, edges or faces), but we would like to estimate its
value at a point within the continuous domain. Averaging is required when quantities that live on different
parts of the mesh need to be multiplied.

**Tutorials:** 

    - :ref:`tutorial for constructing and applying averaging operators <sphx_glr_tutorials_operators_1_averaging.py>`



Interpolation
=============

Presently there is an extensive set of interpolation methods (e.g. polynomial, spline, piecewise constant).
One of the most effective and widely used interpolation methods is linear interpolation.
The *discretize* package primarily uses linear interpolation because 1) it is very fast, and 2) higher order
interpolation methods require the construction of matricies which are less sparse.
Here, we discuss how a sparse matrix can be formed which interpolates the discrete values to
a set of points in continuous space.

The formulation for linear interpolation is adequately presented on Wikipedia, see:

	- `Linear Interpolation (1D) <https://en.wikipedia.org/wiki/Linear_interpolation>`__
	- `Bilinear Interpolation (2D) <https://en.wikipedia.org/wiki/Bilinear_interpolation>`__
	- `Trilinear Interpolation (3D) <https://en.wikipedia.org/wiki/Trilinear_interpolation>`__

Interpolation Matrix in 1D
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Basics for constructing the 1D interpolation matrix**



Interpolation Matrix in 3D
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Basics for constructing the 1D interpolation matrix**



Scalars vs. Vectors
^^^^^^^^^^^^^^^^^^^

**Describe how components are treated separately**


Averaging
=========





