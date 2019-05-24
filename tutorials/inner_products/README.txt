Inner Products
==============

Numerical solutions to differential equations frequently make use
of the **weak formulation**. That is, we take the inner product
of our PDE with some test function. There are many different ways to
evaluate inner products numerically; we could approximate the integral
using trapezoidal, midpoint, or higher-order approximations. A simple
method for evaluating inner products on a numerical grid is to midpoint
rule.
