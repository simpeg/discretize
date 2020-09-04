Inner Products
==============

Numerical solutions to differential equations frequently make use
of the **weak formulation**. That is, we take the inner product
of each PDE with some test function. There are many different ways to
evaluate inner products numerically; i.e. trapezoidal rule, midpoint
rule, or higher-order approximations. A simple method for evaluating
inner products on a numerical grid is to apply the midpoint rule; 
which is used by the *discretize* package.

Here, we demonstrate how to approximate various classes of inner
products numerically. If this is known, the user will be capable
of properly discretizing any term in a problem specific PDE.