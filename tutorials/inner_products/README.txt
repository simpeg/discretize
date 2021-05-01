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
products numerically. The inner products can be approximated in
terms of a linear expression which contains an inner-product matrix.
By learning how to formulate each class of inner products, the user will 
have the building blocks to discretize and solve a problem-specific PDE.