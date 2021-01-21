Solving PDEs with Discretize
============================

Here we show how the *discretize* package can be used to solve partial differential
equations (PDEs) numerically by employing the finite volume method. In each tutorial,
we demonstrate the following steps for a given PDE:

	1. Formulating the problem; i.e. the PDE and its boundary conditions
	2. Taking the inner product of each differential expression
	3. Approximating the inner products as discrete expressions according to the finite volume method
	4. Reducing the set of discrete expressions to a solvable linear system
	5. Implementing the *discretize* package to construct the necessary mesh, operators, matrices, etc... and solve the system
