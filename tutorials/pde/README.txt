Solving PDEs
============

Here we show how the *discretize* package can be used to solve partial differential
equations (PDE) numerically by employing the finite volume method. To solve a PDE
numerically we must complete the following steps:

	1. Formulate the problem; e.g. the PDE and its boundary conditions
	2. Re-formulate the problem using the weak formulation
	3. Evaluate the inner products according to the finite volume method
	4. Construct and solve the final discretized system

Some of the lessons from the following tutorials include:

	- Setting Dirichlet and Neumann boundary conditions
	- Discretization for time-dependent PDEs
