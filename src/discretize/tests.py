"""
===========================================
Testing Utilities (:mod:`discretize.tests`)
===========================================
.. currentmodule:: discretize.tests

This module contains utilities for convergence testing.

Classes
-------
.. autosummary::
  :toctree: generated/

  OrderTest

Functions
---------
.. autosummary::
  :toctree: generated/

  check_derivative
  rosenbrock
  get_quadratic
  setup_mesh
  assert_isadjoint
"""  # NOQA D205

import numpy as np
import scipy.sparse as sp

from discretize.utils import mkvc, example_curvilinear_grid, requires
from discretize.tensor_mesh import TensorMesh
from discretize.curvilinear_mesh import CurvilinearMesh
from discretize.cylindrical_mesh import CylindricalMesh
from discretize.utils.code_utils import deprecate_function

from . import TreeMesh as Tree

import unittest
import inspect

try:
    import getpass

    name = getpass.getuser()[0].upper() + getpass.getuser()[1:]
except Exception:
    name = "You"

happiness = [
    "The test be workin!",
    "You get a gold star!",
    "Yay passed!",
    "Happy little convergence test!",
    "That was easy!",
    "Testing is important.",
    "You are awesome.",
    "Go Test Go!",
    "Once upon a time, a happy little test passed.",
    "And then everyone was happy.",
    "Not just a pretty face " + name,
    "You deserve a pat on the back!",
    "Well done " + name + "!",
    "Awesome, " + name + ", just awesome.",
]
sadness = [
    "No gold star for you.",
    "Try again soon.",
    "Thankfully,  persistence is a great substitute for talent.",
    "It might be easier to call this a feature...",
    "Coffee break?",
    "Boooooooo  :(",
    "Testing is important. Do it again.",
    "Did you put your clever trousers on today?",
    "Just think about a dancing dinosaur and life will get better!",
    "You had so much promise " + name + ", oh well...",
    name.upper() + " ERROR!",
    "Get on it " + name + "!",
    "You break it, you fix it.",
]

# Initiate random number generator
rng = np.random.default_rng()


def setup_mesh(mesh_type, nC, nDim):
    """Generate arbitrary mesh for testing.

    For the mesh type, number of cells along each axis and dimension specified,
    **setup_mesh** will construct a random mesh that can be used for testing.
    By design, the domain width is 1 along each axis direction.

    Parameters
    ----------
    mesh_type : str
        Defines the mesh type. Must be one of **{'uniformTensorMesh',
        'randomTensorMesh', 'uniformCylindricalMesh', 'randomCylindricalMesh',
        'uniformTree', 'randomTree', 'uniformCurv', 'rotateCurv', 'sphereCurv'}**
    nC : int
        Number of cells along each axis. If *mesh_type* is 'Tree', then *nC* defines the
        number of base mesh cells and must be a power of 2.
    nDim : int
        The dimension of the mesh. Must be 1, 2 or 3.

    Returns
    -------
    discretize.base.BaseMesh
        A discretize mesh of class specified by the input argument *mesh_type*
    """
    if "TensorMesh" in mesh_type:
        if "uniform" in mesh_type:
            h = [nC, nC, nC]
        elif "random" in mesh_type:
            h1 = np.random.rand(nC) * nC * 0.5 + nC * 0.5
            h2 = np.random.rand(nC) * nC * 0.5 + nC * 0.5
            h3 = np.random.rand(nC) * nC * 0.5 + nC * 0.5
            h = [hi / np.sum(hi) for hi in [h1, h2, h3]]  # normalize
        else:
            raise Exception("Unexpected mesh_type")

        mesh = TensorMesh(h[:nDim])
        max_h = max([np.max(hi) for hi in mesh.h])

    elif "CylindricalMesh" in mesh_type or "CylMesh" in mesh_type:
        if "uniform" in mesh_type:
            if "symmetric" in mesh_type:
                h = [nC, 1, nC]
            else:
                h = [nC, nC, nC]
        elif "random" in mesh_type:
            h1 = np.random.rand(nC) * nC * 0.5 + nC * 0.5
            if "symmetric" in mesh_type:
                h2 = [
                    2 * np.pi,
                ]
            else:
                h2 = np.random.rand(nC) * nC * 0.5 + nC * 0.5
            h3 = np.random.rand(nC) * nC * 0.5 + nC * 0.5
            h = [hi / np.sum(hi) for hi in [h1, h2, h3]]  # normalize
            h[1] = h[1] * 2 * np.pi
        else:
            raise Exception("Unexpected mesh_type")

        if nDim == 2:
            mesh = CylindricalMesh([h[0], h[1]])
            if "symmetric" in mesh_type:
                max_h = np.max(mesh.h[0])
            else:
                max_h = max([np.max(hi) for hi in mesh.h])
        elif nDim == 3:
            mesh = CylindricalMesh(h)
            if "symmetric" in mesh_type:
                max_h = max([np.max(hi) for hi in [mesh.h[0], mesh.h[2]]])
            else:
                max_h = max([np.max(hi) for hi in mesh.h])

    elif "Curv" in mesh_type:
        if "uniform" in mesh_type:
            kwrd = "rect"
        elif "rotate" in mesh_type:
            kwrd = "rotate"
        elif "sphere" in mesh_type:
            kwrd = "sphere"
        else:
            raise Exception("Unexpected mesh_type")
        if nDim == 1:
            raise Exception("Lom not supported for 1D")
        elif nDim == 2:
            X, Y = example_curvilinear_grid([nC, nC], kwrd)
            mesh = CurvilinearMesh([X, Y])
        elif nDim == 3:
            X, Y, Z = example_curvilinear_grid([nC, nC, nC], kwrd)
            mesh = CurvilinearMesh([X, Y, Z])
        max_h = 1.0 / nC

    elif "Tree" in mesh_type:
        if Tree is None:
            raise Exception("Tree Mesh not installed. Run 'python setup.py install'")
        nC *= 2
        if "uniform" in mesh_type or "notatree" in mesh_type:
            h = [nC, nC, nC]
        elif "random" in mesh_type:
            h1 = np.random.rand(nC) * nC * 0.5 + nC * 0.5
            h2 = np.random.rand(nC) * nC * 0.5 + nC * 0.5
            h3 = np.random.rand(nC) * nC * 0.5 + nC * 0.5
            h = [hi / np.sum(hi) for hi in [h1, h2, h3]]  # normalize
        else:
            raise Exception("Unexpected mesh_type")

        levels = int(np.log(nC) / np.log(2))
        mesh = Tree(h[:nDim], levels=levels)

        def function(cell):
            if "notatree" in mesh_type:
                return levels - 1
            r = cell.center - 0.5
            dist = np.sqrt(r.dot(r))
            if dist < 0.2:
                return levels
            return levels - 1

        mesh.refine(function)
        # mesh.number()
        # mesh.plot_grid(show_it=True)
        max_h = max([np.max(hi) for hi in mesh.h])
    return mesh, max_h


class OrderTest(unittest.TestCase):
    r"""Base class for testing convergence of discrete operators with respect to cell size.

    ``OrderTest`` is a base class for testing the order of convergence of discrete
    operators with respect to cell size. ``OrderTest`` is inherited by the test
    class for the given operator. Within the test class, the user sets the parameters
    for the convergence testing and defines a method :py:attr:`~OrderTest.getError`
    which defines the error as a norm of the residual (see example).

    OrderTest inherits from :class:`unittest.TestCase`.

    Parameters
    ----------
    name : str
        Name the convergence test
    meshTypes : list of str
        List denoting the mesh types on which the convergence will be tested.
        List entries must of the list {'uniformTensorMesh', 'randomTensorMesh',
        'uniformCylindricalMesh', 'randomCylindricalMesh', 'uniformTree', 'randomTree',
        'uniformCurv', 'rotateCurv', 'sphereCurv'}
    expectedOrders : float or list of float (default = 2.0)
        Defines the expect orders of convergence for all meshes defined in *meshTypes*.
        If list, must have same length as argument *meshTypes*.
    tolerance : float or list of float (default = 0.85)
        Defines tolerance for numerical approximate of convergence order.
        If list, must have same length as argument *meshTypes*.
    meshSizes : list of int
        From coarsest to finest, defines the number of mesh cells in each axis direction
        for the meshes used in the convergence test; e.g. [4, 8, 16, 32]
    meshDimension : int
        Mesh dimension. Must be 1, 2 or 3

    Notes
    -----
    Consider an operator :math:`A(f)` that acts on a test function :math:`f`. And let
    :math:`A_h (f)` be the discrete approximation to the original operator constructed
    on a mesh will cell size :math:`h`. ``OrderTest`` assesses the convergence of

    .. math::
        error(h) = \| A_h(f) - A(f) \|

    as :math:`h \rightarrow 0`. Note that you can provide any norm to quantify the error.
    The convergence test is passed when the numerically estimated rate of convergence is within
    a specified tolerance of the expected convergence rate supplied by the user.

    Examples
    --------
    Here, we utilize the ``OrderTest`` class to validate the rate of convergence for
    the :py:attr:`~discretize.differential_operators.face_divergence`. Our convergence
    test is done for a uniform 2D tensor mesh. Under the test class *TestDIV2D*, we
    define the static parameters for the order test. We then define a method *getError*
    for this class which returns the norm of some residual. With these two pieces
    defined, we can call the order test as shown below.

    >>> from discretize.tests import OrderTest
    >>> import unittest
    >>> import numpy as np

    >>> class TestDIV2D(OrderTest):
    ...     # Static properties for OrderTest
    ...     name = "Face Divergence 2D"
    ...     meshTypes = ["uniformTensorMesh"]
    ...     meshDimension = 2
    ...     expectedOrders = 2.0
    ...     tolerance = 0.85
    ...     meshSizes = [8, 16, 32, 64]
    ...     def getError(self):
    ...         # Test function
    ...         fx = lambda x, y: np.sin(2 * np.pi * x)
    ...         fy = lambda x, y: np.sin(2 * np.pi * y)
    ...         # Analytic solution for operator acting on test function
    ...         sol = lambda x, y: 2 * np.pi * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    ...         # Evaluate test function on faces
    ...         f = np.r_[
    ...             fx(self.M.faces_x[:, 0], self.M.faces_x[:, 1]),
    ...             fy(self.M.faces_y[:, 0], self.M.faces_y[:, 1])
    ...         ]
    ...         # Analytic solution at cell centers
    ...         div_f = sol(self.M.cell_centers[:, 0], self.M.cell_centers[:, 1])
    ...         # Numerical approximation of divergence at cell centers
    ...         div_f_num = self.M.face_divergence * f
    ...         # Define the error function as a norm
    ...         err = np.linalg.norm((div_f_num - div_f), np.inf)
    ...         return err
    ...     def test_order(self):
    ...         self.orderTest()
    """

    name = "Order Test"
    expectedOrders = (
        2.0  # This can be a list of orders, must be the same length as meshTypes
    )
    tolerance = 0.85  # This can also be a list, must be the same length as meshTypes
    meshSizes = [4, 8, 16, 32]
    meshTypes = ["uniformTensorMesh"]
    _meshType = meshTypes[0]
    meshDimension = 3

    def setupMesh(self, nC):
        """Generate mesh and set as current mesh for testing.

        Parameters
        ----------
        nC : int
            Number of cells along each axis.

        Returns
        -------
        Float
            Maximum cell width for the mesh
        """
        mesh, max_h = setup_mesh(self._meshType, nC, self.meshDimension)
        self.M = mesh
        return max_h

    def getError(self):
        r"""Compute error defined as a norm of the residual.

        This method is overwritten within the test class of a particular operator.
        Within the method, we define a test function :math:`f`, the analytic solution
        of an operator :math:`A(f)` acting on the test function, and the numerical
        approximation obtained by applying the discretized operator :math:`A_h (f)`.
        **getError** is defined to return the norm of the residual as shown below:

        .. math::
            error(h) = \| A_h(f) - A(f) \|

        """
        return 1.0

    def orderTest(self):
        """Perform an order test.

        For number of cells specified in meshSizes setup mesh, call getError
        and prints mesh size, error, ratio between current and previous error,
        and estimated order of convergence.
        """
        __tracebackhide__ = True
        if not isinstance(self.meshTypes, list):
            raise TypeError("meshTypes must be a list")
        if type(self.tolerance) is not list:
            self.tolerance = np.ones(len(self.meshTypes)) * self.tolerance

        # if we just provide one expected order, repeat it for each mesh type
        if type(self.expectedOrders) == float or type(self.expectedOrders) == int:
            self.expectedOrders = [self.expectedOrders for i in self.meshTypes]
        try:
            self.expectedOrders = list(self.expectedOrders)
        except TypeError:
            raise TypeError("expectedOrders must be array like")
        if len(self.expectedOrders) != len(self.meshTypes):
            raise ValueError(
                "expectedOrders must have the same length as the meshTypes"
            )

        def test_func(n_cells):
            max_h = self.setupMesh(n_cells)
            err = self.getError()
            return err, max_h

        for mesh_type, order, tolerance in zip(
            self.meshTypes, self.tolerance, self.expectedOrders
        ):
            self._meshType = mesh_type
            assert_expected_order(
                test_func,
                self.meshSizes,
                expected_order=order,
                rtol=np.abs(1 - tolerance),
                test_type="mean_at_least",
            )


def assert_expected_order(
    func, n_cells, expected_order=2.0, rtol=0.15, test_type="mean"
):
    """Perform an order test.

    For number of cells specified in `mesh_sizes` call `func`
    and prints mesh size, error, ratio between current and previous error,
    and estimated order of convergence.

    Parameters
    ----------
    func : callable
        Function which should accept an integer representing the number of
        discretizations on the domain and return a tuple of the error and
        the discretization widths.
    n_cells : array_like of int
        List of number of discretizations to pass to func.
    expected_order : float, optional
        The expected order of accuracy for you test
    rtol : float, optional
        The relative tolerance of the order test.
    test_type : {'mean', 'min', 'last', 'all', 'mean_at_least'}
        Which property of the list of calculated orders to test.

    Returns
    -------
    numpy.ndarray
        The calculated order values on success

    Raises
    ------
    AssertionError

    Notes
    -----
    For the different ``test_type`` arguments, different properties of the
    order is tested:

        - `mean`: the mean value of all calculated orders is tested for
          approximate equality with the expected order.
        - `min`: The minimimum value of calculated orders is tested for
          approximate equality with the expected order.
        - `last`: The last calculated order is tested for approximate equality
          with the expected order.
        - `all`: All calculated orders are tested for approximate equality with
          the expected order.
        - `mean_at_least`: The mean is tested to be at least approximately the
          expected order. This is the default test for the previous ``OrderTest``
          class in older versions of `discretize`.

    Examples
    --------
    Testing the convergence order of an central difference operator

    >>> from discretize.tests import assert_expected_order
    >>> func = lambda y: np.cos(y)
    >>> func_deriv = lambda y: -np.sin(y)

    Define the function that returns the error and cell width for
    a given number of discretizations.
    >>> def deriv_error(n):
    ...     # grid points
    ...     nodes = np.linspace(0, 1, n+1)
    ...     cc = 0.5 * (nodes[1:] + nodes[:-1])
    ...     dh = nodes[1]-nodes[0]
    ...     # evaluate the function on nodes
    ...     node_eval = func(nodes)
    ...     # calculate the numerical derivative
    ...     num_deriv = (node_eval[1:] - node_eval[:-1]) / dh
    ...     # calculate the true derivative
    ...     true_deriv = func_deriv(cc)
    ...     # compare the L-inf norm of the error vector
    ...     err = np.linalg.norm(num_deriv - true_deriv, ord=np.inf)
    ...     return err, dh

    Then run the expected order test.
    >>> assert_expected_order(deriv_error, [10, 20, 30, 40, 50])
    """
    __tracebackhide__ = True
    n_cells = np.asarray(n_cells, dtype=int)
    if test_type not in ["mean", "min", "last", "all", "mean_at_least"]:
        raise ValueError
    orders = []
    # Do first values:
    nc = n_cells[0]
    err_last, h_last = func(nc)

    print("_______________________________________________________")
    print("  nc  |    h    |    error    | e(i-1)/e(i) |  order   ")
    print("~~~~~~|~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~")
    print(f"{nc:^6d}|{h_last:^9.2e}|{err_last:^13.3e}|             |")

    for nc in n_cells[1:]:
        err, h = func(nc)
        order = np.log(err / err_last) / np.log(h / h_last)
        print(f"{nc:^6d}|{h:^9.2e}|{err:^13.3e}|{err_last / err:^13.4f}|{order:^10.4f}")
        err_last = err
        h_last = h
        orders.append(order)

    print("-------------------------------------------------------")

    try:
        if test_type == "mean":
            np.testing.assert_allclose(np.mean(orders), expected_order, rtol=rtol)
        elif test_type == "mean_at_least":
            test = np.mean(orders) > expected_order * (1 - rtol)
            if not test:
                raise AssertionError(
                    f"\nOrder mean {np.mean(orders)} is not greater than the expected order "
                    f"{expected_order} within the tolerance {rtol}."
                )
        elif test_type == "min":
            np.testing.assert_allclose(np.min(orders), expected_order, rtol=rtol)
        elif test_type == "last":
            np.testing.assert_allclose(orders[-1], expected_order, rtol=rtol)
        elif test_type == "all":
            np.testing.assert_allclose(orders, expected_order, rtol=rtol)
        print(np.random.choice(happiness))
    except AssertionError as err:
        print(np.random.choice(sadness))
        raise err

    return orders


def rosenbrock(x, return_g=True, return_H=True):
    """Evaluate the Rosenbrock function.

    This is mostly used for testing Gauss-Newton schemes

    Parameters
    ----------
    x : numpy.ndarray
        The (x0, x1) location for the Rosenbrock test
    return_g : bool, optional
        If *True*, return the gradient at *x*
    return_H : bool, optional
        If *True*, return the approximate Hessian at *x*

    Returns
    -------
    tuple
        Rosenbrock function evaluated at (x0, x1), the gradient at (x0, x1) if
        *return_g = True* and the Hessian at (x0, x1) if *return_H = True*
    """
    f = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    g = np.array(
        [2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1), 200 * (x[1] - x[0] ** 2)]
    )
    H = sp.csr_matrix(
        np.array(
            [[-400 * x[1] + 1200 * x[0] ** 2 + 2, -400 * x[0]], [-400 * x[0], 200]]
        )
    )

    out = (f,)
    if return_g:
        out += (g,)
    if return_H:
        out += (H,)
    return out if len(out) > 1 else out[0]


def check_derivative(
    fctn,
    x0,
    num=7,
    plotIt=True,
    dx=None,
    expectedOrder=2,
    tolerance=0.85,
    eps=1e-10,
    ax=None,
):
    """Perform a basic derivative check.

    Compares error decay of 0th and 1st order Taylor approximation at point
    x0 for a randomized search direction.

    Parameters
    ----------
    fctn : function
        Function handle
    x0 : numpy.ndarray
        Point at which to check derivative
    num : int, optional
        number of times to reduce step length to evaluate derivative
    plotIt : bool, optional
        If *True*, plot the convergence of the approximation of the derivative
    dx : numpy.ndarray, optional
        Step direction. By default, this parameter is set to *None* and a random
        step direction is chosen.
    expectedOrder : int, optional
        The expected order of convergence for the numerical derivative
    tolerance : float, optional
        The tolerance on the expected order
    eps : float, optional
        A threshold value for approximately equal to zero
    ax : matplotlib.pyplot.Axes, optional
        An axis object for the convergence plot if *plotIt = True*.
        Otherwise, the function will create a new axis.

    Returns
    -------
    bool
        Whether you passed the test.

    Examples
    --------
    >>> from discretize import tests, utils
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> def simplePass(x):
    ...     return np.sin(x), utils.sdiag(np.cos(x))
    >>> passed = tests.check_derivative(simplePass, np.random.randn(5))
    ==================== check_derivative ====================
    iter    h         |ft-f0|   |ft-f0-h*J0*dx|  Order
    ---------------------------------------------------------
     0   1.00e-01    1.690e-01     8.400e-03      nan
     1   1.00e-02    1.636e-02     8.703e-05      1.985
     2   1.00e-03    1.630e-03     8.732e-07      1.999
     3   1.00e-04    1.629e-04     8.735e-09      2.000
     4   1.00e-05    1.629e-05     8.736e-11      2.000
     5   1.00e-06    1.629e-06     8.736e-13      2.000
     6   1.00e-07    1.629e-07     8.822e-15      1.996
    ========================= PASS! =========================
    Once upon a time, a happy little test passed.
    """
    __tracebackhide__ = True
    # matplotlib is a soft dependencies for discretize,
    # lazy-loaded to decrease load time of discretize.
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        matplotlib = False

    print("{0!s} check_derivative {1!s}".format("=" * 20, "=" * 20))
    print(
        "iter    h         |ft-f0|   |ft-f0-h*J0*dx|  Order\n{0!s}".format(("-" * 57))
    )

    f0, J0 = fctn(x0)

    x0 = mkvc(x0)

    if dx is None:
        dx = np.random.randn(len(x0))

    h = np.logspace(-1, -num, num)
    E0 = np.ones(h.shape)
    E1 = np.ones(h.shape)

    def l2norm(x):
        # because np.norm breaks if they are scalars?
        return np.sqrt(np.real(np.vdot(x, x)))

    for i in range(num):
        # Evaluate at test point
        ft, Jt = fctn(x0 + h[i] * dx)
        # 0th order Taylor
        E0[i] = l2norm(ft - f0)
        # 1st order Taylor
        if inspect.isfunction(J0):
            E1[i] = l2norm(ft - f0 - h[i] * J0(dx))
        else:
            # We assume it is a numpy.ndarray
            E1[i] = l2norm(ft - f0 - h[i] * J0.dot(dx))

        order0 = np.log10(E0[:-1] / E0[1:])
        order1 = np.log10(E1[:-1] / E1[1:])
        print(
            " {0:d}   {1:1.2e}    {2:1.3e}     {3:1.3e}      {4:1.3f}".format(
                i, h[i], E0[i], E1[i], np.nan if i == 0 else order1[i - 1]
            )
        )

    @requires({"matplotlib": matplotlib})
    def _plot_it(axes, passed):
        if axes is None:
            axes = plt.subplot(111)
        axes.loglog(h, E0, "b")
        axes.loglog(h, E1, "g--")
        axes.set_title(
            "Check Derivative - {0!s}".format(("PASSED :)" if passed else "FAILED :("))
        )
        axes.set_xlabel("h")
        axes.set_ylabel("Error")
        leg = axes.legend(
            [r"$\mathcal{O}(h)$", r"$\mathcal{O}(h^2)$"],
            loc="best",
            title=r"$f(x + h\Delta x) - f(x) - h g(x) \Delta x - \mathcal{O}(h^2) = 0$",
            frameon=False,
        )
        plt.setp(leg.get_title(), fontsize=15)
        plt.show()

    # Ensure we are about precision
    order0 = order0[E0[1:] > eps]
    order1 = order1[E1[1:] > eps]

    # belowTol = order1.size == 0 and order0.size >= 0
    # # Make sure we get the correct order
    # correctOrder = order1.size > 0 and np.mean(order1) > tolerance * expectedOrder
    #
    # passTest = belowTol or correctOrder
    try:
        if order1.size == 0:
            # This should happen if all of the 1st order taylor approximation errors
            # were below epsilon, common if the original function was linear.
            # Thus it has no higher order derivatives.
            pass
        else:
            test = np.mean(order1) > tolerance * expectedOrder
            if not test:
                raise AssertionError(
                    f"\n Order mean {np.mean(order1)} is not greater than"
                    f" {tolerance} of the expected order {expectedOrder}."
                )
        print("{0!s} PASS! {1!s}".format("=" * 25, "=" * 25))
        print(np.random.choice(happiness) + "\n")
        if plotIt:
            _plot_it(ax, True)
    except AssertionError as err:
        print(
            "{0!s}\n{1!s} FAIL! {2!s}\n{3!s}".format(
                "*" * 57, "<" * 25, ">" * 25, "*" * 57
            )
        )
        print(np.random.choice(sadness) + "\n")
        if plotIt:
            _plot_it(ax, False)
        raise err

    return True


def get_quadratic(A, b, c=0):
    r"""Return a function that evaluates the given quadratic.

    Given **A**, **b** and *c*, this returns a function that evaluates
    the quadratic for a vector **x**. Where :math:`\mathbf{A} \in \mathbb{R}^{NxN}`,
    :math:`\mathbf{b} \in \mathbb{R}^N` and :math:`c` is a constant,
    this function evaluates the following quadratic:

    .. math::

        Q( \mathbf{x} ) = \frac{1}{2} \mathbf{x^T A x + b^T x} + c

    for a vector :math:`\mathbf{x}`. It also optionally returns the gradient of the
    above equation, and its Hessian.

    Parameters
    ----------
    A : (N, N) numpy.ndarray
        A square matrix
    b : (N) numpy.ndarray
        A vector
    c : float
        A constant

    Returns
    -------
    function :
        The callable function that returns the quadratic evaluation, and optionally its
        gradient, and Hessian.
    """

    def Quadratic(x, return_g=True, return_H=True):
        f = 0.5 * x.dot(A.dot(x)) + b.dot(x) + c
        out = (f,)
        if return_g:
            g = A.dot(x) + b
            out += (g,)
        if return_H:
            H = A
            out += (H,)
        return out if len(out) > 1 else out[0]

    return Quadratic


def assert_isadjoint(
    forward,
    adjoint,
    shape_u,
    shape_v,
    complex_u=False,
    complex_v=False,
    clinear=True,
    rtol=1e-6,
    atol=0.0,
    assert_error=True,
):
    r"""Do a dot product test for the forward operator and its adjoint operator.

    Dot product test to verify the correctness of the adjoint operator
    :math:`F^H` of the forward operator :math:`F`.

    .. math::

        \mathbf{v}^H ( \mathbf{F} \mathbf{u} ) =
        ( \mathbf{F}^H \mathbf{v} )^H \mathbf{u}


    Parameters
    ----------
    forward : callable
        Forward operator.

    adjoint : callable
        Adjoint operator.

    shape_u : int, tuple of int
        Shape of vector ``u`` passed in to ``forward``; it is accordingly the
        expected shape of the vector returned from the ``adjoint``.

    shape_v : int, tuple of int
        Shape of vector ``v`` passed in to ``adjoint``; it is accordingly the
        expected shape of the vector returned from the ``forward``.

    complex_u : bool, default: False
        If True, vector ``u`` passed to ``forward`` is a complex vector;
        accordingly the ``adjoint`` is expected to return a complex vector.

    complex_v : bool, default: False
        If True, vector ``v`` passed to ``adjoint`` is a complex vector;
        accordingly the ``forward`` is expected to return a complex vector.

    clinear : bool, default: True
        If operator is complex-linear (True) or real-linear (False).

    rtol : float, default: 1e-6
        Relative tolerance.

    atol : float, default: 0.0
        Absolute tolerance.

    assert_error : bool, default: True
        By default this test is an assertion (silent if passed, raising an
        assertion error if failed). If set to False, the result of the test is
        returned as boolean and a message is printed.


    Returns
    -------
    passed : bool, optional
        Result of the dot product test; only returned if ``assert_error`` is False.

    Raises
    ------
    AssertionError
        If the dot product test fails (only if assert_error=True).

    """
    __tracebackhide__ = True

    def random(size, iscomplex):
        """Create random data of size and dtype of <size>."""
        out = rng.standard_normal(size)
        if iscomplex:
            out = out + 1j * rng.standard_normal(size)
        return out

    # Create random vectors u and v.
    u = random(np.product(shape_u), complex_u).reshape(shape_u)
    v = random(np.product(shape_v), complex_v).reshape(shape_v)

    # Carry out dot product test.
    fwd_u = forward(u)
    adj_v = adjoint(v)
    if clinear:
        lhs = np.vdot(v, fwd_u)  # lhs := v^H * (fwd * u)
        rhs = np.vdot(adj_v, u)  # rhs := (adj * v)^H * u
    else:
        lhs = np.vdot(v.real, fwd_u.real) + np.vdot(v.imag, fwd_u.imag)
        rhs = np.vdot(adj_v.real, u.real) + np.vdot(adj_v.imag, u.imag)

    # Check if they are the same.
    if assert_error:
        np.testing.assert_allclose(
            rhs, lhs, rtol=rtol, atol=atol, err_msg="Adjoint test failed"
        )

    else:
        passed = np.allclose(rhs, lhs, rtol=rtol, atol=atol)

        print(
            f"Adjoint test {'PASSED' if passed else 'FAILED'} ::  "
            f"{abs(rhs-lhs):.3e} < {atol+rtol*abs(lhs):.3e}  :: "
            f"|rhs-lhs| < atol + rtol|lhs|"
        )

        return passed


# DEPRECATIONS
setupMesh = deprecate_function(
    setup_mesh, "setupMesh", removal_version="1.0.0", error=True
)
Rosenbrock = deprecate_function(
    rosenbrock, "Rosenbrock", removal_version="1.0.0", error=True
)
checkDerivative = deprecate_function(
    check_derivative, "checkDerivative", removal_version="1.0.0", error=True
)
getQuadratic = deprecate_function(
    get_quadratic, "getQuadratic", removal_version="1.0.0", error=True
)
