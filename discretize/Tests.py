from __future__ import print_function
import numpy as np
import scipy.sparse as sp

from discretize.utils import mkvc, sdiag
from discretize import utils
from discretize import TensorMesh, CurvilinearMesh, CylMesh
from discretize.utils.codeutils import requires

try:
    from discretize.TreeMesh import TreeMesh as Tree
except ImportError as e:
    Tree = None

import unittest
import inspect

# matplotlib is a soft dependencies for discretize
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = False

try:
    import getpass
    name = getpass.getuser()[0].upper() + getpass.getuser()[1:]
except Exception as e:
    name = 'You'

happiness = [
    'The test be workin!', 'You get a gold star!', 'Yay passed!',
    'Happy little convergence test!', 'That was easy!',
    'Testing is important.', 'You are awesome.', 'Go Test Go!',
    'Once upon a time, a happy little test passed.',
    'And then everyone was happy.','Not just a pretty face '+name,
    'You deserve a pat on the back!', 'Well done '+name+'!',
    'Awesome, '+name+', just awesome.'
]
sadness = [
    'No gold star for you.', 'Try again soon.',
    'Thankfully,  persistence is a great substitute for talent.',
    'It might be easier to call this a feature...', 'Coffee break?',
    'Boooooooo  :(',  'Testing is important. Do it again.',
    "Did you put your clever trousers on today?",
    'Just think about a dancing dinosaur and life will get better!',
    'You had so much promise '+name+', oh well...', name.upper()+' ERROR!',
    'Get on it '+name+'!', 'You break it, you fix it.'
]


def setupMesh(meshType, nC, nDim):
    """
    For a given number of cells nc, generate a TensorMesh with uniform
    cells with edge length h=1/nc.
    """

    if 'TensorMesh' in meshType:
        if 'uniform' in meshType:
            h = [nC, nC, nC]
        elif 'random' in meshType:
            h1 = np.random.rand(nC)*nC*0.5 + nC*0.5
            h2 = np.random.rand(nC)*nC*0.5 + nC*0.5
            h3 = np.random.rand(nC)*nC*0.5 + nC*0.5
            h = [hi/np.sum(hi) for hi in [h1, h2, h3]]  # normalize
        else:
            raise Exception('Unexpected meshType')

        mesh = TensorMesh(h[:nDim])
        max_h = max([np.max(hi) for hi in mesh.h])

    elif 'CylMesh' in meshType:
        if 'uniform' in meshType:
            h = [nC, nC, nC]
        elif 'random' in meshType:
            h1 = np.random.rand(nC)*nC*0.5 + nC*0.5
            h2 = np.random.rand(nC)*nC*0.5 + nC*0.5
            h3 = np.random.rand(nC)*nC*0.5 + nC*0.5
            h = [hi/np.sum(hi) for hi in [h1, h2, h3]]  # normalize
            h[1] = h[1]*2*np.pi
        else:
            raise Exception('Unexpected meshType')

        if nDim == 2:
            mesh = CylMesh([h[0], 1, h[2]])
            max_h = max([np.max(hi) for hi in [mesh.hx, mesh.hz]])
        elif nDim == 3:
            mesh = CylMesh(h)
            max_h = max([np.max(hi) for hi in mesh.h])

    elif 'Curv' in meshType:
        if 'uniform' in meshType:
            kwrd = 'rect'
        elif 'rotate' in meshType:
            kwrd = 'rotate'
        else:
            raise Exception('Unexpected meshType')
        if nDim == 1:
            raise Exception('Lom not supported for 1D')
        elif nDim == 2:
            X, Y = utils.exampleLrmGrid([nC, nC], kwrd)
            mesh = CurvilinearMesh([X, Y])
        elif nDim == 3:
            X, Y, Z = utils.exampleLrmGrid([nC, nC, nC], kwrd)
            mesh = CurvilinearMesh([X, Y, Z])
        max_h = 1./nC

    elif 'Tree' in meshType:
        if Tree is None:
            raise Exception(
                "Tree Mesh not installed. Run 'python setup.py install'"
            )
        nC *= 2
        if 'uniform' in meshType or 'notatree' in meshType:
            h = [nC, nC, nC]
        elif 'random' in meshType:
            h1 = np.random.rand(nC)*nC*0.5 + nC*0.5
            h2 = np.random.rand(nC)*nC*0.5 + nC*0.5
            h3 = np.random.rand(nC)*nC*0.5 + nC*0.5
            h = [hi/np.sum(hi) for hi in [h1, h2, h3]]  # normalize
        else:
            raise Exception('Unexpected meshType')

        levels = int(np.log(nC)/np.log(2))
        mesh = Tree(h[:nDim], levels=levels)

        def function(cell):
            if 'notatree' in meshType:
                return levels - 1
            r = cell.center - 0.5
            dist = np.sqrt(r.dot(r))
            if dist < 0.2:
                return levels
            return levels - 1
        mesh.refine(function)
        # mesh.number()
        # mesh.plotGrid(showIt=True)
        max_h = max([np.max(hi) for hi in mesh.h])
    return mesh, max_h


class OrderTest(unittest.TestCase):
    """

    OrderTest is a base class for testing convergence orders with respect to mesh
    sizes of integral/differential operators.

    Mathematical Problem:

        Given are an operator A and its discretization A[h]. For a given test function f
        and h --> 0  we compare:

        .. math::
            error(h) = \| A[h](f) - A(f) \|_{\infty}

        Note that you can provide any norm.

        Test is passed when estimated rate order of convergence is  at least within the specified tolerance of the
        estimated rate supplied by the user.

    Minimal example for a curl operator::

        class TestCURL(OrderTest):
            name = "Curl"

            def getError(self):
                # For given Mesh, generate A[h], f and A(f) and return norm of error.


                fun  = lambda x: np.cos(x)  # i (cos(y)) + j (cos(z)) + k (cos(x))
                sol = lambda x: np.sin(x)  # i (sin(z)) + j (sin(x)) + k (sin(y))


                Ex = fun(self.M.gridEx[:, 1])
                Ey = fun(self.M.gridEy[:, 2])
                Ez = fun(self.M.gridEz[:, 0])
                f = np.concatenate((Ex, Ey, Ez))

                Fx = sol(self.M.gridFx[:, 2])
                Fy = sol(self.M.gridFy[:, 0])
                Fz = sol(self.M.gridFz[:, 1])
                Af = np.concatenate((Fx, Fy, Fz))

                # Generate DIV matrix
                Ah = self.M.edgeCurl

                curlE = Ah*E
                err = np.linalg.norm((Ah*f -Af), np.inf)
                return err

            def test_order(self):
                # runs the test
                self.orderTest()

    See also: test_operatorOrder.py

    """

    name = "Order Test"
    expectedOrders = 2.  # This can be a list of orders, must be the same length as meshTypes
    tolerance = 0.85     # This can also be a list, must be the same length as meshTypes
    meshSizes = [4, 8, 16, 32]
    meshTypes = ['uniformTensorMesh']
    _meshType = meshTypes[0]
    meshDimension = 3

    def setupMesh(self, nC):
        mesh, max_h = setupMesh(self._meshType, nC, self.meshDimension)
        self.M = mesh
        return max_h

    def getError(self):
        """For given h, generate A[h], f and A(f) and return norm of error."""
        return 1.

    def orderTest(self):
        """
        For number of cells specified in meshSizes setup mesh, call getError
        and prints mesh size, error, ratio between current and previous error,
        and estimated order of convergence.


        """
        if not isinstance(self.meshTypes, list):
            raise TypeError('meshTypes must be a list')
        if type(self.tolerance) is not list:
            self.tolerance = np.ones(len(self.meshTypes))*self.tolerance

        # if we just provide one expected order, repeat it for each mesh type
        if type(self.expectedOrders) == float or type(self.expectedOrders) == int:
            self.expectedOrders = [self.expectedOrders for i in self.meshTypes]
        if isinstance(self.expectedOrders, np.ndarray):
            self.expectedOrders = list(self.expectedOrders)
        assert type(self.expectedOrders) == list, 'expectedOrders must be a list'
        assert len(self.expectedOrders) == len(self.meshTypes), 'expectedOrders must have the same length as the meshTypes'

        for ii_meshType, meshType in enumerate(self.meshTypes):
            self._meshType = meshType
            self._tolerance = self.tolerance[ii_meshType]
            self._expectedOrder = self.expectedOrders[ii_meshType]

            order = []
            err_old = 0.
            max_h_old = 0.
            for ii, nc in enumerate(self.meshSizes):
                max_h = self.setupMesh(nc)
                err = self.getError()
                if ii == 0:
                    print('')
                    print(self._meshType + ':  ' + self.name)
                    print('_____________________________________________')
                    print('   h  |    error    | e(i-1)/e(i) |  order')
                    print('~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~')
                    print('{0:4d}  |  {1:8.2e}   |'.format(nc, err))
                else:
                    order.append(np.log(err/err_old)/np.log(max_h/max_h_old))
                    print('{0:4d}  |  {1:8.2e}   |   {2:6.4f}    |  {3:6.4f}'.format(nc, err, err_old/err, order[-1]))
                err_old = err
                max_h_old = max_h
            print('---------------------------------------------')
            passTest = np.mean(np.array(order)) > self._tolerance*self._expectedOrder
            if passTest:
                print(happiness[np.random.randint(len(happiness))])
            else:
                print('Failed to pass test on ' + self._meshType + '.')
                print(sadness[np.random.randint(len(sadness))])
            print('')
            self.assertTrue(passTest)


def Rosenbrock(x, return_g=True, return_H=True):
    """Rosenbrock function for testing GaussNewton scheme"""

    f = 100*(x[1]-x[0]**2)**2+(1-x[0])**2
    g = np.array([2*(200*x[0]**3-200*x[0]*x[1]+x[0]-1), 200*(x[1]-x[0]**2)])
    H = sp.csr_matrix(np.array([[-400*x[1]+1200*x[0]**2+2, -400*x[0]], [-400*x[0], 200]]))

    out = (f,)
    if return_g:
        out += (g,)
    if return_H:
        out += (H,)
    return out if len(out) > 1 else out[0]


def checkDerivative(fctn, x0, num=7, plotIt=True, dx=None, expectedOrder=2, tolerance=0.85, eps=1e-10, ax=None):
    """
        Basic derivative check

        Compares error decay of 0th and 1st order Taylor approximation at point
        x0 for a randomized search direction.

        :param callable fctn: function handle
        :param numpy.ndarray x0: point at which to check derivative
        :param int num: number of times to reduce step length, h
        :param bool plotIt: if you would like to plot
        :param numpy.ndarray dx: step direction
        :param int expectedOrder: The order that you expect the derivative to yield.
        :param float tolerance: The tolerance on the expected order.
        :param float eps: What is zero?
        :rtype: bool
        :return: did you pass the test?!


        .. plot::
            :include-source:

            from discretize import Tests, utils
            import numpy as np

            def simplePass(x):
                return np.sin(x), utils.sdiag(np.cos(x))
            Tests.checkDerivative(simplePass, np.random.randn(5))
    """

    print("{0!s} checkDerivative {1!s}".format('='*20, '='*20))
    print("iter    h         |ft-f0|   |ft-f0-h*J0*dx|  Order\n{0!s}".format(('-'*57)))

    f0, J0 = fctn(x0)

    x0 = mkvc(x0)

    if dx is None:
        dx = np.random.randn(len(x0))

    h  = np.logspace(-1, -num, num)
    E0 = np.ones(h.shape)
    E1 = np.ones(h.shape)

    def l2norm(x):
        # because np.norm breaks if they are scalars?
        return np.sqrt(np.real(np.vdot(x, x)))

    for i in range(num):
        # Evaluate at test point
        ft, Jt = fctn( x0 + h[i]*dx )
        # 0th order Taylor
        E0[i] = l2norm( ft - f0 )
        # 1st order Taylor
        if inspect.isfunction(J0):
            E1[i] = l2norm( ft - f0 - h[i]*J0(dx) )
        else:
            # We assume it is a numpy.ndarray
            E1[i] = l2norm( ft - f0 - h[i]*J0.dot(dx) )

        order0 = np.log10(E0[:-1]/E0[1:])
        order1 = np.log10(E1[:-1]/E1[1:])
        print(" {0:d}   {1:1.2e}    {2:1.3e}     {3:1.3e}      {4:1.3f}".format(i, h[i], E0[i], E1[i], np.nan if i == 0 else order1[i-1]))

    # Ensure we are about precision
    order0 = order0[E0[1:] > eps]
    order1 = order1[E1[1:] > eps]
    belowTol = (order1.size == 0 and order0.size >= 0)
    # Make sure we get the correct order
    correctOrder = order1.size > 0 and np.mean(order1) > tolerance * expectedOrder

    passTest = belowTol or correctOrder

    if passTest:
        print("{0!s} PASS! {1!s}".format('='*25, '='*25))
        print(happiness[np.random.randint(len(happiness))]+'\n')
    else:
        print("{0!s}\n{1!s} FAIL! {2!s}\n{3!s}".format('*'*57, '<'*25, '>'*25, '*'*57))
        print(sadness[np.random.randint(len(sadness))]+'\n')


    @requires({'matplotlib': matplotlib})
    def plot_it():
        if plotIt:
            if ax is None:
                ax = plt.subplot(111)
            ax.loglog(h, E0, 'b')
            ax.loglog(h, E1, 'g--')
            ax.set_title(
                'Check Derivative - {0!s}'.format(('PASSED :)'
                if passTest else 'FAILED :('))
            )
            ax.set_xlabel('h')
            ax.set_ylabel('Error')
            leg = ax.legend(
                ['$\mathcal{O}(h)$', '$\mathcal{O}(h^2)$'], loc='best',
                title="$f(x + h\Delta x) - f(x) - h g(x) \Delta x - \mathcal{O}(h^2) = 0$",
                frameon=False)
            plt.setp(leg.get_title(),fontsize=15)
            plt.show()

    plot_it()

    return passTest


def getQuadratic(A, b, c=0):
    """
        Given A, b and c, this returns a quadratic, Q

        .. math::

            \mathbf{Q( x ) = 0.5 x A x + b x} + c
    """
    def Quadratic(x, return_g=True, return_H=True):
        f = 0.5 * x.dot( A.dot(x)) + b.dot( x ) + c
        out = (f,)
        if return_g:
            g = A.dot(x) + b
            out += (g,)
        if return_H:
            H = A
            out += (H,)
        return out if len(out) > 1 else out[0]
    return Quadratic
