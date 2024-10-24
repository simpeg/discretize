import sys
import pytest
import discretize
import subprocess
import numpy as np
import scipy.sparse as sp
from discretize.tests import assert_isadjoint, check_derivative, assert_expected_order


class TestAssertIsAdjoint:
    def test_defaults(self, capsys):
        # Use volume_average to test default case.

        h = [[11, 20.5, 33], [222, 111, 333], np.ones(4) * 55.5]
        mesh1 = discretize.TensorMesh(h, "CCC")

        hx = np.ones(20) * 20
        mesh2 = discretize.TensorMesh([hx, hx, hx], "CCC")

        P = discretize.utils.volume_average(mesh1, mesh2)

        # return
        out1 = assert_isadjoint(
            lambda u: P * u,
            lambda v: P.T * v,
            mesh1.n_cells,
            mesh2.n_cells,
            assert_error=False,
            random_seed=41,
        )
        out2, _ = capsys.readouterr()
        assert out1
        assert "Adjoint test PASSED" in out2

        # raise error
        with pytest.raises(AssertionError, match="Adjoint test failed"):
            assert_isadjoint(
                lambda u: P * u * 2,  # Add erroneous factor
                lambda v: P.T * v,
                mesh1.n_cells,
                mesh2.n_cells,
                random_seed=42,
            )

    def test_different_shape(self):
        # Def sum operator over axis 1; ravelled in 'F' so needs 'F'-order.

        nt = 3

        def fwd(inp):
            return np.sum(inp, 1)

        # Def adj of fwd
        def adj(inp):
            out = np.expand_dims(inp, 1)
            return np.tile(out, nt)

        assert_isadjoint(
            fwd,
            adj,
            shape_u=(4, nt),
            shape_v=(4,),
            random_seed=42,
        )

    def test_complex_clinear(self):
        # The complex conjugate is self-adjoint, real-linear.
        assert_isadjoint(
            np.conj,
            np.conj,
            (4, 3),
            (4, 3),
            complex_u=True,
            complex_v=True,
            clinear=False,
            random_seed=112,
        )


class TestCheckDerivative:
    def test_simplePass(self):
        def simplePass(x):
            return np.sin(x), sp.diags(np.cos(x))

        rng = np.random.default_rng(5322)
        check_derivative(
            simplePass, rng.standard_normal(5), plotIt=False, random_seed=42
        )

    def test_simpleFunction(self):
        def simpleFunction(x):
            return np.sin(x), lambda xi: np.cos(x) * xi

        rng = np.random.default_rng(5322)
        check_derivative(
            simpleFunction, rng.standard_normal(5), plotIt=False, random_seed=23
        )

    def test_simpleFail(self):
        def simpleFail(x):
            return np.sin(x), -sp.diags(np.cos(x))

        rng = np.random.default_rng(5322)
        with pytest.raises(AssertionError):
            check_derivative(
                simpleFail, rng.standard_normal(5), plotIt=False, random_seed=64
            )


@pytest.mark.parametrize("test_type", ["mean", "min", "last", "all", "mean_at_least"])
def test_expected_order_pass(test_type):
    func = lambda y: np.cos(y)
    func_deriv = lambda y: -np.sin(y)

    def deriv_error(n):
        # The l-inf norm of the average error vector does
        # follow second order convergence.
        nodes = np.linspace(0, 1, n + 1)
        cc = 0.5 * (nodes[1:] + nodes[:-1])
        dh = nodes[1] - nodes[0]
        node_eval = func(nodes)
        num_deriv = (node_eval[1:] - node_eval[:-1]) / dh
        true_deriv = func_deriv(cc)
        err = np.linalg.norm(num_deriv - true_deriv, ord=np.inf)
        return err, dh

    assert_expected_order(deriv_error, [10, 20, 30, 40, 50], test_type=test_type)


@pytest.mark.parametrize("test_type", ["mean", "min", "last", "all", "mean_at_least"])
def test_expected_order_failed(test_type):
    func = lambda y: np.cos(y)
    func_deriv = lambda y: -np.sin(y)

    def deriv_error(n):
        # The l2 norm of the average error vector does not
        # follow second order convergence.
        nodes = np.linspace(0, 1, n + 1)
        cc = 0.5 * (nodes[1:] + nodes[:-1])
        dh = nodes[1] - nodes[0]
        node_eval = func(nodes)
        num_deriv = (node_eval[1:] - node_eval[:-1]) / dh
        true_deriv = func_deriv(cc)
        err = np.linalg.norm(num_deriv - true_deriv)
        return err, dh

    with pytest.raises(AssertionError):
        assert_expected_order(deriv_error, [10, 20, 30, 40, 50], test_type=test_type)


def test_expected_order_bad_test_type():
    # should fail fast if given a bad test_type
    with pytest.raises(ValueError):
        assert_expected_order(None, [10, 20, 30, 40, 50], test_type="not_a_test")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Not Linux.")
def test_import_time():
    # Relevant for the CLI: How long does it take to import?
    cmd = ["time", "-f", "%U", "python", "-c", "import discretize"]
    # Run it twice, just in case.
    subprocess.run(cmd)
    subprocess.run(cmd)
    # Capture it
    out = subprocess.run(cmd, capture_output=True)

    # Currently we check t < 1.0s.
    assert float(out.stderr.decode("utf-8")[:-1]) < 1.0
