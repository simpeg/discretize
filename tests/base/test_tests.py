import pytest
import discretize
import numpy as np
from discretize.utils import volume_average
from discretize.tests import assert_isadjoint


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

        assert_isadjoint(fwd, adj, shape_u=(4, nt), shape_v=(4,))

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
        )
