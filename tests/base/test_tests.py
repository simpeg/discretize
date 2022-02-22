import pytest
import discretize
import numpy as np
from discretize.tests import dottest
from discretize.utils import volume_average


class TestDottest:
    def test_defaults(self, capsys):
        # Use volume_average to test default case.

        h = [[11, 20.5, 33], [222, 111, 333], np.ones(4) * 55.5]
        mesh1 = discretize.TensorMesh(h, "CCC")

        hx = np.ones(20) * 20
        mesh2 = discretize.TensorMesh([hx, hx, hx], "CCC")

        P = discretize.utils.volume_average(mesh1, mesh2)

        # return
        assert dottest(
            lambda u: P * u,
            lambda v: P.T * v,
            mesh1.n_cells,
            mesh2.n_cells,
            verb=True,
        )
        out, _ = capsys.readouterr()
        assert "Dot test PASSED" in out

        # raise error
        with pytest.raises(AssertionError, match="Dot test FAILED "):
            dottest(
                lambda u: P * u * 2,  # Add erroneous factor
                lambda v: P.T * v,
                mesh1.n_cells,
                mesh2.n_cells,
                raise_error=True,
            )

    def test_shape_order(self):
        # Def sum operator over axis 1; ravelled in 'F' so needs 'F'-order.

        nt = 3

        def fwd(inp):
            return np.sum(inp, 1).ravel("F")

        # Def adj of fwd
        def adj(inp):
            out = np.expand_dims(inp, 1)
            return np.tile(out, nt).ravel("F")

        assert dottest(fwd, adj, (4, nt), (4,), fwd_order="F", verb=True)

    def test_complex_clinear(self):
        # The complex conjugate is self-adjoint, real-linear.
        assert dottest(
            np.conj,
            np.conj,
            (4, 3),
            (4, 3),
            fwd_complex=True,
            adj_complex=True,
            clinear=False,
            verb=True,
        )
