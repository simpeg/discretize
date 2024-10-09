import unittest
import numpy as np
import matplotlib.pyplot as plt

import discretize

import pytest

TOL = 1e-1


class Cyl3DView(unittest.TestCase):
    def setUp(self):
        self.mesh = discretize.CylindricalMesh([10, 4, 12])

    def test_incorrectAxesWarnings(self):
        # axes aren't polar
        fig, ax = plt.subplots(1, 1)
        # test z-slice
        with pytest.warns(UserWarning):
            self.mesh.plot_grid(slice="z", ax=ax)

        # axes aren't right shape
        with pytest.warns(UserWarning):
            self.mesh.plot_grid(slice="both", ax=ax)
            self.mesh.plot_grid(ax=ax)

        # this should be fine
        self.mesh.plot_grid(slice="theta", ax=ax)

        fig, ax = plt.subplots(2, 1)
        # axes are right shape, but not polar
        with pytest.warns(UserWarning):
            self.mesh.plot_grid(slice="both", ax=ax)
            self.mesh.plot_grid(ax=ax)

        # these should be fine
        self.mesh.plot_grid()

        ax0 = plt.subplot(121, projection="polar")
        ax1 = plt.subplot(122)

        self.mesh.plot_grid(slice="z", ax=ax0)  # plot z only
        self.mesh.plot_grid(slice="theta", ax=ax1)  # plot theta only
        self.mesh.plot_grid(slice="both", ax=[ax0, ax1])  # plot both
        self.mesh.plot_grid(slice="both", ax=[ax1, ax0])  # plot both
        self.mesh.plot_grid(ax=[ax1, ax0])  # plot both
        plt.close("all")

    def test_plot_image(self):
        with self.assertRaises(NotImplementedError):
            self.mesh.plot_image(np.empty(self.mesh.nC))


if __name__ == "__main__":
    unittest.main()
