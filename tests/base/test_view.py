from __future__ import print_function

import unittest
import numpy as np
import matplotlib.pyplot as plt

import discretize

import pytest

np.random.seed(16)

TOL = 1e-1


class Cyl3DView(unittest.TestCase):
    def setUp(self):
        self.mesh = discretize.CylMesh([10, 4, 12])

    def test_incorrectAxesWarnings(self):

        # axes aren't polar
        fig, ax = plt.subplots(1, 1)
        # test z-slice
        with pytest.warns(UserWarning):
            self.mesh.plotGrid(slice='z', ax=ax)

        # axes aren't right shape
        with pytest.warns(UserWarning):
            self.mesh.plotGrid(slice='both', ax=ax)
            self.mesh.plotGrid(ax=ax)

        # this should be fine
        self.mesh.plotGrid(slice='theta', ax=ax)

        fig, ax = plt.subplots(2, 1)
        # axes are right shape, but not polar
        with pytest.warns(UserWarning):
            self.mesh.plotGrid(slice='both', ax=ax)
            self.mesh.plotGrid(ax=ax)

        # these should be fine
        self.mesh.plotGrid()

        ax0 = plt.subplot(121, projection='polar')
        ax1 = plt.subplot(122)

        self.mesh.plotGrid(slice='z', ax=ax0)  # plot z only
        self.mesh.plotGrid(slice='theta', ax=ax1)  # plot theta only
        self.mesh.plotGrid(slice='both', ax=[ax0, ax1])  # plot both
        self.mesh.plotGrid(slice='both', ax=[ax1, ax0])  # plot both
        self.mesh.plotGrid(ax=[ax1, ax0])  # plot both

    def test_plotImage(self):
        with self.assertRaises(Exception):
            self.mesh.plotImage(np.random.rand(self.mesh.nC))

if __name__ == '__main__':
    unittest.main()
