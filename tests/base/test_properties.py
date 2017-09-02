import unittest
import os
import discretize


def compare_meshes(mesh0, mesh1):
    assert mesh0.nC == mesh1.nC , (
        'Number of cells not the same, {} != {}'.format(mesh0.nC, mesh1.nC)
    )

    assert (mesh0.x0 == mesh1.x0).all(), (
        'x0 different. {} != {}'.format(mesh0.x0, mesh1.x0)
    )

    for i in range(len(mesh0.h)):
        assert (mesh0.h[i] == mesh1.h[i]).all(), (
            'mesh h[{}] different'.format(i)
        )

    if hasattr(mesh0, 'cells'):
        assert (mesh0.cells == mesh1.cells)


class TensorTest(unittest.TestCase):

    n = [4, 5, 9]
    x0 = [-0.5, -0.25, 0]

    def setUp(self):
        self.mesh = discretize.TensorMesh(self.n, x0=self.x0)

    def test_save_load(self):
        print('\nTesting save / load of Tensor Mesh ...')
        mesh0 = self.mesh
        f = mesh0.save()
        mesh1 = discretize.utils.load_mesh(f)
        compare_meshes(mesh0, mesh1)
        os.remove(f)
        print('ok\n')

    def test_copy(self):
        print('\nTesting copy of Tensor Mesh ...')
        mesh0 = self.mesh
        mesh1 = mesh0.copy()
        compare_meshes(mesh0, mesh1)
        print('ok\n')


class CylTest(unittest.TestCase):

    n = [4, 1, 9]

    def setUp(self):
        self.mesh = discretize.CylMesh(self.n, x0='00C')

    def test_save_load(self):
        print('\nTesting save / load of Cyl Mesh ...')
        mesh0 = self.mesh
        f = mesh0.save()
        mesh1 = discretize.utils.load_mesh(f)
        compare_meshes(mesh0, mesh1)
        os.remove(f)
        print('ok\n')

    def test_copy(self):
        print('\nTesting copy of Cyl Mesh ...')
        mesh0 = self.mesh
        mesh1 = mesh0.copy()
        compare_meshes(mesh0, mesh1)
        print('ok\n')


class TreeTest(unittest.TestCase):

    def setUp(self):
        M = discretize.TreeMesh([8, 8])

        def refine(cell):
            xyz = cell.center
            dist = ((xyz - [0.25, 0.25])**2).sum()**0.5
            if dist < 0.25:
                return 3
            return 2

        M.refine(refine)
        M.number()

        self.mesh = M

    def test_save_load(self):
        print('\nTesting save / load of Tree Mesh ...')
        mesh0 = self.mesh
        f = mesh0.save()
        mesh1 = discretize.utils.load_mesh(f)
        compare_meshes(mesh0, mesh1)
        os.remove(f)
        print('ok\n')

    def test_copy(self):
        print('\nTesting copy of Tree Mesh ...')
        mesh0 = self.mesh
        mesh1 = mesh0.copy()
        compare_meshes(mesh0, mesh1)
        print('ok\n')


if __name__ == '__main__':
    unittest.main()
