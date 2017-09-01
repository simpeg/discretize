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




if __name__ == '__main__':
    unittest.main()
