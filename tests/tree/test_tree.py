from __future__ import print_function
import numpy as np
import unittest
import discretize

TOL = 1e-8


class TestSimpleQuadTree(unittest.TestCase):
    def test_counts(self):
        nc = 8
        h1 = np.random.rand(nc) * nc * 0.5 + nc * 0.5
        h2 = np.random.rand(nc) * nc * 0.5 + nc * 0.5
        h = [hi / np.sum(hi) for hi in [h1, h2]]  # normalize
        M = discretize.TreeMesh(h)
        points = np.array([[0.1, 0.1, 0.3]])
        level = np.array([3])

        M.insert_cells(points, level)
        M.number()

        self.assertEqual(M.nhFx, 4)
        self.assertEqual(M.nFx, 12)

        self.assertTrue(np.allclose(M.vol.sum(), 1.0))
        # self.assertTrue(np.allclose(np.r_[M._areaFxFull, M._areaFyFull], M._deflationMatrix('F') * M.area)

    def test_getitem(self):
        M = discretize.TreeMesh([4, 4])
        M.refine(1)
        self.assertEqual(M.nC, 4)
        self.assertEqual(len(M), M.nC)
        self.assertTrue(np.allclose(M[0].center, [0.25, 0.25]))
        # actual = [[0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5]]
        # for i, n in enumerate(M[0].nodes):
        #    self.assertTrue(np.allclose(, actual[i])

    def test_getitem3D(self):
        M = discretize.TreeMesh([4, 4, 4])
        M.refine(1)
        self.assertEqual(M.nC, 8)
        self.assertEqual(len(M), M.nC)
        self.assertTrue(np.allclose(M[0].center, [0.25, 0.25, 0.25]))
        # actual = [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0],
        #          [0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]]
        # for i, n in enumerate(M[0].nodes):
        #    self.assertTrue(np.allclose(M._gridN[n, :], actual[i])

    def test_refine(self):
        M = discretize.TreeMesh([4, 4, 4])
        M.refine(1)
        self.assertEqual(M.nC, 8)

    def test_h_gridded_2D(self):
        hx, hy = np.ones(4), np.r_[1.0, 2.0, 3.0, 4.0]

        M = discretize.TreeMesh([hx, hy])

        def refinefcn(cell):
            xyz = cell.center
            d = (xyz ** 2).sum() ** 0.5
            if d < 3:
                return 2
            return 1

        M.refine(refinefcn)
        H = M.h_gridded

        test_hx = np.all(H[:, 0] == np.r_[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        test_hy = np.all(H[:, 1] == np.r_[1.0, 1.0, 2.0, 2.0, 3.0, 7.0, 7.0])

        self.assertTrue(test_hx and test_hy)

    #    def test_h_gridded_updates(self):
    #        mesh = discretize.TreeMesh([8, 8])
    #        mesh.refine(1)
    #
    #        H = mesh.h_gridded
    #        self.assertTrue(np.all(H[:, 0] == 0.5*np.ones(4)))
    #        self.assertTrue(np.all(H[:, 1] == 0.5*np.ones(4)))
    #
    #        # refine the mesh and make sure h_gridded is updated
    #        mesh.refine(2)
    #        H = mesh.h_gridded
    #        self.assertTrue(np.all(H[:, 0] == 0.25*np.ones(16)))
    #        self.assertTrue(np.all(H[:, 1] == 0.25*np.ones(16)))

    def test_faceDiv(self):

        hx, hy = np.r_[1.0, 2, 3, 4], np.r_[5.0, 6, 7, 8]
        T = discretize.TreeMesh([hx, hy], levels=2)
        T.refine(lambda xc: 2)
        # T.plotGrid(show_it=True)
        M = discretize.TensorMesh([hx, hy])
        self.assertEqual(M.nC, T.nC)
        self.assertEqual(M.nF, T.nF)
        self.assertEqual(M.nFx, T.nFx)
        self.assertEqual(M.nFy, T.nFy)
        self.assertEqual(M.nE, T.nE)
        self.assertEqual(M.nEx, T.nEx)
        self.assertEqual(M.nEy, T.nEy)

        self.assertTrue(np.allclose(M.area, T.permuteF * T.area))
        self.assertTrue(np.allclose(M.edge, T.permuteE * T.edge))
        self.assertTrue(np.allclose(M.vol, T.permuteCC * T.vol))

        # plt.subplot(211).spy(M.faceDiv)
        # plt.subplot(212).spy(T.permuteCC*T.faceDiv*T.permuteF.T)
        # plt.show()

        self.assertEqual((M.faceDiv - T.permuteCC * T.faceDiv * T.permuteF.T).nnz, 0)

    def test_serialization(self):
        hx, hy = np.r_[1.0, 2, 3, 4], np.r_[5.0, 6, 7, 8]
        mesh1 = discretize.TreeMesh([hx, hy], levels=2, x0=np.r_[-1, -1])
        mesh1.refine(2)
        mesh2 = discretize.TreeMesh.deserialize(mesh1.serialize())
        self.assertTrue(np.all(mesh1.x0 == mesh2.x0))
        self.assertTrue(np.all(mesh1._n == mesh2._n))
        self.assertTrue(np.all(mesh1.gridCC == mesh2.gridCC))

        mesh1.x0 = np.r_[-2.0, 2]
        mesh2 = discretize.TreeMesh.deserialize(mesh1.serialize())
        self.assertTrue(np.all(mesh1.x0 == mesh2.x0))


class TestOcTree(unittest.TestCase):
    def test_counts(self):
        nc = 8
        h1 = np.random.rand(nc) * nc * 0.5 + nc * 0.5
        h2 = np.random.rand(nc) * nc * 0.5 + nc * 0.5
        h3 = np.random.rand(nc) * nc * 0.5 + nc * 0.5
        h = [hi / np.sum(hi) for hi in [h1, h2, h3]]  # normalize
        M = discretize.TreeMesh(h, levels=3)
        points = np.array([[0.2, 0.1, 0.7], [0.8, 0.4, 0.2]])
        levels = np.array([1, 2])
        M.insert_cells(points, levels)
        M.number()
        # M.plotGrid(show_it=True)
        self.assertEqual(M.nhFx, 4)
        self.assertTrue(M.nFx, 19)
        self.assertTrue(M.nC, 15)

        self.assertTrue(np.allclose(M.vol.sum(), 1.0))

        # self.assertTrue(np.allclose(M._areaFxFull, (M._deflationMatrix('F') * M.area)[:M.ntFx]))
        # self.assertTrue(np.allclose(M._areaFyFull, (M._deflationMatrix('F') * M.area)[M.ntFx:(M.ntFx+M.ntFy)])
        # self.assertTrue(np.allclose(M._areaFzFull, (M._deflationMatrix('F') * M.area)[(M.ntFx+M.ntFy):])

        # self.assertTrue(np.allclose(M._edgeExFull, (M._deflationMatrix('E') * M.edge)[:M.ntEx])
        # self.assertTrue(np.allclose(M._edgeEyFull, (M._deflationMatrix('E') * M.edge)[M.ntEx:(M.ntEx+M.ntEy)])
        # self.assertTrue(np.allclose(M._edgeEzFull, (M._deflationMatrix('E') * M.edge)[(M.ntEx+M.ntEy):]))

    def test_faceDiv(self):

        hx, hy, hz = np.r_[1.0, 2, 3, 4], np.r_[5.0, 6, 7, 8], np.r_[9.0, 10, 11, 12]
        M = discretize.TreeMesh([hx, hy, hz], levels=2)
        M.refine(lambda xc: 2)
        # M.plotGrid(show_it=True)
        Mr = discretize.TensorMesh([hx, hy, hz])
        self.assertEqual(M.nC, Mr.nC)
        self.assertEqual(M.nF, Mr.nF)
        self.assertEqual(M.nFx, Mr.nFx)
        self.assertEqual(M.nFy, Mr.nFy)
        self.assertEqual(M.nE, Mr.nE)
        self.assertEqual(M.nEx, Mr.nEx)
        self.assertEqual(M.nEy, Mr.nEy)

        self.assertTrue(np.allclose(Mr.area, M.permuteF * M.area))
        self.assertTrue(np.allclose(Mr.edge, M.permuteE * M.edge))
        self.assertTrue(np.allclose(Mr.vol, M.permuteCC * M.vol))

        A = Mr.faceDiv - M.permuteCC * M.faceDiv * M.permuteF.T
        self.assertTrue(np.allclose(A.data, 0))

    def test_edgeCurl(self):

        hx, hy, hz = np.r_[1.0, 2, 3, 4], np.r_[5.0, 6, 7, 8], np.r_[9.0, 10, 11, 12]
        M = discretize.TreeMesh([hx, hy, hz], levels=2)
        M.refine(lambda xc: 2)

        Mr = discretize.TensorMesh([hx, hy, hz])

        A = Mr.edgeCurl - M.permuteF * M.edgeCurl * M.permuteE.T

        self.assertTrue(len(A.data) == 0 or np.allclose(A.data, 0))

    def test_faceInnerProduct(self):

        hx, hy, hz = np.r_[1.0, 2, 3, 4], np.r_[5.0, 6, 7, 8], np.r_[9.0, 10, 11, 12]
        # hx, hy, hz = [[(1, 4)], [(1, 4)], [(1, 4)]]

        M = discretize.TreeMesh([hx, hy, hz], levels=2)
        M.refine(lambda xc: 2)
        # M.plotGrid(show_it=True)
        Mr = discretize.TensorMesh([hx, hy, hz])

        # print(M.nC, M.nF, M.getFaceInnerProduct().shape, M.permuteF.shape)
        A_face = (
            Mr.getFaceInnerProduct()
            - M.permuteF * M.getFaceInnerProduct() * M.permuteF.T
        )
        A_edge = (
            Mr.getEdgeInnerProduct()
            - M.permuteE * M.getEdgeInnerProduct() * M.permuteE.T
        )

        self.assertTrue(len(A_face.data) == 0 or np.allclose(A_face.data, 0))
        self.assertTrue(len(A_edge.data) == 0 or np.allclose(A_edge.data, 0))

    def test_VectorIdenties(self):
        hx, hy, hz = [[(1, 4)], [(1, 4)], [(1, 4)]]

        M = discretize.TreeMesh([hx, hy, hz], levels=2)
        Mr = discretize.TensorMesh([hx, hy, hz])
        M.refine(2)  # Why wasn't this here before?

        self.assertTrue(np.allclose((M.faceDiv * M.edgeCurl).data, 0))

        hx, hy, hz = np.r_[1.0, 2, 3, 4], np.r_[5.0, 6, 7, 8], np.r_[9.0, 10, 11, 12]

        M = discretize.TreeMesh([hx, hy, hz], levels=2)
        Mr = discretize.TensorMesh([hx, hy, hz])
        M.refine(2)
        A1 = M.faceDiv * M.edgeCurl
        A2 = Mr.faceDiv * Mr.edgeCurl

        self.assertTrue(len(A1.data) == 0 or np.allclose(A1.data, 0))
        self.assertTrue(len(A2.data) == 0 or np.allclose(A2.data, 0))

    def test_h_gridded_3D(self):
        hx, hy, hz = np.ones(4), np.r_[1.0, 2.0, 3.0, 4.0], 2 * np.ones(4)

        M = discretize.TreeMesh([hx, hy, hz])

        def refinefcn(cell):
            xyz = cell.center
            d = (xyz ** 2).sum() ** 0.5
            if d < 3:
                return 2
            return 1

        M.refine(refinefcn)
        H = M.h_gridded

        test_hx = np.all(
            H[:, 0]
            == np.r_[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
            ]
        )
        test_hy = np.all(
            H[:, 1]
            == np.r_[
                1.0,
                1.0,
                2.0,
                2.0,
                1.0,
                1.0,
                2.0,
                2.0,
                3.0,
                7.0,
                7.0,
                3.0,
                3.0,
                7.0,
                7.0,
            ]
        )
        test_hz = np.all(
            H[:, 2]
            == np.r_[
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
            ]
        )

        self.assertTrue(test_hx and test_hy and test_hz)


class Test2DInterpolation(unittest.TestCase):
    def setUp(self):
        def topo(x):
            return np.sin(x * (2.0 * np.pi)) * 0.3 + 0.5

        def function(cell):
            r = cell.center - np.array([0.5] * len(cell.center))
            dist1 = np.sqrt(r.dot(r)) - 0.08
            dist2 = np.abs(cell.center[-1] - topo(cell.center[0]))

            dist = min([dist1, dist2])
            # if dist < 0.05:
            #     return 5
            if dist < 0.05:
                return 6
            if dist < 0.2:
                return 5
            if dist < 0.3:
                return 4
            if dist < 1.0:
                return 3
            else:
                return 0

        M = discretize.TreeMesh([64, 64], levels=6)
        M.refine(function)
        self.M = M

    def test_fx(self):
        r = np.random.rand(self.M.nFx)
        P = self.M.getInterpolationMat(self.M.gridFx, "Fx")
        self.assertLess(np.abs(P[:, : self.M.nFx] * r - r).max(), TOL)

    def test_fy(self):
        r = np.random.rand(self.M.nFy)
        P = self.M.getInterpolationMat(self.M.gridFy, "Fy")
        self.assertLess(np.abs(P[:, self.M.nFx :] * r - r).max(), TOL)


class Test3DInterpolation(unittest.TestCase):
    def setUp(self):
        def function(cell):
            r = cell.center - np.array([0.5] * len(cell.center))
            dist = np.sqrt(r.dot(r))
            if dist < 0.2:
                return 4
            if dist < 0.3:
                return 3
            if dist < 1.0:
                return 2
            else:
                return 0

        M = discretize.TreeMesh([16, 16, 16], levels=4)
        M.refine(function)
        # M.plotGrid(show_it=True)
        self.M = M

    def test_Fx(self):
        r = np.random.rand(self.M.nFx)
        P = self.M.getInterpolationMat(self.M.gridFx, "Fx")
        self.assertLess(np.abs(P[:, : self.M.nFx] * r - r).max(), TOL)

    def test_Fy(self):
        r = np.random.rand(self.M.nFy)
        P = self.M.getInterpolationMat(self.M.gridFy, "Fy")
        self.assertLess(
            np.abs(P[:, self.M.nFx : (self.M.nFx + self.M.nFy)] * r - r).max(), TOL
        )

    def test_Fz(self):
        r = np.random.rand(self.M.nFz)
        P = self.M.getInterpolationMat(self.M.gridFz, "Fz")
        self.assertLess(np.abs(P[:, (self.M.nFx + self.M.nFy) :] * r - r).max(), TOL)

    def test_Ex(self):
        r = np.random.rand(self.M.nEx)
        P = self.M.getInterpolationMat(self.M.gridEx, "Ex")
        self.assertLess(np.abs(P[:, : self.M.nEx] * r - r).max(), TOL)

    def test_Ey(self):
        r = np.random.rand(self.M.nEy)
        P = self.M.getInterpolationMat(self.M.gridEy, "Ey")
        self.assertLess(
            np.abs(P[:, self.M.nEx : (self.M.nEx + self.M.nEy)] * r - r).max(), TOL
        )

    def test_Ez(self):
        r = np.random.rand(self.M.nEz)
        P = self.M.getInterpolationMat(self.M.gridEz, "Ez")
        self.assertLess(np.abs(P[:, (self.M.nEx + self.M.nEy) :] * r - r).max(), TOL)


if __name__ == "__main__":
    unittest.main()
