import numpy as np
import discretize
from discretize.utils import example_simplex_mesh


class TestInterpolation2d(discretize.tests.OrderTest):
    name = "Interpolation 2D"
    meshSizes = [8, 16, 32, 64]
    meshTypes = ['uniform simplex mesh']
    interp_points = np.random.rand(200, 2) * 0.9 + 0.1
    meshDimension = 2
    expectedOrders = 1

    def setupMesh(self, n):
        points, simplices = example_simplex_mesh((n, n))
        self.M = discretize.SimplexMesh(points, simplices)
        return 1.0 / n

    def getError(self):
        funX = lambda x, y: -y**2
        funY = lambda x, y: x**2

        if "x" in self.type:
            ana = funX(*self.interp_points.T)
        elif "y" in self.type:
            ana = funY(*self.interp_points.T)
        else:
            ana = funX(*self.interp_points.T) + funY(*self.interp_points.T)

        mesh = self.M
        if "F" in self.type:
            Fc = np.c_[funX(*mesh.faces.T), funY(*mesh.faces.T)]
            grid = mesh.project_face_vector(Fc)
        elif "E" in self.type:
            Ec = np.c_[funX(*mesh.edges.T), funY(*mesh.edges.T)]
            grid = mesh.project_edge_vector(Ec)
        elif "CC" == self.type:
            grid = funX(*mesh.cell_centers.T) + funY(*mesh.cell_centers.T)
        elif "N" == self.type:
            grid = funX(*mesh.nodes.T) + funY(*mesh.nodes.T)

        comp = mesh.get_interpolation_matrix(self.interp_points, self.type) * grid

        err = np.linalg.norm((comp - ana), np.inf)
        return err

    def test_orderCC(self):
        self.type = "CC"
        self.name = "Interpolation 2D: CC"
        self.orderTest()

    def test_orderN(self):
        self.type = "N"
        self.expectedOrders = 2
        self.name = "Interpolation 2D: N"
        self.orderTest()
        self.expectedOrders = 1

    def test_orderFx(self):
        self.type = "Fx"
        self.name = "Interpolation 2D: Fx"
        self.orderTest()

    def test_orderFy(self):
        self.type = "Fy"
        self.name = "Interpolation 2D: Fy"
        self.orderTest()

    def test_orderEx(self):
        self.type = "Ex"
        self.name = "Interpolation 2D: Ex"
        self.orderTest()

    def test_orderEy(self):
        self.type = "Ey"
        self.name = "Interpolation 2D: Ey"
        self.orderTest()

class TestInterpolation3d(discretize.tests.OrderTest):
    name = "Interpolation 3D"
    meshSizes = [8, 16, 32]
    meshTypes = ['uniform simplex mesh']
    interp_points = np.random.rand(200, 3) * 0.9 + 0.1
    meshDimension = 3
    expectedOrders = 1

    def setupMesh(self, n):
        points, simplices = example_simplex_mesh((n, n, n))
        self.M = discretize.SimplexMesh(points, simplices)
        return 1.0 / n

    def getError(self):
        funX = lambda x, y, z: np.cos(2 * np.pi * y)
        funY = lambda x, y, z: np.cos(2 * np.pi * z)
        funZ = lambda x, y, z: np.cos(2 * np.pi * x)

        if "x" in self.type:
            ana = funX(*self.interp_points.T)
        elif "y" in self.type:
            ana = funY(*self.interp_points.T)
        elif "z" in self.type:
            ana = funZ(*self.interp_points.T)
        else:
            ana = (
                funX(*self.interp_points.T)
                + funY(*self.interp_points.T)
                + funZ(*self.interp_points.T)
            )

        mesh = self.M
        if "F" in self.type:
            Fc = np.c_[funX(*mesh.faces.T), funY(*mesh.faces.T), funZ(*mesh.faces.T)]
            grid = mesh.project_face_vector(Fc)
        elif "E" in self.type:
            Ec = np.c_[funX(*mesh.edges.T), funY(*mesh.edges.T), funZ(*mesh.edges.T)]
            grid = mesh.project_edge_vector(Ec)
        elif "CC" == self.type:
            grid = (
                funX(*mesh.cell_centers.T)
                + funY(*mesh.cell_centers.T)
                + funZ(*mesh.cell_centers.T)
            )
        elif "N" == self.type:
            grid = (
                funX(*mesh.nodes.T)
                + funY(*mesh.nodes.T)
                + funZ(*mesh.nodes.T)
            )

        comp = mesh.get_interpolation_matrix(self.interp_points, self.type) * grid

        err = np.linalg.norm((comp - ana), np.inf)
        return err

    def test_orderCC(self):
        self.type = "CC"
        self.name = "Interpolation 3D: CC"
        self.orderTest()

    def test_orderN(self):
        self.type = "N"
        self.expectedOrders = 2
        self.name = "Interpolation 3D: N"
        self.orderTest()
        self.expectedOrders = 1

    def test_orderFx(self):
        self.type = "Fx"
        self.name = "Interpolation 3D: Fx"
        self.orderTest()

    def test_orderFy(self):
        self.type = "Fy"
        self.name = "Interpolation 3D: Fy"
        self.orderTest()

    def test_orderFz(self):
        self.type = "Fz"
        self.name = "Interpolation 3D: Fy"
        self.orderTest()

    def test_orderEx(self):
        self.type = "Ex"
        self.name = "Interpolation 3D: Ex"
        self.orderTest()

    def test_orderEy(self):
        self.type = "Ey"
        self.name = "Interpolation 3D: Ey"
        self.orderTest()

    def test_orderEz(self):
        self.type = "Ez"
        self.name = "Interpolation 3D: Ey"
        self.orderTest()
