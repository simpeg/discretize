import numpy as np
import discretize
from discretize.utils import example_simplex_mesh


class TestInterpolation2d(discretize.tests.OrderTest):
    name = "Interpolation 2D"
    meshSizes = [8, 16, 32, 64]
    meshTypes = ['uniform simplex mesh']
    interp_points = np.stack(np.mgrid[
        0.25:0.75:32j, 0.25:0.75:32j
    ], axis=-1).reshape(-1, 2)
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
        self.expectedOrders = 2
        self.name = "Interpolation 2D: CC"
        self.expectedOrders = 1
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
    meshSizes = [5, 10, 20, 40]
    meshTypes = ['uniform simplex mesh']
    interp_points = np.stack(np.mgrid[
        0.25:0.75:32j, 0.25:0.75:32j, 0.25:0.75:32j
    ], axis=-1).reshape(-1, 3)
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
        self.expectedOrders = 2
        self.name = "Interpolation 3D: CC"
        self.orderTest()
        self.expectedOrders = 1

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
        self.name = "Interpolation 3D: Fz"
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
        self.name = "Interpolation 3D: Ez"
        self.orderTest()


class TestAveraging(discretize.tests.OrderTest):
    meshSizes = [8, 16, 32]
    meshTypes = ['uniform simplex mesh']

    def setupMesh(self, n):
        dim = self.meshDimension
        points, simplices = example_simplex_mesh(dim * (n, ) )
        self.M = discretize.SimplexMesh(points, simplices)
        return 1.0 / n

    def getError(self):
        mesh = self.M
        if mesh.dim == 2:
            func = lambda x, y: np.cos(x**2) - np.sin(y**2) + 2*x*y
        else:
            func = lambda x, y, z: np.cos(x**2) - np.sin(y**2) + 2*x*y + np.cos(2*z)**2 + z**3


        if self.target_type == "CC":
            interp_points = mesh.cell_centers
        elif self.target_type == "N":
            interp_points = mesh.nodes
        elif self.target_type == "E":
            interp_points = mesh.edges
        elif self.target_type == "F":
            interp_points = mesh.faces
        ana = func(*interp_points.T)

        if self.source_type == "CC":
            source_points = mesh.cell_centers
        elif self.source_type == "N":
            source_points = mesh.nodes
        elif self.source_type == "E":
            source_points = mesh.edges
        elif self.source_type == "F":
            source_points = mesh.faces

        v = func(*source_points.T)

        if self.source_type == "N":
            if self.target_type == "CC":
                avg_mat = mesh.average_node_to_cell
            elif self.target_type == "E":
                avg_mat = mesh.average_node_to_edge
            elif self.target_type == "F":
                avg_mat = mesh.average_node_to_face
        elif self.source_type == "CC":
            if self.target_type == "N":
                avg_mat = mesh.average_cell_to_node
            elif self.target_type == "E":
                avg_mat = mesh.average_cell_to_edge
            elif self.target_type == "F":
                avg_mat = mesh.average_cell_to_face
        elif self.source_type == "E":
            if self.target_type == "CC":
                avg_mat = mesh.average_edge_to_cell
            elif self.target_type == "F":
                avg_mat = mesh.average_edge_to_face
        elif self.source_type == "F":
            if self.target_type == "CC":
                avg_mat = mesh.average_face_to_cell

        comp = avg_mat @ v

        err = np.linalg.norm((comp - ana), np.inf)
        return err

    # 2D
    def test_AvgN2CC_2D(self):
        self.source_type = "N"
        self.target_type = "CC"
        self.name = "average_node_to_cell 2D"
        self.meshDimension = 2
        self.expectedOrders = 2
        self.orderTest()

    def test_AvgN2F_2D(self):
        self.source_type = "N"
        self.target_type = "F"
        self.name = "average_node_to_face 2D"
        self.meshDimension = 2
        self.expectedOrders = 2
        self.orderTest()

    def test_AvgN2E_2D(self):
        self.source_type = "N"
        self.target_type = "E"
        self.name = "average_node_to_edge 2D"
        self.meshDimension = 2
        self.expectedOrders = 2
        self.orderTest()

    def test_AvgCC2E_2D(self):
        self.source_type = "CC"
        self.target_type = "E"
        self.name = "average_cell_to_edge 2D"
        self.meshDimension = 2
        self.expectedOrders = 1
        self.orderTest()

    def test_AvgE2CC_2D(self):
        self.source_type = "E"
        self.target_type = "CC"
        self.name = "average_edge_to_cell 2D"
        self.meshDimension = 2
        self.expectedOrders = 2
        self.orderTest()

    def test_AvgF2CC_2D(self):
        self.source_type = "F"
        self.target_type = "CC"
        self.name = "average_face_to_cell 2D"
        self.meshDimension = 2
        self.expectedOrders = 2
        self.orderTest()

    def test_AvgCC2F_2D(self):
        self.source_type = "CC"
        self.target_type = "F"
        self.name = "average_cell_to_face 2D"
        self.meshDimension = 2
        self.expectedOrders = 1
        self.orderTest()

    # 3D
    def test_AvgN2CC_3D(self):
        self.source_type = "N"
        self.target_type = "CC"
        self.name = "average_node_to_cell 3D"
        self.meshDimension = 3
        self.expectedOrders = 2
        self.orderTest()

    def test_AvgN2F_3D(self):
        self.source_type = "N"
        self.target_type = "F"
        self.name = "average_node_to_face 3D"
        self.meshDimension = 3
        self.expectedOrders = 2
        self.orderTest()

    def test_AvgN2E_3D(self):
        self.source_type = "N"
        self.target_type = "E"
        self.name = "average_node_to_edge 3D"
        self.meshDimension = 3
        self.expectedOrders = 2
        self.orderTest()

    def test_AvgCC2N_3D(self):
        self.source_type = "CC"
        self.target_type = "N"
        self.name = "average_cell_to_node 3D"
        self.meshDimension = 3
        self.expectedOrders = 1
        self.orderTest()

    def test_AvgE2CC_3D(self):
        self.source_type = "E"
        self.target_type = "CC"
        self.name = "average_edge_to_cell 3D"
        self.meshDimension = 3
        self.expectedOrders = 2
        self.orderTest()

    def test_AvgF2CC_3D(self):
        self.source_type = "F"
        self.target_type = "CC"
        self.name = "average_face_to_cell 3D"
        self.meshDimension = 3
        self.expectedOrders = 2
        self.orderTest()

    def test_AvgCC2F_3D(self):
        self.source_type = "CC"
        self.target_type = "F"
        self.name = "average_cell_to_face 3D"
        self.meshDimension = 3
        self.expectedOrders = 1
        self.orderTest()


class TestVectorAveraging2D(discretize.tests.OrderTest):
    name = "Averaging 2D"
    meshSizes = [8, 16, 32, 64]
    meshTypes = ['uniform simplex mesh']
    meshDimension = 2
    expectedOrders = 1

    def setupMesh(self, n):
        points, simplices = example_simplex_mesh((n, n))
        self.M = discretize.SimplexMesh(points, simplices)
        return 1.0 / n


    def getError(self):
        funX = lambda x, y: -y**2
        funY = lambda x, y: x**2
        mesh = self.M
        ana = np.c_[funX(*mesh.cell_centers.T), funY(*mesh.cell_centers.T)].reshape(-1, order='F')

        if self.source_type == "F":
            Fc = np.c_[funX(*mesh.faces.T), funY(*mesh.faces.T)]
            grid = mesh.project_face_vector(Fc)
            comp = mesh.average_face_to_cell_vector @ grid
        elif self.source_type == "E":
            Ec = np.c_[funX(*mesh.edges.T), funY(*mesh.edges.T)]
            grid = mesh.project_edge_vector(Ec)
            comp = mesh.average_edge_to_cell_vector @ grid

        err = np.linalg.norm((comp - ana), np.inf)
        return err

    def test_AvgE2CCV(self):
        self.source_type = "E"
        self.name = "average_edge_to_cell_vector 2D"
        self.orderTest()

    def test_AvgF2CCV(self):
        self.source_type = "F"
        self.name = "average_face_to_cell_vector 2D"
        self.orderTest()


class TestVectorAveraging3D(discretize.tests.OrderTest):
    name = "Averaging 3D"
    meshSizes = [8, 16, 32]
    meshTypes = ['uniform simplex mesh']
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
        mesh = self.M
        ana = np.c_[funX(*mesh.cell_centers.T), funY(*mesh.cell_centers.T), funZ(*mesh.cell_centers.T)].reshape(-1, order='F')

        if self.source_type == "F":
            Fc = np.c_[funX(*mesh.faces.T), funY(*mesh.faces.T), funZ(*mesh.faces.T)]
            grid = mesh.project_face_vector(Fc)
            comp = mesh.average_face_to_cell_vector @ grid
        elif self.source_type == "E":
            Ec = np.c_[funX(*mesh.edges.T), funY(*mesh.edges.T), funZ(*mesh.edges.T)]
            grid = mesh.project_edge_vector(Ec)
            comp = mesh.average_edge_to_cell_vector @ grid

        err = np.linalg.norm((comp - ana), np.inf)
        return err

    def test_AvgE2CCV(self):
        self.source_type = "E"
        self.name = "average_edge_to_cell_vector 2D"
        self.expectedOrders = 1
        self.orderTest()

    def test_AvgF2CCV(self):
        self.source_type = "F"
        self.name = "average_face_to_cell_vector 2D"
        self.expectedOrders = 1
        self.orderTest()


def test_cell_to_face_extrap():
    # 2D
    points, simplices = example_simplex_mesh((10, 10))
    mesh = discretize.SimplexMesh(points, simplices)

    v = np.ones(len(mesh))

    Fv = mesh.average_cell_to_face @ v

    np.testing.assert_equal(1.0, Fv)

    # 3D
    points, simplices = example_simplex_mesh((4, 4, 4))
    mesh = discretize.SimplexMesh(points, simplices)

    v = np.ones(len(mesh))

    Fv = mesh.average_cell_to_face @ v

    np.testing.assert_equal(1.0, Fv)
