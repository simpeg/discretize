import numpy as np
import scipy.sparse as sp
import discretize
from discretize.utils import cart2cyl, cyl2cart


def u(*args):
    if len(args) == 1:
        x = args[0]
        return x**3
    if len(args) == 2:
        x, y = args
        return x**3 + y**2
    x, y, z = args
    return x**3 + y**2 + z**4


def u_cyl(*args):
    xyz = cyl2cart(np.stack(args, axis=-1))
    return u(*xyz.T)


def v(*args):
    if len(args) == 1:
        x = args[0]
        return 2 * x**2
    if len(args) == 2:
        x, y = args
        return np.c_[2 * x**2, 3 * y**3]
    x, y, z = args
    return np.c_[2 * x**2, 3 * y**3, -4 * z**2]


def v_cyl(*args):
    xyz = cyl2cart(np.stack(args, axis=-1))
    xyz_vec = v(*xyz.T)
    return cart2cyl(xyz, xyz_vec)


def w(*args):
    if len(args) == 2:
        x, y = args
        return np.c_[(y - 2) ** 2, (x + 2) ** 2]
    x, y, z = args
    return np.c_[(y - 2) ** 2 + z**2, (x + 2) ** 2 - (z - 4) ** 2, y**2 - x**2]


def w_cyl(*args):
    xyz = cyl2cart(np.stack(args, axis=-1))
    xyz_vec = w(*xyz.T)
    return cart2cyl(xyz, xyz_vec)


# mesh will be on [0, 1] square

# 1D
# int_V grad_u dot v dV = 6/5
# int_V u dot div v dV = 4/5

# 2D
# square vals:
# int_V grad_u dot v dV = 12/5
# int_V u div_v dV = 241/60
# int_v curl_w dot v dV = -173/30

# circle vals:
# int_V grad_u dot  dV = 3*np.pi/2
# int_V u div_v dV = 13*np.pi/8
# int_v curl_w dot v dV = -43*np.pi/8

# 3D square vals:

# int_V grad_u dot v dV = -4/15
# int_V u div_v dV = 27/20
# int_v curl_w dot v dV = 17/6

# 3D Cylindrical Values
# int_V grad_u dot v dV = -7 * np.pi/6
# int_V u div_v dV = -31 * np.pi/120
# int_v curl_w dot v dV = -85 * np.pi/6


class Test1DBoundaryIntegral(discretize.tests.OrderTest):
    name = "1D Boundary Integrals"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 1
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):
        mesh = self.M
        if self.myTest == "cell_grad":
            u_cc = u(mesh.cell_centers)
            v_f = v(mesh.nodes)
            u_bf = u(mesh.boundary_faces)

            D = mesh.face_divergence
            M_c = sp.diags(mesh.cell_volumes)
            M_bf = mesh.boundary_face_scalar_integral

            discrete_val = -(v_f.T @ D.T) @ M_c @ u_cc + v_f.T @ (M_bf @ u_bf)
            true_val = 6 / 5
        if self.myTest == "edge_div":
            u_n = u(mesh.nodes)
            v_e = v(mesh.edges)
            v_bn = v(mesh.boundary_nodes).reshape(-1, order="F")

            M_e = mesh.get_edge_inner_product()
            G = mesh.nodal_gradient
            M_bn = mesh.boundary_node_vector_integral

            discrete_val = -(u_n.T @ G.T) @ M_e @ v_e + u_n.T @ (M_bn @ v_bn)
            true_val = 4 / 5
        return np.abs(discrete_val - true_val)

    def test_orderWeakCellGradIntegral(self):
        self.name = "1D - weak cell gradient integral w/boundary"
        self.myTest = "cell_grad"
        self.orderTest()

    def test_orderWeakEdgeDivIntegral(self):
        self.name = "1D - weak edge divergence integral w/boundary"
        self.myTest = "edge_div"
        self.orderTest()


class Test2DBoundaryIntegral(discretize.tests.OrderTest):
    name = "2D Boundary Integrals"
    meshTypes = [
        "uniformTensorMesh",
        "uniformTree",
        "uniformCurv",
        "rotateCurv",
        "sphereCurv",
    ]
    meshDimension = 2
    expectedOrders = [2, 2, 2, 2, 1]
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):
        mesh = self.M
        if self.myTest == "cell_grad":
            # Functions:
            u_cc = u(*mesh.cell_centers.T)
            v_f = mesh.project_face_vector(v(*mesh.faces.T))
            u_bf = u(*mesh.boundary_faces.T)

            D = mesh.face_divergence
            M_c = sp.diags(mesh.cell_volumes)
            M_bf = mesh.boundary_face_scalar_integral

            discrete_val = -(v_f.T @ D.T) @ M_c @ u_cc + v_f.T @ (M_bf @ u_bf)
            if "sphere" not in self._meshType:
                true_val = 12 / 5
            else:
                true_val = 3 * np.pi / 2
        elif self.myTest == "edge_div":
            u_n = u(*mesh.nodes.T)
            v_e = mesh.project_edge_vector(v(*mesh.edges.T))
            v_bn = v(*mesh.boundary_nodes.T).reshape(-1, order="F")

            M_e = mesh.get_edge_inner_product()
            G = mesh.nodal_gradient
            M_bn = mesh.boundary_node_vector_integral

            discrete_val = -(u_n.T @ G.T) @ M_e @ v_e + u_n.T @ (M_bn @ v_bn)
            if "sphere" not in self._meshType:
                true_val = 241 / 60
            else:
                true_val = 13 * np.pi / 8
        elif self.myTest == "face_curl":
            w_e = mesh.project_edge_vector(w(*mesh.edges.T))
            u_c = u(*mesh.cell_centers.T)
            u_be = u(*mesh.boundary_edges.T)

            M_c = sp.diags(mesh.cell_volumes)
            Curl = mesh.edge_curl
            M_be = mesh.boundary_edge_vector_integral

            discrete_val = (w_e.T @ Curl.T) @ M_c @ u_c - w_e.T @ (M_be @ u_be)
            if "Curv" in self._meshType:
                self._expectedOrder = -1.0
            if "sphere" not in self._meshType:
                true_val = -173 / 30
            else:
                true_val = -43 * np.pi / 8

        return np.abs(discrete_val - true_val)

    def test_orderWeakCellGradIntegral(self):
        self.name = "2D - weak cell gradient integral w/boundary"
        self.myTest = "cell_grad"
        self.orderTest()

    def test_orderWeakEdgeDivIntegral(self):
        self.name = "2D - weak edge divergence integral w/boundary"
        self.myTest = "edge_div"
        self.orderTest()

    def test_orderWeakFaceCurlIntegral(self):
        self.name = "2D - weak face curl integral w/boundary"
        self.myTest = "face_curl"
        self.orderTest()


class Test3DBoundaryIntegral(discretize.tests.OrderTest):
    name = "3D Boundary Integrals"
    meshTypes = [
        "uniformTensorMesh",
        "randomTensorMesh",
        "uniformTree",
        "uniformCurv",
        "rotateCurv",
        "sphereCurv",
    ]
    meshDimension = 3
    expectedOrders = [2, 1, 2, 2, 2, 2]
    meshSizes = [4, 8, 16, 32]
    rng = np.random.default_rng(57681234)

    def getError(self):
        mesh = self.M
        if self.myTest == "cell_grad":
            # Functions:
            u_cc = u(*mesh.cell_centers.T)
            v_f = mesh.project_face_vector(v(*mesh.faces.T))
            u_bf = u(*mesh.boundary_faces.T)

            D = mesh.face_divergence
            M_c = sp.diags(mesh.cell_volumes)
            M_bf = mesh.boundary_face_scalar_integral

            discrete_val = -(v_f.T @ D.T) @ M_c @ u_cc + v_f.T @ (M_bf @ u_bf)
            if "sphere" not in self._meshType:
                true_val = -4 / 15
            else:
                true_val = 48 * np.pi / 35
        elif self.myTest == "edge_div":
            u_n = u(*mesh.nodes.T)
            v_e = mesh.project_edge_vector(v(*mesh.edges.T))
            v_bn = v(*mesh.boundary_nodes.T).reshape(-1, order="F")

            M_e = mesh.get_edge_inner_product()
            G = mesh.nodal_gradient
            M_bn = mesh.boundary_node_vector_integral

            discrete_val = -(u_n.T @ G.T) @ M_e @ v_e + u_n.T @ (M_bn @ v_bn)
            if "sphere" not in self._meshType:
                true_val = 27 / 20
            else:
                true_val = 8 * np.pi / 5
        elif self.myTest == "face_curl":
            w_f = mesh.project_face_vector(w(*mesh.faces.T))
            v_e = mesh.project_edge_vector(v(*mesh.edges.T))
            w_be = w(*mesh.boundary_edges.T).reshape(-1, order="F")

            M_f = mesh.get_face_inner_product()
            Curl = mesh.edge_curl
            M_be = mesh.boundary_edge_vector_integral

            discrete_val = (v_e.T @ Curl.T) @ M_f @ w_f - v_e.T @ (M_be @ w_be)
            if "sphere" not in self._meshType:
                true_val = -79 / 6
            else:
                true_val = -64 * np.pi / 5

        return np.abs(discrete_val - true_val)

    def test_orderWeakCellGradIntegral(self):
        self.name = "3D - weak cell gradient integral w/boundary"
        self.myTest = "cell_grad"
        self.orderTest()

    def test_orderWeakEdgeDivIntegral(self):
        self.name = "3D - weak edge divergence integral w/boundary"
        self.myTest = "edge_div"
        self.orderTest()

    def test_orderWeakFaceCurlIntegral(self):
        self.name = "3D - weak face curl integral w/boundary"
        self.myTest = "face_curl"
        self.orderTest()
