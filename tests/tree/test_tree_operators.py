import numpy as np
import discretize

MESHTYPES = ["uniformTree", "randomTree"]
# MESHTYPES = ['randomTree']
call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1])
call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cart_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cart_row3 = lambda g, xfun, yfun, zfun: np.c_[
    call3(xfun, g), call3(yfun, g), call3(zfun, g)
]
cartF2 = lambda M, fx, fy: np.vstack(
    (cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFy, fx, fy))
)
cartE2 = lambda M, ex, ey: np.vstack(
    (cart_row2(M.gridEx, ex, ey), cart_row2(M.gridEy, ex, ey))
)
cartF3 = lambda M, fx, fy, fz: np.vstack(
    (
        cart_row3(M.gridFx, fx, fy, fz),
        cart_row3(M.gridFy, fx, fy, fz),
        cart_row3(M.gridFz, fx, fy, fz),
    )
)
cartE3 = lambda M, ex, ey, ez: np.vstack(
    (
        cart_row3(M.gridEx, ex, ey, ez),
        cart_row3(M.gridEy, ex, ey, ez),
        cart_row3(M.gridEz, ex, ey, ez),
    )
)


class TestCellGrad2D(discretize.tests.OrderTest):
    name = "Cell Gradient 2D, using cellGradx and cellGrady"
    meshTypes = MESHTYPES
    meshDimension = 2
    meshSizes = [8, 16]
    # because of the averaging involved in the ghost point. u_b = (u_n + u_g)/2
    expectedOrders = 1

    def getError(self):
        # Test function
        sol = lambda x, y: np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
        fx = lambda x, y: -2 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        fy = lambda x, y: -2 * np.pi * np.sin(2 * np.pi * y) * np.cos(2 * np.pi * x)

        phi = call2(sol, self.M.gridCC)
        gradF = self.M.cell_gradient * phi
        Fc = cartF2(self.M, fx, fy)
        gradF_ana = self.M.project_face_vector(Fc)

        err = np.linalg.norm((gradF - gradF_ana), np.inf)

        return err

    def test_order(self):
        self.orderTest(random_seed=421)


class TestCellGrad3D(discretize.tests.OrderTest):
    name = "Cell Gradient 3D, using cellGradx, cellGrady, and cellGradz"
    meshTypes = MESHTYPES
    meshDimension = 3
    meshSizes = [8, 16]
    # because of the averaging involved in the ghost point. u_b = (u_n + u_g)/2
    expectedOrders = 1

    def getError(self):
        # Test function
        sol = (
            lambda x, y, z: np.cos(2 * np.pi * x)
            * np.cos(2 * np.pi * y)
            * np.cos(2 * np.pi * z)
        )
        fx = (
            lambda x, y, z: -2
            * np.pi
            * np.sin(2 * np.pi * x)
            * np.cos(2 * np.pi * y)
            * np.cos(2 * np.pi * z)
        )
        fy = (
            lambda x, y, z: -2
            * np.pi
            * np.cos(2 * np.pi * x)
            * np.sin(2 * np.pi * y)
            * np.cos(2 * np.pi * z)
        )
        fz = (
            lambda x, y, z: -2
            * np.pi
            * np.cos(2 * np.pi * x)
            * np.cos(2 * np.pi * y)
            * np.sin(2 * np.pi * z)
        )
        phi = call3(sol, self.M.gridCC)
        gradF = self.M.cell_gradient * phi
        Fc = cartF3(self.M, fx, fy, fz)
        gradF_ana = self.M.project_face_vector(Fc)

        err = np.linalg.norm((gradF - gradF_ana), np.inf)

        return err

    def test_order(self):
        self.orderTest(5532)


class TestFaceDivxy2D(discretize.tests.OrderTest):
    name = "Face Divergence 2D, Testing faceDivx and faceDivy"
    meshTypes = MESHTYPES
    meshDimension = 2
    meshSizes = [16, 32]

    def getError(self):
        # Test function
        fx = lambda x, y: np.sin(2 * np.pi * x)
        fy = lambda x, y: np.sin(2 * np.pi * y)
        sol = lambda x, y: 2 * np.pi * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

        Fx = call2(fx, self.M.gridFx)
        Fy = call2(fy, self.M.gridFy)
        divFx = self.M.face_x_divergence.dot(Fx)
        divFy = self.M.face_y_divergence.dot(Fy)
        divF = divFx + divFy

        divF_ana = call2(sol, self.M.gridCC)

        err = np.linalg.norm((divF - divF_ana), np.inf)

        # self.M.plot_image(divF-divF_ana, show_it=True)

        return err

    def test_order(self):
        self.orderTest(random_seed=19647823)


class TestFaceDiv3D(discretize.tests.OrderTest):
    name = "Face Divergence 3D"
    meshTypes = MESHTYPES
    meshSizes = [8, 16, 32]

    def getError(self):
        fx = lambda x, y, z: np.sin(2 * np.pi * x)
        fy = lambda x, y, z: np.sin(2 * np.pi * y)
        fz = lambda x, y, z: np.sin(2 * np.pi * z)
        sol = lambda x, y, z: (
            2 * np.pi * np.cos(2 * np.pi * x)
            + 2 * np.pi * np.cos(2 * np.pi * y)
            + 2 * np.pi * np.cos(2 * np.pi * z)
        )

        Fc = cartF3(self.M, fx, fy, fz)
        F = self.M.project_face_vector(Fc)

        divF = self.M.face_divergence.dot(F)
        divF_ana = call3(sol, self.M.gridCC)

        return np.linalg.norm((divF - divF_ana), np.inf)

    def test_order(self):
        self.orderTest(random_seed=81725364)


class TestFaceDivxyz3D(discretize.tests.OrderTest):
    name = "Face Divergence 3D, Testing faceDivx, faceDivy, and faceDivz"
    meshTypes = MESHTYPES
    meshDimension = 3
    meshSizes = [8, 16, 32]

    def getError(self):
        # Test function
        fx = lambda x, y, z: np.sin(2 * np.pi * x)
        fy = lambda x, y, z: np.sin(2 * np.pi * y)
        fz = lambda x, y, z: np.sin(2 * np.pi * z)
        sol = lambda x, y, z: (
            2 * np.pi * np.cos(2 * np.pi * x)
            + 2 * np.pi * np.cos(2 * np.pi * y)
            + 2 * np.pi * np.cos(2 * np.pi * z)
        )

        Fx = call3(fx, self.M.gridFx)
        Fy = call3(fy, self.M.gridFy)
        Fz = call3(fz, self.M.gridFz)
        divFx = self.M.face_x_divergence.dot(Fx)
        divFy = self.M.face_y_divergence.dot(Fy)
        divFz = self.M.face_z_divergence.dot(Fz)
        divF = divFx + divFy + divFz

        divF_ana = call3(sol, self.M.gridCC)

        err = np.linalg.norm((divF - divF_ana), np.inf)

        # self.M.plot_image(divF-divF_ana, show_it=True)

        return err

    def test_order(self):
        self.orderTest(random_seed=6172824)


class TestCurl(discretize.tests.OrderTest):
    name = "Curl"
    meshTypes = ["notatreeTree", "uniformTree"]
    meshSizes = [8, 16]  # , 32]
    expectedOrders = [2, 1]  # This is due to linear interpolation in the Re projection

    def getError(self):
        # fun: i (cos(y)) + j (cos(z)) + k (cos(x))
        # sol: i (sin(z)) + j (sin(x)) + k (sin(y))

        funX = lambda x, y, z: np.cos(2 * np.pi * y)
        funY = lambda x, y, z: np.cos(2 * np.pi * z)
        funZ = lambda x, y, z: np.cos(2 * np.pi * x)

        solX = lambda x, y, z: 2 * np.pi * np.sin(2 * np.pi * z)
        solY = lambda x, y, z: 2 * np.pi * np.sin(2 * np.pi * x)
        solZ = lambda x, y, z: 2 * np.pi * np.sin(2 * np.pi * y)

        Ec = cartE3(self.M, funX, funY, funZ)
        E = self.M.project_edge_vector(Ec)

        Fc = cartF3(self.M, solX, solY, solZ)
        curlE_ana = self.M.project_face_vector(Fc)

        curlE = self.M.edge_curl.dot(E)

        err = np.linalg.norm((curlE - curlE_ana), np.inf)
        # err = np.linalg.norm((curlE - curlE_ana)*self.M.face_areas, 2)

        return err

    def test_order(self):
        self.orderTest()


class TestNodalGrad(discretize.tests.OrderTest):
    name = "Nodal Gradient"
    meshTypes = ["notatreeTree", "uniformTree"]
    meshSizes = [8, 16]  # , 32]
    expectedOrders = [2, 1]

    def getError(self):
        # Test function
        fun = lambda x, y, z: (np.cos(x) + np.cos(y) + np.cos(z))
        # i (sin(x)) + j (sin(y)) + k (sin(z))
        solX = lambda x, y, z: -np.sin(x)
        solY = lambda x, y, z: -np.sin(y)
        solZ = lambda x, y, z: -np.sin(z)

        phi = call3(fun, self.M.gridN)
        gradE = self.M.nodal_gradient.dot(phi)

        Ec = cartE3(self.M, solX, solY, solZ)
        gradE_ana = self.M.project_edge_vector(Ec)

        err = np.linalg.norm((gradE - gradE_ana), np.inf)

        return err

    def test_order(self):
        self.orderTest()


class TestNodalGrad2D(discretize.tests.OrderTest):
    name = "Nodal Gradient 2D"
    meshTypes = ["notatreeTree", "uniformTree"]
    meshSizes = [8, 16]  # , 32]
    expectedOrders = [2, 1]
    meshDimension = 2

    def getError(self):
        # Test function
        fun = lambda x, y: (np.cos(x) + np.cos(y))
        # i (sin(x)) + j (sin(y)) + k (sin(z))
        solX = lambda x, y: -np.sin(x)
        solY = lambda x, y: -np.sin(y)

        phi = call2(fun, self.M.gridN)
        gradE = self.M.nodal_gradient.dot(phi)

        Ec = cartE2(self.M, solX, solY)
        gradE_ana = self.M.project_edge_vector(Ec)

        err = np.linalg.norm((gradE - gradE_ana), np.inf)

        return err

    def test_order(self):
        self.orderTest()


class TestTreeInnerProducts(discretize.tests.OrderTest):
    """Integrate an function over a unit cube domain using edgeInnerProducts and faceInnerProducts."""

    meshTypes = [
        "uniformTree",
        "notatreeTree",
    ]  # ['uniformTensorMesh', 'uniformCurv', 'rotateCurv']
    meshDimension = 3
    meshSizes = [4, 8]

    def getError(self):
        call = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])

        ex = lambda x, y, z: x**2 + y * z
        ey = lambda x, y, z: (z**2) * x + y * z
        ez = lambda x, y, z: y**2 + x * z

        sigma1 = lambda x, y, z: x * y + 1
        sigma2 = lambda x, y, z: x * z + 2
        sigma3 = lambda x, y, z: 3 + z * y
        sigma4 = lambda x, y, z: 0.1 * x * y * z
        sigma5 = lambda x, y, z: 0.2 * x * y
        sigma6 = lambda x, y, z: 0.1 * z

        Gc = self.M.gridCC
        if self.sigmaTest == 1:
            sigma = np.c_[call(sigma1, Gc)]
            analytic = 647.0 / 360  # Found using sympy.
        elif self.sigmaTest == 3:
            sigma = np.r_[call(sigma1, Gc), call(sigma2, Gc), call(sigma3, Gc)]
            analytic = 37.0 / 12  # Found using sympy.
        elif self.sigmaTest == 6:
            sigma = np.c_[
                call(sigma1, Gc),
                call(sigma2, Gc),
                call(sigma3, Gc),
                call(sigma4, Gc),
                call(sigma5, Gc),
                call(sigma6, Gc),
            ]
            analytic = 69881.0 / 21600  # Found using sympy.

        if self.location == "edges":
            cart = lambda g: np.c_[call(ex, g), call(ey, g), call(ez, g)]
            Ec = np.vstack(
                (cart(self.M.gridEx), cart(self.M.gridEy), cart(self.M.gridEz))
            )
            E = self.M.project_edge_vector(Ec)

            if self.invert_model:
                A = self.M.get_edge_inner_product(
                    discretize.utils.inverse_property_tensor(self.M, sigma),
                    invert_model=True,
                )
            else:
                A = self.M.get_edge_inner_product(sigma)
            numeric = E.T.dot(A.dot(E))
        elif self.location == "faces":
            cart = lambda g: np.c_[call(ex, g), call(ey, g), call(ez, g)]
            Fc = np.vstack(
                (cart(self.M.gridFx), cart(self.M.gridFy), cart(self.M.gridFz))
            )
            F = self.M.project_face_vector(Fc)

            if self.invert_model:
                A = self.M.get_face_inner_product(
                    discretize.utils.inverse_property_tensor(self.M, sigma),
                    invert_model=True,
                )
            else:
                A = self.M.get_face_inner_product(sigma)
            numeric = F.T.dot(A.dot(F))

        err = np.abs(numeric - analytic)
        return err

    def test_order1_edges(self):
        self.name = "Edge Inner Product - Isotropic"
        self.location = "edges"
        self.sigmaTest = 1
        self.invert_model = False
        self.orderTest()

    def test_order1_edges_invert_model(self):
        self.name = "Edge Inner Product - Isotropic - invert_model"
        self.location = "edges"
        self.sigmaTest = 1
        self.invert_model = True
        self.orderTest()

    def test_order3_edges(self):
        self.name = "Edge Inner Product - Anisotropic"
        self.location = "edges"
        self.sigmaTest = 3
        self.invert_model = False
        self.orderTest()

    def test_order3_edges_invert_model(self):
        self.name = "Edge Inner Product - Anisotropic - invert_model"
        self.location = "edges"
        self.sigmaTest = 3
        self.invert_model = True
        self.orderTest()

    def test_order6_edges(self):
        self.name = "Edge Inner Product - Full Tensor"
        self.location = "edges"
        self.sigmaTest = 6
        self.invert_model = False
        self.orderTest()

    def test_order6_edges_invert_model(self):
        self.name = "Edge Inner Product - Full Tensor - invert_model"
        self.location = "edges"
        self.sigmaTest = 6
        self.invert_model = True
        self.orderTest()

    def test_order1_faces(self):
        self.name = "Face Inner Product - Isotropic"
        self.location = "faces"
        self.sigmaTest = 1
        self.invert_model = False
        self.orderTest()

    def test_order1_faces_invert_model(self):
        self.name = "Face Inner Product - Isotropic - invert_model"
        self.location = "faces"
        self.sigmaTest = 1
        self.invert_model = True
        self.orderTest()

    def test_order3_faces(self):
        self.name = "Face Inner Product - Anisotropic"
        self.location = "faces"
        self.sigmaTest = 3
        self.invert_model = False
        self.orderTest()

    def test_order3_faces_invert_model(self):
        self.name = "Face Inner Product - Anisotropic - invert_model"
        self.location = "faces"
        self.sigmaTest = 3
        self.invert_model = True
        self.orderTest()

    def test_order6_faces(self):
        self.name = "Face Inner Product - Full Tensor"
        self.location = "faces"
        self.sigmaTest = 6
        self.invert_model = False
        self.orderTest()

    def test_order6_faces_invert_model(self):
        self.name = "Face Inner Product - Full Tensor - invert_model"
        self.location = "faces"
        self.sigmaTest = 6
        self.invert_model = True
        self.orderTest()


class TestInnerProductsFaceProperties3D(discretize.tests.OrderTest):
    """Integrate a function over a surface within a unit cube domain
    using edgeInnerProducts and faceInnerProducts."""

    meshTypes = ["uniformTree", "notatreeTree"]
    meshDimension = 3
    meshSizes = [8, 16]

    def getError(self):
        call = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])

        ex = lambda x, y, z: x**2 + y * z
        ey = lambda x, y, z: (z**2) * x + y * z
        ez = lambda x, y, z: y**2 + x * z

        tau_x = lambda x, y, z: y * z + 1  # x-face properties
        tau_y = lambda x, y, z: x * z + 2  # y-face properties
        tau_z = lambda x, y, z: 3 + x * y  # z-face properties
        tau_funcs = [tau_x, tau_y, tau_z]

        tau = 3 * [None]
        for ii, comp in enumerate(["x", "y", "z"]):
            faces = getattr(self.M, f"faces_{comp}")
            k = np.isclose(faces[:, ii], 0.5)  # x, y or z location for each plane
            tau_ii = 1e-8 * np.ones(len(faces))  # effectively zeros but stable
            tau_ii[k] = tau_funcs[ii](*faces[k].T)
            tau[ii] = tau_ii
        tau = np.hstack(tau)

        # integrate components parallel to the plane of integration
        if self.location == "edges":
            analytic = 5.02760416666667  # Found using sympy.

            cart = lambda g: np.c_[call(ex, g), call(ey, g), call(ez, g)]

            Ec = np.vstack(
                (cart(self.M.gridEx), cart(self.M.gridEy), cart(self.M.gridEz))
            )
            E = self.M.project_edge_vector(Ec)

            if self.invert_model:
                A = self.M.get_edge_inner_product_surface(1 / tau, invert_model=True)
            else:
                A = self.M.get_edge_inner_product_surface(tau)

            numeric = E.T.dot(A.dot(E))

        # integrate component normal to the plane of integration
        elif self.location == "faces":
            analytic = 2.66979166666667  # Found using sympy.

            cart = lambda g: np.c_[call(ex, g), call(ey, g), call(ez, g)]

            Fc = np.vstack(
                (cart(self.M.gridFx), cart(self.M.gridFy), cart(self.M.gridFz))
            )
            F = self.M.project_face_vector(Fc)

            if self.invert_model:
                A = self.M.get_face_inner_product_surface(1 / tau, invert_model=True)
            else:
                A = self.M.get_face_inner_product_surface(tau)

            numeric = F.T.dot(A.dot(F))

        err = np.abs(numeric - analytic)

        return err

    def test_order1_edges(self):
        self.name = "Edge Inner Product - Isotropic"
        self.location = "edges"
        self.invert_model = False
        self.orderTest()

    def test_order1_edges_invert_model(self):
        self.name = "Edge Inner Product - Isotropic - invert_model"
        self.location = "edges"
        self.invert_model = True
        self.orderTest()

    def test_order1_faces(self):
        self.name = "Face Inner Product - Isotropic"
        self.location = "faces"
        self.invert_model = False
        self.orderTest()

    def test_order1_faces_invert_model(self):
        self.name = "Face Inner Product - Isotropic - invert_model"
        self.location = "faces"
        self.invert_model = True
        self.orderTest()


class TestInnerProductsEdgeProperties3D(discretize.tests.OrderTest):
    """Integrate a function over a line within a unit cube domain
    using edgeInnerProducts."""

    meshTypes = ["uniformTree", "notatreeTree"]
    meshDimension = 3
    meshSizes = [16, 32]

    def getError(self):
        call = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])

        ex = lambda x, y, z: x**2 + y * z
        ey = lambda x, y, z: (z**2) * x + y * z
        ez = lambda x, y, z: y**2 + x * z

        tau_x = lambda x, y, z: x + 1  # x-face properties  # NOQA F841
        tau_y = lambda x, y, z: y + 2  # y-face properties  # NOQA F841
        tau_z = lambda x, y, z: 3 * z + 1  # z-face properties  # NOQA F841

        tau = 3 * [None]
        for ii, comp in enumerate(["x", "y", "z"]):
            k = np.isclose(
                eval("self.M.edges_{}".format(comp))[:, ii - 1], 0.5
            ) & np.isclose(
                eval("self.M.edges_{}".format(comp))[:, ii - 2], 0.5
            )  # x, y or z location for each line
            tau_ii = 1e-8 * eval(
                "np.ones(self.M.nE{})".format(comp)
            )  # effectively zeros but stable
            tau_ii[k] = eval("call(tau_{}, self.M.edges_{}[k, :])".format(comp, comp))
            tau[ii] = tau_ii
        tau = np.hstack(tau)

        analytic = 1.98906250000000  # Found using sympy.

        cart = lambda g: np.c_[call(ex, g), call(ey, g), call(ez, g)]

        Ec = np.vstack((cart(self.M.gridEx), cart(self.M.gridEy), cart(self.M.gridEz)))
        E = self.M.project_edge_vector(Ec)

        if self.invert_model:
            A = self.M.get_edge_inner_product_line(1 / tau, invert_model=True)
        else:
            A = self.M.get_edge_inner_product_line(tau)

        numeric = E.T.dot(A.dot(E))

        err = np.abs(numeric - analytic)

        return err

    def test_order1_edges(self):
        self.name = "Edge Inner Product - Isotropic"
        self.location = "edges"
        self.invert_model = False
        self.orderTest()

    def test_order1_edges_invert_model(self):
        self.name = "Edge Inner Product - Isotropic - invert_model"
        self.location = "edges"
        self.invert_model = True
        self.orderTest()


class TestTreeInnerProducts2D(discretize.tests.OrderTest):
    """Integrate an function over a unit cube domain using edgeInnerProducts and faceInnerProducts."""

    meshTypes = ["uniformTree"]
    meshDimension = 2
    meshSizes = [4, 8]

    def getError(self):
        z = 5  # Because 5 is just such a great number.

        call = lambda fun, xy: fun(xy[:, 0], xy[:, 1])

        ex = lambda x, y: x**2 + y * z
        ey = lambda x, y: (z**2) * x + y * z

        sigma1 = lambda x, y: x * y + 1
        sigma2 = lambda x, y: x * z + 2
        sigma3 = lambda x, y: 3 + z * y

        Gc = self.M.gridCC
        if self.sigmaTest == 1:
            sigma = np.c_[call(sigma1, Gc)]
            analytic = 144877.0 / 360  # Found using sympy. z=5
        elif self.sigmaTest == 2:
            sigma = np.c_[call(sigma1, Gc), call(sigma2, Gc)]
            analytic = 189959.0 / 120  # Found using sympy. z=5
        elif self.sigmaTest == 3:
            sigma = np.r_[call(sigma1, Gc), call(sigma2, Gc), call(sigma3, Gc)]
            analytic = 781427.0 / 360  # Found using sympy. z=5

        if self.location == "edges":
            cart = lambda g: np.c_[call(ex, g), call(ey, g)]
            Ec = np.vstack((cart(self.M.gridEx), cart(self.M.gridEy)))
            E = self.M.project_edge_vector(Ec)
            if self.invert_model:
                A = self.M.get_edge_inner_product(
                    discretize.utils.inverse_property_tensor(self.M, sigma),
                    invert_model=True,
                )
            else:
                A = self.M.get_edge_inner_product(sigma)
            numeric = E.T.dot(A.dot(E))
        elif self.location == "faces":
            cart = lambda g: np.c_[call(ex, g), call(ey, g)]
            Fc = np.vstack((cart(self.M.gridFx), cart(self.M.gridFy)))
            F = self.M.project_face_vector(Fc)

            if self.invert_model:
                A = self.M.get_face_inner_product(
                    discretize.utils.inverse_property_tensor(self.M, sigma),
                    invert_model=True,
                )
            else:
                A = self.M.get_face_inner_product(sigma)
            numeric = F.T.dot(A.dot(F))

        err = np.abs(numeric - analytic)
        return err

    def test_order1_edges(self):
        self.name = "2D Edge Inner Product - Isotropic"
        self.location = "edges"
        self.sigmaTest = 1
        self.invert_model = False
        self.orderTest()

    def test_order1_edges_invert_model(self):
        self.name = "2D Edge Inner Product - Isotropic - invert_model"
        self.location = "edges"
        self.sigmaTest = 1
        self.invert_model = True
        self.orderTest()

    def test_order3_edges(self):
        self.name = "2D Edge Inner Product - Anisotropic"
        self.location = "edges"
        self.sigmaTest = 2
        self.invert_model = False
        self.orderTest()

    def test_order3_edges_invert_model(self):
        self.name = "2D Edge Inner Product - Anisotropic - invert_model"
        self.location = "edges"
        self.sigmaTest = 2
        self.invert_model = True
        self.orderTest()

    # def test_order6_edges(self):
    #     self.name = "2D Edge Inner Product - Full Tensor"
    #     self.location = 'edges'
    #     self.sigmaTest = 3
    #     self.invert_model = False
    #     self.orderTest()

    # def test_order6_edges_invert_model(self):
    #     self.name = "2D Edge Inner Product - Full Tensor - invert_model"
    #     self.location = 'edges'
    #     self.sigmaTest = 3
    #     self.invert_model = True
    #     self.orderTest()

    def test_order1_faces(self):
        self.name = "2D Face Inner Product - Isotropic"
        self.location = "faces"
        self.sigmaTest = 1
        self.invert_model = False
        self.orderTest()

    def test_order1_faces_invert_model(self):
        self.name = "2D Face Inner Product - Isotropic - invert_model"
        self.location = "faces"
        self.sigmaTest = 1
        self.invert_model = True
        self.orderTest()

    def test_order2_faces(self):
        self.name = "2D Face Inner Product - Anisotropic"
        self.location = "faces"
        self.sigmaTest = 2
        self.invert_model = False
        self.orderTest()

    def test_order2_faces_invert_model(self):
        self.name = "2D Face Inner Product - Anisotropic - invert_model"
        self.location = "faces"
        self.sigmaTest = 2
        self.invert_model = True
        self.orderTest()

    def test_order3_faces(self):
        self.name = "2D Face Inner Product - Full Tensor"
        self.location = "faces"
        self.sigmaTest = 3
        self.invert_model = False
        self.orderTest()

    def test_order3_faces_invert_model(self):
        self.name = "2D Face Inner Product - Full Tensor - invert_model"
        self.location = "faces"
        self.sigmaTest = 3
        self.invert_model = True
        self.orderTest()


class TestInnerProductsFaceProperties2D(discretize.tests.OrderTest):
    """Integrate a function over a surface within a unit cube domain
    using edgeInnerProducts and faceInnerProducts."""

    meshTypes = ["uniformTree", "notatreeTree"]
    meshDimension = 2
    meshSizes = [16, 32]

    def getError(self):
        call = lambda fun, xy: fun(xy[:, 0], xy[:, 1])

        ex = lambda x, y: x**2 + y
        ey = lambda x, y: (y**2) * x

        tau_x = lambda x, y: 2 * y + 1  # x-face properties  # NOQA F841
        tau_y = lambda x, y: x + 2  # y-face properties  # NOQA F841

        tau = 2 * [None]
        for ii, comp in enumerate(["x", "y"]):
            k = np.isclose(
                eval("self.M.faces_{}".format(comp))[:, ii], 0.5
            )  # x, or y location for each plane
            tau_ii = 1e-8 * eval(
                "np.ones(self.M.nF{})".format(comp)
            )  # effectively zeros but stable
            tau_ii[k] = eval("call(tau_{}, self.M.faces_{}[k, :])".format(comp, comp))
            tau[ii] = tau_ii
        tau = np.hstack(tau)

        # integrate components parallel to the plane of integration
        if self.location == "edges":
            analytic = 2.24166666666667  # Found using sympy.

            cart = lambda g: np.c_[call(ex, g), call(ey, g)]

            Ec = np.vstack((cart(self.M.gridEx), cart(self.M.gridEy)))
            E = self.M.project_edge_vector(Ec)

            if self.invert_model:
                A = self.M.get_edge_inner_product_surface(1 / tau, invert_model=True)
            else:
                A = self.M.get_edge_inner_product_surface(tau)

            numeric = E.T.dot(A.dot(E))

        # integrate component normal to the plane of integration
        elif self.location == "faces":
            analytic = 1.59895833333333  # Found using sympy.

            cart = lambda g: np.c_[call(ex, g), call(ey, g)]

            Fc = np.vstack((cart(self.M.gridFx), cart(self.M.gridFy)))
            F = self.M.project_face_vector(Fc)

            if self.invert_model:
                A = self.M.get_face_inner_product_surface(1 / tau, invert_model=True)
            else:
                A = self.M.get_face_inner_product_surface(tau)

            numeric = F.T.dot(A.dot(F))

        err = np.abs(numeric - analytic)

        return err

    def test_order1_edges(self):
        self.name = "Edge Inner Product - Isotropic"
        self.location = "edges"
        self.invert_model = False
        self.orderTest()

    def test_order1_edges_invert_model(self):
        self.name = "Edge Inner Product - Isotropic - invert_model"
        self.location = "edges"
        self.invert_model = True
        self.orderTest()

    def test_order1_faces(self):
        self.name = "Face Inner Product - Isotropic"
        self.location = "faces"
        self.invert_model = False
        self.orderTest()

    def test_order1_faces_invert_model(self):
        self.name = "Face Inner Product - Isotropic - invert_model"
        self.location = "faces"
        self.invert_model = True
        self.orderTest()


class TestInnerProductsEdgeProperties2D(discretize.tests.OrderTest):
    """Integrate a function over a line within a unit cube domain
    using edgeInnerProducts."""

    meshTypes = ["uniformTree", "notatreeTree"]
    meshDimension = 2
    meshSizes = [16, 32]

    def getError(self):
        call = lambda fun, xy: fun(xy[:, 0], xy[:, 1])

        ex = lambda x, y: x**2 + y
        ey = lambda x, y: (x**2) * y

        tau_x = lambda x, y: x + 1  # x-face properties  # NOQA F841
        tau_y = lambda x, y: y + 2  # y-face properties  # NOQA F841

        tau = 2 * [None]
        for ii, comp in enumerate(["x", "y"]):
            k = np.isclose(
                eval("self.M.edges_{}".format(comp))[:, ii - 1], 0.5
            ) & np.isclose(
                eval("self.M.edges_{}".format(comp))[:, ii - 2], 0.5
            )  # x, y or z location for each line
            tau_ii = 1e-8 * eval(
                "np.ones(self.M.nE{})".format(comp)
            )  # effectively zeros but stable
            tau_ii[k] = eval("call(tau_{}, self.M.edges_{}[k, :])".format(comp, comp))
            tau[ii] = tau_ii
        tau = np.hstack(tau)

        analytic = 1.38229166666667  # Found using sympy.

        cart = lambda g: np.c_[call(ex, g), call(ey, g)]

        Ec = np.vstack((cart(self.M.gridEx), cart(self.M.gridEy)))
        E = self.M.project_edge_vector(Ec)

        if self.invert_model:
            A = self.M.get_edge_inner_product_line(1 / tau, invert_model=True)
        else:
            A = self.M.get_edge_inner_product_line(tau)

        numeric = E.T.dot(A.dot(E))

        err = np.abs(numeric - analytic)

        return err

    def test_order1_edges(self):
        self.name = "Edge Inner Product - Isotropic"
        self.location = "edges"
        self.invert_model = False
        self.orderTest()

    def test_order1_edges_invert_model(self):
        self.name = "Edge Inner Product - Isotropic - invert_model"
        self.location = "edges"
        self.invert_model = True
        self.orderTest()


class TestTreeAveraging2D(discretize.tests.OrderTest):
    """Integrate an function over a unit cube domain using edgeInnerProducts and faceInnerProducts."""

    meshTypes = ["notatreeTree", "uniformTree"]
    meshDimension = 2
    meshSizes = [4, 8, 16]
    expectedOrders = [2, 1]

    def getError(self):
        num = self.getAve(self.M) * self.getHere(self.M)
        err = np.linalg.norm((self.getThere(self.M) - num), np.inf)
        return err

    def test_orderN2CC(self):
        self.name = "Averaging 2D: N2CC"
        fun = lambda x, y: (np.cos(x) + np.sin(y))
        self.getHere = lambda M: call2(fun, M.gridN)
        self.getThere = lambda M: call2(fun, M.gridCC)
        self.getAve = lambda M: M.aveN2CC
        self.orderTest()

    def test_orderN2Fx(self):
        self.name = "Averaging 2D: N2Fx"
        fun = lambda x, y: (np.cos(x) + np.sin(y))
        self.getHere = lambda M: call2(fun, M.gridN)
        self.getThere = lambda M: np.r_[call2(fun, M.gridFx), call2(fun, M.gridFy)]
        self.getAve = lambda M: M.aveN2F
        self.orderTest()

    def test_orderN2E(self):
        self.name = "Averaging 2D: N2E"
        fun = lambda x, y: (np.cos(x) + np.sin(y))
        self.getHere = lambda M: call2(fun, M.gridN)
        self.getThere = lambda M: np.r_[call2(fun, M.gridEx), call2(fun, M.gridEy)]
        self.getAve = lambda M: M.aveN2E
        self.orderTest()

    def test_orderF2CC(self):
        self.name = "Averaging 2D: F2CC"
        fun = lambda x, y: (np.cos(x) + np.sin(y))
        self.getHere = lambda M: np.r_[call2(fun, np.r_[M.gridFx, M.gridFy])]
        self.getThere = lambda M: call2(fun, M.gridCC)
        self.getAve = lambda M: M.aveF2CC
        self.orderTest()

    def test_orderFx2CC(self):
        self.name = "Averaging 2D: Fx2CC"
        funX = lambda x, y: (np.cos(x) + np.sin(y))
        self.getHere = lambda M: np.r_[call2(funX, M.gridFx)]
        self.getThere = lambda M: np.r_[call2(funX, M.gridCC)]
        self.getAve = lambda M: M.aveFx2CC
        self.orderTest()

    def test_orderFy2CC(self):
        self.name = "Averaging 2D: Fy2CC"
        funY = lambda x, y: (np.cos(y) * np.sin(x))
        self.getHere = lambda M: np.r_[call2(funY, M.gridFy)]
        self.getThere = lambda M: np.r_[call2(funY, M.gridCC)]
        self.getAve = lambda M: M.aveFy2CC
        self.orderTest()

    def test_orderF2CCV(self):
        self.name = "Averaging 2D: F2CCV"
        funX = lambda x, y: (np.cos(x) + np.sin(y))
        funY = lambda x, y: (np.cos(y) * np.sin(x))
        self.getHere = lambda M: np.r_[call2(funX, M.gridFx), call2(funY, M.gridFy)]
        self.getThere = lambda M: np.r_[call2(funX, M.gridCC), call2(funY, M.gridCC)]
        self.getAve = lambda M: M.aveF2CCV
        self.orderTest()

    def test_orderCC2F(self):
        self.name = "Averaging 2D: CC2F"
        fun = lambda x, y: (np.cos(x) + np.sin(y))
        self.getHere = lambda M: call2(fun, M.gridCC)
        self.getThere = lambda M: np.r_[call2(fun, M.gridFx), call2(fun, M.gridFy)]
        self.getAve = lambda M: M.aveCC2F
        self.expectedOrders = 1
        self.orderTest()
        self.expectedOrders = 2


class TestAveraging3D(discretize.tests.OrderTest):
    name = "Averaging 3D"
    meshTypes = ["notatreeTree", "uniformTree"]
    meshDimension = 3
    meshSizes = [8, 16]
    expectedOrders = [2, 1]

    def getError(self):
        num = self.getAve(self.M) * self.getHere(self.M)
        err = np.linalg.norm((self.getThere(self.M) - num), np.inf)
        return err

    def test_orderN2CC(self):
        self.name = "Averaging 3D: N2CC"
        fun = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: call3(fun, M.gridN)
        self.getThere = lambda M: call3(fun, M.gridCC)
        self.getAve = lambda M: M.aveN2CC
        self.orderTest()

    def test_orderN2F(self):
        self.name = "Averaging 3D: N2F"
        fun = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: call3(fun, M.gridN)
        self.getThere = lambda M: np.r_[
            call3(fun, M.gridFx), call3(fun, M.gridFy), call3(fun, M.gridFz)
        ]
        self.getAve = lambda M: M.aveN2F
        self.orderTest()

    def test_orderN2E(self):
        self.name = "Averaging 3D: N2E"
        fun = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: call3(fun, M.gridN)
        self.getThere = lambda M: np.r_[
            call3(fun, M.gridEx), call3(fun, M.gridEy), call3(fun, M.gridEz)
        ]
        self.getAve = lambda M: M.aveN2E
        self.orderTest()

    def test_orderF2CC(self):
        self.name = "Averaging 3D: F2CC"
        fun = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: np.r_[
            call3(fun, M.gridFx), call3(fun, M.gridFy), call3(fun, M.gridFz)
        ]
        self.getThere = lambda M: call3(fun, M.gridCC)
        self.getAve = lambda M: M.aveF2CC
        self.orderTest()

    def test_orderFx2CC(self):
        self.name = "Averaging 3D: Fx2CC"
        funX = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: np.r_[call3(funX, M.gridFx)]
        self.getThere = lambda M: np.r_[call3(funX, M.gridCC)]
        self.getAve = lambda M: M.aveFx2CC
        self.orderTest()

    def test_orderFy2CC(self):
        self.name = "Averaging 3D: Fy2CC"
        funY = lambda x, y, z: (np.cos(x) + np.sin(y) * np.exp(z))
        self.getHere = lambda M: np.r_[call3(funY, M.gridFy)]
        self.getThere = lambda M: np.r_[call3(funY, M.gridCC)]
        self.getAve = lambda M: M.aveFy2CC
        self.orderTest()

    def test_orderFz2CC(self):
        self.name = "Averaging 3D: Fz2CC"
        funZ = lambda x, y, z: (np.cos(x) + np.sin(y) * np.exp(z))
        self.getHere = lambda M: np.r_[call3(funZ, M.gridFz)]
        self.getThere = lambda M: np.r_[call3(funZ, M.gridCC)]
        self.getAve = lambda M: M.aveFz2CC
        self.orderTest()

    def test_orderF2CCV(self):
        self.name = "Averaging 3D: F2CCV"
        funX = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        funY = lambda x, y, z: (np.cos(x) + np.sin(y) * np.exp(z))
        funZ = lambda x, y, z: (np.cos(x) * np.sin(y) + np.exp(z))
        self.getHere = lambda M: np.r_[
            call3(funX, M.gridFx), call3(funY, M.gridFy), call3(funZ, M.gridFz)
        ]
        self.getThere = lambda M: np.r_[
            call3(funX, M.gridCC), call3(funY, M.gridCC), call3(funZ, M.gridCC)
        ]
        self.getAve = lambda M: M.aveF2CCV
        self.orderTest()

    def test_orderEx2CC(self):
        self.name = "Averaging 3D: Ex2CC"
        funX = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: np.r_[call3(funX, M.gridEx)]
        self.getThere = lambda M: np.r_[call3(funX, M.gridCC)]
        self.getAve = lambda M: M.aveEx2CC
        self.orderTest()

    def test_orderEy2CC(self):
        self.name = "Averaging 3D: Ey2CC"
        funY = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: np.r_[call3(funY, M.gridEy)]
        self.getThere = lambda M: np.r_[call3(funY, M.gridCC)]
        self.getAve = lambda M: M.aveEy2CC
        self.orderTest()

    def test_orderEz2CC(self):
        self.name = "Averaging 3D: Ez2CC"
        funZ = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: np.r_[call3(funZ, M.gridEz)]
        self.getThere = lambda M: np.r_[call3(funZ, M.gridCC)]
        self.getAve = lambda M: M.aveEz2CC
        self.orderTest()

    def test_orderE2CC(self):
        self.name = "Averaging 3D: E2CC"
        fun = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: np.r_[
            call3(fun, M.gridEx), call3(fun, M.gridEy), call3(fun, M.gridEz)
        ]
        self.getThere = lambda M: call3(fun, M.gridCC)
        self.getAve = lambda M: M.aveE2CC
        self.orderTest()

    def test_orderE2CCV(self):
        self.name = "Averaging 3D: E2CCV"
        funX = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        funY = lambda x, y, z: (np.cos(x) + np.sin(y) * np.exp(z))
        funZ = lambda x, y, z: (np.cos(x) * np.sin(y) + np.exp(z))
        self.getHere = lambda M: np.r_[
            call3(funX, M.gridEx), call3(funY, M.gridEy), call3(funZ, M.gridEz)
        ]
        self.getThere = lambda M: np.r_[
            call3(funX, M.gridCC), call3(funY, M.gridCC), call3(funZ, M.gridCC)
        ]
        self.getAve = lambda M: M.aveE2CCV
        self.orderTest()

    def test_orderCC2F(self):
        self.name = "Averaging 3D: CC2F"
        fun = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: call3(fun, M.gridCC)
        self.getThere = lambda M: np.r_[
            call3(fun, M.gridFx), call3(fun, M.gridFy), call3(fun, M.gridFz)
        ]
        self.getAve = lambda M: M.aveCC2F
        self.expectedOrders = 1
        self.orderTest()
        self.expectedOrders = 2
