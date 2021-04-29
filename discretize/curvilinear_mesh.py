import numpy as np

from discretize.utils import (
    mkvc,
    index_cube,
    face_info,
    volume_tetrahedron,
    make_boundary_bool,
)
from discretize.base import BaseRectangularMesh
from discretize.operators import DiffOperators, InnerProducts
from discretize.mixins import InterfaceMixins
from discretize.utils.code_utils import deprecate_property

# Some helper functions.
def _length2D(x):
    return (x[:, 0] ** 2 + x[:, 1] ** 2) ** 0.5


def _length3D(x):
    return (x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2) ** 0.5


def _normalize2D(x):
    return x / np.kron(np.ones((1, 2)), mkvc(_length2D(x), 2))


def _normalize3D(x):
    return x / np.kron(np.ones((1, 3)), mkvc(_length3D(x), 2))


class CurvilinearMesh(
    BaseRectangularMesh, DiffOperators, InnerProducts, InterfaceMixins
):
    """CurvilinearMesh is a mesh class that deals with curvilinear meshes.

    Example of a curvilinear mesh:

    .. plot::
        :include-source:

        import discretize
        X, Y = discretize.utils.exampleLrmGrid([3,3],'rotate')
        mesh = discretize.CurvilinearMesh([X, Y])
        mesh.plot_grid(show_it=True)
    """

    _meshType = "Curv"
    _aliases = {
        **DiffOperators._aliases,
        **BaseRectangularMesh._aliases,
        **{
            "gridCC": "cell_centers",
            "gridN": "nodes",
            "gridFx": "faces_x",
            "gridFy": "faces_y",
            "gridFz": "faces_z",
            "gridEx": "edges_x",
            "gridEy": "edges_y",
            "gridEz": "edges_z",
        },
    }
    _items = {"node_list"}

    def __init__(self, node_list, **kwargs):

        if "nodes" in kwargs:
            node_list = kwargs.pop("nodes")

        node_list = list(np.asarray(item, dtype=np.float64) for item in node_list)
        # check shapes of each node array match
        dim = len(node_list)
        if dim not in [2, 3]:
            raise ValueError(
                f"Only supports 2 and 3 dimensional meshes, saw a node_list of length {dim}"
            )
        for i, nodes in enumerate(node_list):
            if len(nodes.shape) != dim:
                raise ValueError(
                    f"Unexpected shape of item in node list, expect array with {dim} dimensions, got {len(nodes.shape)}"
                )
            if node_list[0].shape != nodes.shape:
                raise ValueError(
                    f"The shape of nodes are not consistent, saw {node_list[0].shape} and {nodes.shape}"
                )
        self._node_list = tuple(node_list)

        # Save nodes to private variable _nodes as vectors
        self._nodes = np.ones((self.node_list[0].size, dim))
        for i, nodes in enumerate(self.node_list):
            self._nodes[:, i] = mkvc(nodes)

        shape_cells = (n - 1 for n in self.node_list[0].shape)

        # absorb the rest of kwargs, and do not pass to super
        super().__init__(shape_cells, origin=self.nodes[0])

    @property
    def node_list(self):
        return self._node_list

    @classmethod
    def deserialize(cls, value, **kwargs):
        if "nodes" in value:
            value["node_list"] = value.pop("nodes")
        return super().deserialize(value, **kwargs)

    @property
    def cell_centers(self):
        """
        Cell-centered grid
        """
        if getattr(self, "_cell_centers", None) is None:
            self._cell_centers = np.concatenate(
                [self.aveN2CC * self.gridN[:, i] for i in range(self.dim)]
            ).reshape((-1, self.dim), order="F")
        return self._cell_centers

    @property
    def nodes(self):
        """
        Nodal grid.
        """
        if getattr(self, "_nodes", None) is None:
            raise Exception("Someone deleted this. I blame you.")
        return self._nodes

    @property
    def faces_x(self):
        """
        Face staggered grid in the x direction.
        """

        if getattr(self, "_faces_x", None) is None:
            N = self.reshape(self.gridN, "N", "N", "M")
            if self.dim == 2:
                XY = [mkvc(0.5 * (n[:, :-1] + n[:, 1:])) for n in N]
                self._faces_x = np.c_[XY[0], XY[1]]
            elif self.dim == 3:
                XYZ = [
                    mkvc(
                        0.25
                        * (
                            n[:, :-1, :-1]
                            + n[:, :-1, 1:]
                            + n[:, 1:, :-1]
                            + n[:, 1:, 1:]
                        )
                    )
                    for n in N
                ]
                self._faces_x = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._faces_x

    @property
    def faces_y(self):
        """
        Face staggered grid in the y direction.
        """

        if getattr(self, "_faces_y", None) is None:
            N = self.reshape(self.gridN, "N", "N", "M")
            if self.dim == 2:
                XY = [mkvc(0.5 * (n[:-1, :] + n[1:, :])) for n in N]
                self._faces_y = np.c_[XY[0], XY[1]]
            elif self.dim == 3:
                XYZ = [
                    mkvc(
                        0.25
                        * (
                            n[:-1, :, :-1]
                            + n[:-1, :, 1:]
                            + n[1:, :, :-1]
                            + n[1:, :, 1:]
                        )
                    )
                    for n in N
                ]
                self._faces_y = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._faces_y

    @property
    def faces_z(self):
        """
        Face staggered grid in the y direction.
        """

        if getattr(self, "_faces_z", None) is None:
            N = self.reshape(self.gridN, "N", "N", "M")
            XYZ = [
                mkvc(
                    0.25
                    * (n[:-1, :-1, :] + n[:-1, 1:, :] + n[1:, :-1, :] + n[1:, 1:, :])
                )
                for n in N
            ]
            self._faces_z = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._faces_z

    @property
    def faces(self):
        "Face grid"
        faces = np.r_[self.faces_x, self.faces_y]
        if self.dim > 2:
            faces = np.r_[faces, self.faces_z]
        return faces

    @property
    def edges_x(self):
        """
        Edge staggered grid in the x direction.
        """
        if getattr(self, "_edges_x", None) is None:
            N = self.reshape(self.gridN, "N", "N", "M")
            if self.dim == 2:
                XY = [mkvc(0.5 * (n[:-1, :] + n[1:, :])) for n in N]
                self._edges_x = np.c_[XY[0], XY[1]]
            elif self.dim == 3:
                XYZ = [mkvc(0.5 * (n[:-1, :, :] + n[1:, :, :])) for n in N]
                self._edges_x = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._edges_x

    @property
    def edges_y(self):
        """
        Edge staggered grid in the y direction.
        """
        if getattr(self, "_edges_y", None) is None:
            N = self.reshape(self.gridN, "N", "N", "M")
            if self.dim == 2:
                XY = [mkvc(0.5 * (n[:, :-1] + n[:, 1:])) for n in N]
                self._edges_y = np.c_[XY[0], XY[1]]
            elif self.dim == 3:
                XYZ = [mkvc(0.5 * (n[:, :-1, :] + n[:, 1:, :])) for n in N]
                self._edges_y = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._edges_y

    @property
    def edges_z(self):
        """
        Edge staggered grid in the z direction.
        """
        if getattr(self, "_edges_z", None) is None and self.dim == 3:
            N = self.reshape(self.gridN, "N", "N", "M")
            XYZ = [mkvc(0.5 * (n[:, :, :-1] + n[:, :, 1:])) for n in N]
            self._edges_z = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._edges_z

    @property
    def edges(self):
        "Edge grid"
        edges = np.r_[self.edges_x, self.edges_y]
        if self.dim > 2:
            edges = np.r_[edges, self.edges_z]
        return edges

    @property
    def boundary_nodes(self):
        """Boundary node locations

        Returns
        -------
        np.ndarray of float
            location array of shape (mesh.n_boundary_nodes, dim)
        """
        return self.nodes[make_boundary_bool(self.shape_nodes)]

    @property
    def boundary_edges(self):
        """Boundary edge locations

        Returns
        -------
        np.ndarray of float
            location array of shape (mesh.n_boundary_edges, dim)
        """
        if self.dim == 2:
            ex = self.edges_x[make_boundary_bool(self.shape_edges_x, dir="y")]
            ey = self.edges_y[make_boundary_bool(self.shape_edges_y, dir="x")]
            return np.r_[ex, ey]
        elif self.dim == 3:
            ex = self.edges_x[make_boundary_bool(self.shape_edges_x, dir="yz")]
            ey = self.edges_y[make_boundary_bool(self.shape_edges_y, dir="xz")]
            ez = self.edges_z[make_boundary_bool(self.shape_edges_z, dir="xy")]
            return np.r_[ex, ey, ez]

    @property
    def boundary_faces(self):
        """Boundary face locations

        Returns
        -------
        np.ndarray of float
            location array of shape (mesh.n_boundary_faces, dim)
        """
        fx = self.faces_x[make_boundary_bool(self.shape_faces_x, dir="x")]
        fy = self.faces_y[make_boundary_bool(self.shape_faces_y, dir="y")]
        if self.dim == 2:
            return np.r_[fx, fy]
        elif self.dim == 3:
            fz = self.faces_z[make_boundary_bool(self.shape_faces_z, dir="z")]
            return np.r_[fx, fy, fz]

    @property
    def boundary_face_outward_normals(self):
        """Outward directed normal vectors for the boundary faces

        Returns
        -------
        np.ndarray of float
            Array of vectors of shape (mesh.n_boundary_faces, dim)
        """
        is_bxm = np.zeros(self.shape_faces_x, order="F", dtype=bool)
        is_bxm[0, :] = True
        is_bxm = is_bxm.reshape(-1, order="F")

        is_bym = np.zeros(self.shape_faces_y, order="F", dtype=bool)
        is_bym[:, 0] = True
        is_bym = is_bym.reshape(-1, order="F")

        is_b = np.r_[
            make_boundary_bool(self.shape_faces_x, dir="x"),
            make_boundary_bool(self.shape_faces_y, dir="y"),
        ]
        switch = np.r_[is_bxm, is_bym]
        if self.dim == 3:
            is_bzm = np.zeros(self.shape_faces_z, order="F", dtype=bool)
            is_bzm[:, :, 0] = True
            is_bzm = is_bzm.reshape(-1, order="F")

            is_b = np.r_[is_b, make_boundary_bool(self.shape_faces_z, dir="z")]
            switch = np.r_[switch, is_bzm]
        face_normals = self.face_normals.copy()
        face_normals[switch] *= -1
        return face_normals[is_b]

    # --------------- Geometries ---------------------
    #
    #
    # ------------------- 2D -------------------------
    #
    #         node(i,j)          node(i,j+1)
    #              A -------------- B
    #              |                |
    #              |    cell(i,j)   |
    #              |        I       |
    #              |                |
    #             D -------------- C
    #         node(i+1,j)        node(i+1,j+1)
    #
    # ------------------- 3D -------------------------
    #
    #
    #             node(i,j,k+1)       node(i,j+1,k+1)
    #                 E --------------- F
    #                /|               / |
    #               / |              /  |
    #              /  |             /   |
    #       node(i,j,k)         node(i,j+1,k)
    #            A -------------- B     |
    #            |    H ----------|---- G
    #            |   /cell(i,j)   |   /
    #            |  /     I       |  /
    #            | /              | /
    #            D -------------- C
    #       node(i+1,j,k)      node(i+1,j+1,k)

    @property
    def cell_volumes(self):
        """
        Construct cell volumes of the 3D model as 1d array
        """

        if getattr(self, "_cell_volumes", None) is None:
            if self.dim == 2:
                A, B, C, D = index_cube("ABCD", self.vnN)
                normal, area = face_info(
                    np.c_[self.gridN, np.zeros((self.nN, 1))], A, B, C, D
                )
                self._cell_volumes = area
            elif self.dim == 3:
                # Each polyhedron can be decomposed into 5 tetrahedrons
                # However, this presents a choice so we may as well divide in
                # two ways and average.
                A, B, C, D, E, F, G, H = index_cube("ABCDEFGH", self.vnN)

                vol1 = (
                    volume_tetrahedron(self.gridN, A, B, D, E)
                    + volume_tetrahedron(self.gridN, B, E, F, G)  # cutted edge top
                    + volume_tetrahedron(self.gridN, B, D, E, G)  # cutted edge top
                    + volume_tetrahedron(self.gridN, B, C, D, G)  # middle
                    + volume_tetrahedron(self.gridN, D, E, G, H)  # cutted edge bottom
                )  # cutted edge bottom

                vol2 = (
                    volume_tetrahedron(self.gridN, A, F, B, C)
                    + volume_tetrahedron(self.gridN, A, E, F, H)  # cutted edge top
                    + volume_tetrahedron(self.gridN, A, H, F, C)  # cutted edge top
                    + volume_tetrahedron(self.gridN, C, H, D, A)  # middle
                    + volume_tetrahedron(self.gridN, C, G, H, F)  # cutted edge bottom
                )  # cutted edge bottom

                self._cell_volumes = (vol1 + vol2) / 2
        return self._cell_volumes

    @property
    def face_areas(self):
        """
        Area of the faces
        """
        if (
            getattr(self, "_face_areas", None) is None
            or getattr(self, "_normals", None) is None
        ):
            # Compute areas of cell faces
            if self.dim == 2:
                xy = self.gridN
                A, B = index_cube("AB", self.vnN, self.vnFx)
                edge1 = xy[B, :] - xy[A, :]
                normal1 = np.c_[edge1[:, 1], -edge1[:, 0]]
                area1 = _length2D(edge1)
                A, D = index_cube("AD", self.vnN, self.vnFy)
                # Note that we are doing A-D to make sure the normal points the
                # right way.
                # Think about it. Look at the picture. Normal points towards C
                # iff you do this.
                edge2 = xy[A, :] - xy[D, :]
                normal2 = np.c_[edge2[:, 1], -edge2[:, 0]]
                area2 = _length2D(edge2)
                self._face_areas = np.r_[mkvc(area1), mkvc(area2)]
                self._normals = [_normalize2D(normal1), _normalize2D(normal2)]

            elif self.dim == 3:

                A, E, F, B = index_cube("AEFB", self.vnN, self.vnFx)
                normal1, area1 = face_info(
                    self.gridN, A, E, F, B, average=False, normalizeNormals=False
                )

                A, D, H, E = index_cube("ADHE", self.vnN, self.vnFy)
                normal2, area2 = face_info(
                    self.gridN, A, D, H, E, average=False, normalizeNormals=False
                )

                A, B, C, D = index_cube("ABCD", self.vnN, self.vnFz)
                normal3, area3 = face_info(
                    self.gridN, A, B, C, D, average=False, normalizeNormals=False
                )

                self._face_areas = np.r_[mkvc(area1), mkvc(area2), mkvc(area3)]
                self._normals = [normal1, normal2, normal3]
        return self._face_areas

    @property
    def face_normals(self):
        """
        Face normals: calling this will average
        the computed normals so that there is one
        per face. This is especially relevant in
        3D, as there are up to 4 different normals
        for each face that will be different.

        To reshape the normals into a matrix and get the y component::

            NyX, NyY, NyZ = M.reshape(M.face_normals, 'F', 'Fy', 'M')
        """

        if getattr(self, "_normals", None) is None:
            self.face_areas  # calling .face_areas will create the face normals
        if self.dim == 2:
            return _normalize2D(np.r_[self._normals[0], self._normals[1]])
        elif self.dim == 3:
            normal1 = (
                self._normals[0][0]
                + self._normals[0][1]
                + self._normals[0][2]
                + self._normals[0][3]
            ) / 4
            normal2 = (
                self._normals[1][0]
                + self._normals[1][1]
                + self._normals[1][2]
                + self._normals[1][3]
            ) / 4
            normal3 = (
                self._normals[2][0]
                + self._normals[2][1]
                + self._normals[2][2]
                + self._normals[2][3]
            ) / 4
            return _normalize3D(np.r_[normal1, normal2, normal3])

    @property
    def edge_lengths(self):
        """Edge lengths"""
        if getattr(self, "_edge_lengths", None) is None:
            if self.dim == 2:
                xy = self.gridN
                A, D = index_cube("AD", self.vnN, self.vnEx)
                edge1 = xy[D, :] - xy[A, :]
                A, B = index_cube("AB", self.vnN, self.vnEy)
                edge2 = xy[B, :] - xy[A, :]
                self._edge_lengths = np.r_[
                    mkvc(_length2D(edge1)), mkvc(_length2D(edge2))
                ]
                self._edge_tangents = (
                    np.r_[edge1, edge2] / np.c_[self._edge_lengths, self._edge_lengths]
                )
            elif self.dim == 3:
                xyz = self.gridN
                A, D = index_cube("AD", self.vnN, self.vnEx)
                edge1 = xyz[D, :] - xyz[A, :]
                A, B = index_cube("AB", self.vnN, self.vnEy)
                edge2 = xyz[B, :] - xyz[A, :]
                A, E = index_cube("AE", self.vnN, self.vnEz)
                edge3 = xyz[E, :] - xyz[A, :]
                self._edge_lengths = np.r_[
                    mkvc(_length3D(edge1)),
                    mkvc(_length3D(edge2)),
                    mkvc(_length3D(edge3)),
                ]
                self._edge_tangents = (
                    np.r_[edge1, edge2, edge3]
                    / np.c_[self._edge_lengths, self._edge_lengths, self._edge_lengths]
                )
        return self._edge_lengths

    @property
    def edge_tangents(self):
        """Edge tangents"""
        if getattr(self, "_edge_tangents", None) is None:
            self.edge_lengths  # calling .edge_lengths will create the tangents
        return self._edge_tangents

    # DEPRECATIONS
    vol = deprecate_property("cell_volumes", "vol", removal_version="1.0.0")
    area = deprecate_property("face_areas", "area", removal_version="1.0.0")
    edge = deprecate_property("edge_lengths", "edge", removal_version="1.0.0")
    # tangent already deprecated in BaseMesh
    # normals already deprecated in BaseMesh


CurvilinearMesh.__module__ = "discretize"
