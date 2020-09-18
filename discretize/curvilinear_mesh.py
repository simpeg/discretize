from __future__ import print_function
import numpy as np
import properties

from .utils import mkvc, indexCube, faceInfo, volTetra
from .base import BaseRectangularMesh
from .operators import DiffOperators, InnerProducts
from discretize.mixins.mpl_mod import CurviView
from .utils.code_utils import deprecate_property


# Some helper functions.
def _length2D(x):
    return (x[:, 0]**2 + x[:, 1]**2)**0.5


def _length3D(x):
    return (x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)**0.5


def _normalize2D(x):
    return x/np.kron(np.ones((1, 2)), mkvc(_length2D(x), 2))


def _normalize3D(x):
    return x/np.kron(np.ones((1, 3)), mkvc(_length3D(x), 2))


class CurvilinearMesh(
    BaseRectangularMesh, DiffOperators, InnerProducts, CurviView
):
    """CurvilinearMesh is a mesh class that deals with curvilinear meshes.

    Example of a curvilinear mesh:

    .. plot::
        :include-source:

        import discretize
        X, Y = discretize.utils.exampleLrmGrid([3,3],'rotate')
        mesh = discretize.CurvilinearMesh([X, Y])
        mesh.plotGrid(show_it=True)
    """

    _meshType = 'Curv'
    _aliases = {
        **DiffOperators._aliases,
        **BaseRectangularMesh._aliases,
        **{
            'gridCC': 'grid_cell_centers',
            'gridN': 'grid_nodes',
            'gridFx': 'grid_faces_x',
            'gridFy': 'grid_faces_y',
            'gridFz': 'grid_faces_z',
            'gridEx': 'grid_edges_x',
            'gridEy': 'grid_edges_y',
            'gridEz': 'grid_edges_z',
        },
    }

    nodes = properties.List(
        "List of arrays describing the node locations",
        prop=properties.Array(
            "node locations in an n-dimensional array",
            shape={('*', '*'), ('*', '*', '*')}
        ),
        min_length=2,
        max_length=3
    )

    def __init__(self, nodes=None, **kwargs):

        self.nodes = nodes

        if '_n' in kwargs.keys():
            n = kwargs.pop('_n')
            assert (n == np.array(self.nodes[0].shape)-1).all(), (
                "Unexpected n-values. {} was provided, {} was expected".format(
                    n, np.array(self.nodes[0].shape)-1
                )
            )
        else:
            n = np.array(self.nodes[0].shape)-1

        BaseRectangularMesh.__init__(self, n, **kwargs)

        # Save nodes to private variable _gridN as vectors
        self._gridN = np.ones((self.nodes[0].size, self.dim))
        for i, node_i in enumerate(self.nodes):
            self._gridN[:, i] = mkvc(node_i.astype(float))

    @properties.validator('nodes')
    def _check_nodes(self, change):
        assert len(change['value']) > 1, "len(node) must be greater than 1"

        for i, change['value'][i] in enumerate(change['value']):
            assert change['value'][i].shape == change['value'][0].shape, (
                "change['value'][{0:d}] is not the same shape as "
                "change['value'][0]".format(i)
            )

        assert len(change['value'][0].shape) == len(change['value']), (
            "Dimension mismatch"
        )

    @property
    def grid_cell_centers(self):
        """
        Cell-centered grid
        """
        if getattr(self, '_gridCC', None) is None:
            self._gridCC = np.concatenate(
                [self.aveN2CC*self.gridN[:, i] for i in range(self.dim)]
            ).reshape((-1, self.dim), order='F')
        return self._gridCC

    @property
    def grid_nodes(self):
        """
        Nodal grid.
        """
        if getattr(self, '_gridN', None) is None:
            raise Exception("Someone deleted this. I blame you.")
        return self._gridN

    @property
    def grid_faces_x(self):
        """
        Face staggered grid in the x direction.
        """

        if getattr(self, '_gridFx', None) is None:
            N = self.r(self.gridN, 'N', 'N', 'M')
            if self.dim == 2:
                XY = [mkvc(0.5 * (n[:, :-1] + n[:, 1:])) for n in N]
                self._gridFx = np.c_[XY[0], XY[1]]
            elif self.dim == 3:
                XYZ = [mkvc(0.25 * (n[:, :-1, :-1] + n[:, :-1, 1:] +
                       n[:, 1:, :-1] + n[:, 1:, 1:])) for n in N]
                self._gridFx = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridFx

    @property
    def grid_faces_y(self):
        """
        Face staggered grid in the y direction.
        """

        if getattr(self, '_gridFy', None) is None:
            N = self.r(self.gridN, 'N', 'N', 'M')
            if self.dim == 2:
                XY = [mkvc(0.5 * (n[:-1, :] + n[1:, :])) for n in N]
                self._gridFy = np.c_[XY[0], XY[1]]
            elif self.dim == 3:
                XYZ = [mkvc(0.25 * (n[:-1, :, :-1] + n[:-1, :, 1:] +
                       n[1:, :, :-1] + n[1:, :, 1:])) for n in N]
                self._gridFy = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridFy

    @property
    def grid_faces_z(self):
        """
        Face staggered grid in the y direction.
        """

        if getattr(self, '_gridFz', None) is None:
            N = self.r(self.gridN, 'N', 'N', 'M')
            XYZ = [mkvc(0.25 * (n[:-1, :-1, :] + n[:-1, 1:, :] +
                   n[1:, :-1, :] + n[1:, 1:, :])) for n in N]
            self._gridFz = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridFz

    @property
    def grid_edges_x(self):
        """
        Edge staggered grid in the x direction.
        """
        if getattr(self, '_gridEx', None) is None:
            N = self.r(self.gridN, 'N', 'N', 'M')
            if self.dim == 2:
                XY = [mkvc(0.5 * (n[:-1, :] + n[1:, :])) for n in N]
                self._gridEx = np.c_[XY[0], XY[1]]
            elif self.dim == 3:
                XYZ = [mkvc(0.5 * (n[:-1, :, :] + n[1:, :, :])) for n in N]
                self._gridEx = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridEx

    @property
    def grid_edges_y(self):
        """
        Edge staggered grid in the y direction.
        """
        if getattr(self, '_gridEy', None) is None:
            N = self.r(self.gridN, 'N', 'N', 'M')
            if self.dim == 2:
                XY = [mkvc(0.5 * (n[:, :-1] + n[:, 1:])) for n in N]
                self._gridEy = np.c_[XY[0], XY[1]]
            elif self.dim == 3:
                XYZ = [mkvc(0.5 * (n[:, :-1, :] + n[:, 1:, :])) for n in N]
                self._gridEy = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridEy

    @property
    def grid_edges_z(self):
        """
        Edge staggered grid in the z direction.
        """
        if getattr(self, '_gridEz', None) is None and self.dim == 3:
            N = self.r(self.gridN, 'N', 'N', 'M')
            XYZ = [mkvc(0.5 * (n[:, :, :-1] + n[:, :, 1:])) for n in N]
            self._gridEz = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridEz

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

        if getattr(self, '_vol', None) is None:
            if self.dim == 2:
                A, B, C, D = indexCube('ABCD', self.vnN)
                normal, area = faceInfo(np.c_[self.gridN, np.zeros(
                                              (self.nN, 1))], A, B, C, D)
                self._vol = area
            elif self.dim == 3:
                # Each polyhedron can be decomposed into 5 tetrahedrons
                # However, this presents a choice so we may as well divide in
                # two ways and average.
                A, B, C, D, E, F, G, H = indexCube('ABCDEFGH', self.vnN)

                vol1 = (volTetra(self.gridN, A, B, D, E) +  # cutted edge top
                        volTetra(self.gridN, B, E, F, G) +  # cutted edge top
                        volTetra(self.gridN, B, D, E, G) +  # middle
                        volTetra(self.gridN, B, C, D, G) +  # cutted edge bottom
                        volTetra(self.gridN, D, E, G, H))   # cutted edge bottom

                vol2 = (volTetra(self.gridN, A, F, B, C) +  # cutted edge top
                        volTetra(self.gridN, A, E, F, H) +  # cutted edge top
                        volTetra(self.gridN, A, H, F, C) +  # middle
                        volTetra(self.gridN, C, H, D, A) +  # cutted edge bottom
                        volTetra(self.gridN, C, G, H, F))   # cutted edge bottom

                self._vol = (vol1 + vol2)/2
        return self._vol

    @property
    def face_areas(self):
        """
        Area of the faces
        """
        if (getattr(self, '_area', None) is None or
            getattr(self, '_normals', None) is None):
            # Compute areas of cell faces
            if(self.dim == 2):
                xy = self.gridN
                A, B = indexCube('AB', self.vnN, self.vnFx)
                edge1 = xy[B, :] - xy[A, :]
                normal1 = np.c_[edge1[:, 1], -edge1[:, 0]]
                area1 = _length2D(edge1)
                A, D = indexCube('AD', self.vnN, self.vnFy)
                # Note that we are doing A-D to make sure the normal points the
                # right way.
                # Think about it. Look at the picture. Normal points towards C
                # iff you do this.
                edge2 = xy[A, :] - xy[D, :]
                normal2 = np.c_[edge2[:, 1], -edge2[:, 0]]
                area2 = _length2D(edge2)
                self._area = np.r_[mkvc(area1), mkvc(area2)]
                self._normals = [_normalize2D(normal1), _normalize2D(normal2)]

            elif(self.dim == 3):

                A, E, F, B = indexCube('AEFB', self.vnN, self.vnFx)
                normal1, area1 = faceInfo(self.gridN, A, E, F, B,
                                                average=False,
                                                normalizeNormals=False)

                A, D, H, E = indexCube('ADHE', self.vnN, self.vnFy)
                normal2, area2 = faceInfo(self.gridN, A, D, H, E,
                                                average=False,
                                                normalizeNormals=False)

                A, B, C, D = indexCube('ABCD', self.vnN, self.vnFz)
                normal3, area3 = faceInfo(self.gridN, A, B, C, D,
                                                average=False,
                                                normalizeNormals=False)

                self._area = np.r_[mkvc(area1), mkvc(area2),
                                   mkvc(area3)]
                self._normals = [normal1, normal2, normal3]
        return self._area

    @property
    def face_normals(self):
        """
        Face normals: calling this will average
        the computed normals so that there is one
        per face. This is especially relevant in
        3D, as there are up to 4 different normals
        for each face that will be different.

        To reshape the normals into a matrix and get the y component::

            NyX, NyY, NyZ = M.r(M.normals, 'F', 'Fy', 'M')
        """

        if getattr(self, '_normals', None) is None:
            self.area  # calling .area will create the face normals
        if self.dim == 2:
            return _normalize2D(np.r_[self._normals[0], self._normals[1]])
        elif self.dim == 3:
            normal1 = (
                self._normals[0][0] + self._normals[0][1] +
                self._normals[0][2] + self._normals[0][3]
            )/4
            normal2 = (
                self._normals[1][0] + self._normals[1][1] +
                self._normals[1][2] + self._normals[1][3]
            )/4
            normal3 = (
                self._normals[2][0] + self._normals[2][1] +
                self._normals[2][2] + self._normals[2][3]
            )/4
            return _normalize3D(np.r_[normal1, normal2, normal3])

    @property
    def edge_lengths(self):
        """Edge lengths"""
        if getattr(self, '_edge', None) is None:
            if(self.dim == 2):
                xy = self.gridN
                A, D = indexCube('AD', self.vnN, self.vnEx)
                edge1 = xy[D, :] - xy[A, :]
                A, B = indexCube('AB', self.vnN, self.vnEy)
                edge2 = xy[B, :] - xy[A, :]
                self._edge = np.r_[mkvc(_length2D(edge1)),
                                   mkvc(_length2D(edge2))]
                self._tangents = np.r_[edge1, edge2]/np.c_[self._edge,
                                                           self._edge]
            elif(self.dim == 3):
                xyz = self.gridN
                A, D = indexCube('AD', self.vnN, self.vnEx)
                edge1 = xyz[D, :] - xyz[A, :]
                A, B = indexCube('AB', self.vnN, self.vnEy)
                edge2 = xyz[B, :] - xyz[A, :]
                A, E = indexCube('AE', self.vnN, self.vnEz)
                edge3 = xyz[E, :] - xyz[A, :]
                self._edge = np.r_[mkvc(_length3D(edge1)),
                                   mkvc(_length3D(edge2)),
                                   mkvc(_length3D(edge3))]
                self._tangents = (np.r_[edge1, edge2, edge3] /
                                  np.c_[self._edge, self._edge, self._edge])
            return self._edge
        return self._edge

    @property
    def edge_tangents(self):
        """Edge tangents"""
        if getattr(self, '_tangents', None) is None:
            self.edge  # calling .edge will create the tangents
        return self._tangents

    # DEPRECATIONS
    vol = deprecate_property(cell_volumes, 'vol', removal_version="1.0.0")
    area = deprecate_property(face_areas, 'area', removal_version="1.0.0")
    edge = deprecate_property(edge_lengths, 'edge', removal_version="1.0.0")
    # tangent already deprecated in BaseMesh
    # normals already deprecated in BaseMesh
