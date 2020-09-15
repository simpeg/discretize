"""
Base classes for all discretize meshes
"""

import numpy as np
import properties
import os
import json

from ..utils import mkvc
from ..utils.code_utils import deprecate_property, deprecate_method
from ..mixins import InterfaceMixins


class BaseMesh(properties.HasProperties, InterfaceMixins):
    """
    BaseMesh does all the counting you don't want to do.
    BaseMesh should be inherited by meshes with a regular structure.
    """

    _REGISTRY = {}

    # Properties
    _n = properties.Array(
        "number of cells in each direction (dim, )",
        dtype=int,
        required=True,
        shape=('*',)
    )

    x0 = properties.Array(
        "origin of the mesh (dim, )",
        dtype=(float, int),
        shape=('*',),
        required=True,
    )

    # Instantiate the class
    def __init__(self, n=None, x0=None, **kwargs):
        if n is not None:
            self._n = n  # number of dimensions

        if x0 is None:
            self.x0 = np.zeros(len(self._n))
        else:
            self.x0 = x0

        super(BaseMesh, self).__init__(**kwargs)

    # Validators
    @properties.validator('_n')
    def _check_n_shape(self, change):
        if not (
            not isinstance(change['value'], properties.utils.Sentinel) and
            change['value'] is not None
        ):
            raise Exception("Cannot delete n. Instead, create a new mesh")

        change['value'] = np.array(change['value'], dtype=int).ravel()
        if len(change['value']) > 3:
            raise Exception(
                "Dimensions of {}, which is higher than 3 are not "
                "supported".format(change['value'])
            )

        if np.any(change['previous'] != properties.undefined):
            # can't change dimension of the mesh
            if len(change['previous']) != len(change['value']):
                raise Exception(
                    "Cannot change dimensionality of the mesh. Expected {} "
                    "dimensions, got {} dimensions".format(
                        len(change['previous']), len(change['value'])
                    )
                )

            # check that if h has been set, sizes still agree
            if getattr(self, 'h', None) is not None and len(self.h) > 0:
                for i in range(len(change['value'])):
                    if len(self.h[i]) != change['value'][i]:
                        raise Exception(
                            "Mismatched shape of n. Expected {}, len(h[{}]), got "
                            "{}".format(
                                len(self.h[i]), i, change['value'][i]
                            )
                        )

            # check that if nodes have been set for curvi mesh, sizes still
            # agree
            if (
                getattr(self, 'nodes', None) is not None and
                len(self.nodes) > 0
            ):
                for i in range(len(change['value'])):
                    if self.nodes[0].shape[i]-1 != change['value'][i]:
                        raise Exception(
                            "Mismatched shape of n. Expected {}, len(nodes[{}]), "
                            "got {}".format(
                                self.nodes[0].shape[i]-1, i, change['value'][i]
                            )
                        )

    @properties.validator('x0')
    def _check_x0(self, change):
        if not (
            not isinstance(change['value'], properties.utils.Sentinel) and
            change['value'] is not None
        ):
            raise Exception("n must be set prior to setting x0")

        if len(self._n) != len(change['value']):
            raise Exception(
                "Dimension mismatch. x0 has length {} != len(n) which is "
                "{}".format(len(x0), len(n))
            )

    @property
    def dim(self):
        """The dimension of the mesh (1, 2, or 3).

        Returns
        -------
        int
            dimension of the mesh
        """
        return len(self._n)

    @property
    def n_cells(self):
        """Total number of cells in the mesh.

        Returns
        -------
        int
            number of cells in the mesh

        Examples
        --------
        >>> import discretize
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> mesh = discretize.TensorMesh([np.ones(n) for n in [2,3]])
        >>> mesh.plotGrid(centers=True, show_it=True)
        >>> print(mesh.n_cells)
        """
        return int(self._n.prod())

    @property
    def n_nodes(self):
        """Total number of nodes

        Returns
        -------
        int
            number of nodes in the mesh

        Examples
        --------
        >>> import discretize
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> mesh = discretize.TensorMesh([np.ones(n) for n in [2,3]])
        >>> mesh.plotGrid(nodes=True, show_it=True)
        >>> print(mesh.n_nodes)
        """
        return int((self._n+1).prod())

    @property
    def n_edges_x(self):
        """Number of x-edges

        Returns
        -------
        int

        """
        return int((self._n + np.r_[0, 1, 1][:self.dim]).prod())

    @property
    def n_edges_y(self):
        """Number of y-edges

        Returns
        -------
        int

        """
        if self.dim < 2:
            return None
        return int((self._n + np.r_[1, 0, 1][:self.dim]).prod())

    @property
    def n_edges_z(self):
        """Number of z-edges

        Returns
        -------
        int

        """
        if self.dim < 3:
            return None
        return int((self._n + np.r_[1, 1, 0][:self.dim]).prod())

    @property
    def vnE(self):
        """The number of edges in each direction

        Returns
        -------
        vnE : tuple
            [n_edges_x, n_edges_y, n_edges_z], (dim, )

        Examples
        --------
        >>> import discretize
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> M = discretize.TensorMesh([np.ones(n) for n in [2,3]])
        >>> M.plotGrid(edges=True, show_it=True)
        """
        return (x for x in [self.n_edges_x, self.n_edges_y, self.n_edges_z] if x is not None)

    @property
    def n_edges(self):
        """Total number of edges.

        Returns
        -------
        int
            sum([n_edges_x, n_edges_y, n_edges_z])

        """
        return int(self.vnE.sum())

    @property
    def n_faces_x(self):
        """Number of x-faces

        Returns
        -------
        int
        """
        return int((self._n + np.r_[1, 0, 0][:self.dim]).prod())

    @property
    def n_faces_y(self):
        """Number of y-faces

        Returns
        -------
        int
        """
        if self.dim < 2:
            return None
        return int((self._n + np.r_[0, 1, 0][:self.dim]).prod())

    @property
    def n_faces_z(self):
        """Number of z-faces

        Returns
        -------
        int
        """
        if self.dim < 3:
            return None
        return int((self._n + np.r_[0, 0, 1][:self.dim]).prod())

    @property
    def vnF(self):
        """The number of faces in each direction

        Returns
        -------
        vnF : tuple
            [n_faces_x, n_faces_y, n_faces_z], (dim, )

        Examples
        --------
        >>> import discretize
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> M = discretize.TensorMesh([np.ones(n) for n in [2,3]])
        >>> M.plotGrid(faces=True, show_it=True)
        """
        return (x for x in [self.n_faces_x, self.n_faces_y, self.n_faces_z] if x is not None)

    @property
    def n_faces(self):
        """Total number of faces.

        Returns
        -------
        int
            sum([n_faces_x, n_faces_y, n_faces_z])

        """
        return int(self.vnF.sum())

    @property
    def face_normals(self):
        """Face Normals

        Returns
        -------
        numpy.ndarray
            normals, (n_faces, dim)
        """
        if self.dim == 2:
            nX = np.c_[
                np.ones(self.n_faces_x), np.zeros(self.n_faces_x)
            ]
            nY = np.c_[
                np.zeros(self.n_faces_y), np.ones(self.n_faces_y)
            ]
            return np.r_[nX, nY]
        elif self.dim == 3:
            nX = np.c_[
                np.ones(self.n_faces_x), np.zeros(self.n_faces_x), np.zeros(self.n_faces_x)
            ]
            nY = np.c_[
                np.zeros(self.n_faces_y), np.ones(self.n_faces_y), np.zeros(self.n_faces_y)
            ]
            nZ = np.c_[
                np.zeros(self.n_faces_z), np.zeros(self.n_faces_z), np.ones(self.n_faces_z)
            ]
            return np.r_[nX, nY, nZ]

    @property
    def edge_tangents(self):
        """Edge Tangents

        Returns
        -------
        numpy.ndarray
            normals, (n_edges, dim)
        """
        if self.dim == 2:
            tX = np.c_[
                np.ones(self.n_edges_z), np.zeros(self.n_edges_z)
            ]
            tY = np.c_[
                np.zeros(self.n_edges_y), np.ones(self.n_edges_y)
            ]
            return np.r_[tX, tY]
        elif self.dim == 3:
            tX = np.c_[
                np.ones(self.n_edges_z), np.zeros(self.n_edges_z), np.zeros(self.n_edges_z)
            ]
            tY = np.c_[
                np.zeros(self.n_edges_y), np.ones(self.n_edges_y), np.zeros(self.n_edges_y)
            ]
            tZ = np.c_[
                np.zeros(self.n_edges_z), np.zeros(self.n_edges_z), np.ones(self.n_edges_z)
            ]
            return np.r_[tX, tY, tZ]

    def project_face_vector(self, fV):
        """ Project vectors onto the faces of the mesh.

        Given a vector, fV, in cartesian coordinates, this will project
        it onto the mesh using the normals

        Parameters
        ----------
        fV : numpy.ndarray
            face vector with shape (n_faces, dim)

        Returns
        -------
        numpy.ndarray
            projected face vector, (n_faces, )

        """
        if not isinstance(fV, np.ndarray):
            raise Exception('fV must be an ndarray')
        if not (
            len(fV.shape) == 2 and
            fV.shape[0] == self.n_faces and
            fV.shape[1] == self.dim
        ):
            raise Exception('fV must be an ndarray of shape (n_faces x dim)')
        return np.sum(fV*self.normals, 1)

    def project_edge_vector(self, eV):
        """Project vectors onto the edges of the mesh

        Given a vector, eV, in cartesian coordinates, this will project
        it onto the mesh using the tangents

        Parameters
        ----------
        eV : numpy.ndarray
            edge vector with shape (n_edges, dim)

        Returns
        -------
        numpy.ndarray
            projected edge vector, (n_edges, )

        """
        if not isinstance(eV, np.ndarray):
            raise Exception('eV must be an ndarray')
        if not (
            len(eV.shape) == 2 and
            eV.shape[0] == self.n_edges and
            eV.shape[1] == self.dim
        ):
            raise Exception('eV must be an ndarray of shape (nE x dim)')
        return np.sum(eV*self.edge_tangents, 1)

    def save(self, filename='mesh.json', verbose=False):
        """
        Save the mesh to json
        :param str file: filename for saving the casing properties
        :param str directory: working directory for saving the file
        """

        f = os.path.abspath(filename)  # make sure we are working with abs path
        with open(f, 'w') as outfile:
            json.dump(self.serialize(), outfile)

        if verbose:
            print('Saved {}'.format(f))

        return f

    def copy(self):
        """
        Make a copy of the current mesh
        """
        return properties.copy(self)

    axis_u = properties.Vector3(
        'Vector orientation of u-direction. For more details see the docs for the :attr:`~discretize.base.BaseMesh.rotation_matrix` property.',
        default='X',
        length=1
    )
    axis_v = properties.Vector3(
        'Vector orientation of v-direction. For more details see the docs for the :attr:`~discretize.base.BaseMesh.rotation_matrix` property.',
        default='Y',
        length=1
    )
    axis_w = properties.Vector3(
        'Vector orientation of w-direction. For more details see the docs for the :attr:`~discretize.base.BaseMesh.rotation_matrix` property.',
        default='Z',
        length=1
    )

    @properties.validator
    def _validate_orientation(self):
        """Check if axes are orthogonal"""
        if not (np.abs(self.axis_u.dot(self.axis_v) < 1e-6) and
                np.abs(self.axis_v.dot(self.axis_w) < 1e-6) and
                np.abs(self.axis_w.dot(self.axis_u) < 1e-6)):
            raise ValueError('axis_u, axis_v, and axis_w must be orthogonal')
        return True

    @property
    def reference_is_rotated(self):
        """True if the axes are rotated from the traditional <X,Y,Z> system
        with vectors of :math:`(1,0,0)`, :math:`(0,1,0)`, and :math:`(0,0,1)`
        """
        if (    np.allclose(self.axis_u, (1, 0, 0)) and
                np.allclose(self.axis_v, (0, 1, 0)) and
                np.allclose(self.axis_w, (0, 0, 1)) ):
            return False
        return True

    @property
    def rotation_matrix(self):
        """Builds a rotation matrix to transform coordinates from their coordinate
        system into a conventional cartesian system. This is built off of the
        three `axis_u`, `axis_v`, and `axis_w` properties; these mapping
        coordinates use the letters U, V, and W (the three letters preceding X,
        Y, and Z in the alphabet) to define the projection of the X, Y, and Z
        durections. These UVW vectors describe the placement and transformation
        of the mesh's coordinate sytem assuming at most 3 directions.

        Why would you want to use these UVW mapping vectors the this
        `rotation_matrix` property? They allow us to define the realationship
        between local and global coordinate systems and provide a tool for
        switching between the two while still maintaing the connectivity of the
        mesh's cells. For a visual example of this, please see the figure in the
        docs for the :class:`~discretize.mixins.vtk_mod.InterfaceVTK`.
        """
        return np.array([self.axis_u, self.axis_v, self.axis_w])


    reference_system = properties.String(
        'The type of coordinate reference frame. Can take on the values ' +
        'cartesian, cylindrical, or spherical. Abbreviations of these are allowed.',
        default='cartesian',
        change_case='lower',
    )

    @properties.validator
    def _validate_reference_system(self):
        """Check if the reference system is of a known type."""
        choices = ['cartesian', 'cylindrical', 'spherical']
        # Here are a few abbreviations that users can harnes
        abrevs = {
            'car': choices[0],
            'cart': choices[0],
            'cy': choices[1],
            'cyl': choices[1],
            'sph': choices[2],
        }
        # Get the name and fix it if it is abbreviated
        self.reference_system = abrevs.get(self.reference_system, self.reference_system)
        if self.reference_system not in choices:
            raise ValueError('Coordinate system ({}) unknown.'.format(self.reference_system))
        return True

    # SHORTHAND
    nC = n_cells
    nN = n_nodes
    nEx = n_edges_x
    nEy = n_edges_y
    nEz = n_edges_z
    nE = n_edges
    nFx = n_edges_x
    nFy = n_edges_y
    nFz = n_edges_z
    nF = n_edges

    # DEPRECATED
    normals = deprecate_property(face_normals, 'normals', removal_version='1.0.0')
    tangents = deprecate_property(edge_tangents, 'tangents', removal_version='1.0.0')
    projectEdgeVector = deprecate_method(project_edge_vector, 'projectEdgeVector', removal_version='1.0.0')
    projectFaceVector = deprecate_method(project_face_vector, 'projectFaceVector', removal_version='1.0.0')



class BaseRectangularMesh(BaseMesh):
    """
    BaseRectangularMesh
    """


    def __init__(self, n=None, x0=None, **kwargs):
        BaseMesh.__init__(self, n=n, x0=x0, **kwargs)

    @property
    def nx_cells(self):
        """Number of cells in the x direction

        Returns
        -------
        int
        """
        return int(self._n[0])

    @property
    def ny_cells(self):
        """Number of cells in the y direction

        Returns
        -------
        int or None
            None if dim < 2
        """
        if self.dim < 2:
            return None
        return int(self._n[1])

    @property
    def nz_cells(self):
        """Number of cells in the z direction

        Returns
        -------
        int or None
            None if dim < 3
        """
        if self.dim < 3:
            return None
        return int(self._n[2])

    @property
    def shape_cells(self):
        """The number of cells in each direction

        Returns
        -------
        tuple of int
            [nx_cells, ny_cells, nz_cells]
        """
        return (self.nx_cells, self.ny_cells, self.nz_cells] if x is not None)
        )

    @property
    def nx_nodes(self):
        """Number of nodes in the x-direction

        Returns
        -------
        int
        """
        return self.nx_cells + 1

    @property
    def ny_nodes(self):
        """Number of nodes in the y-direction

        Returns
        -------
        int or None
            None if dim < 2
        """
        if self.dim < 2:
            return None
        return self.ny_cells + 1

    @property
    def nz_nodes(self):
        """Number of nodes in the z-direction

        Returns
        -------
        int or None
            None if dim < 3
        """
        if self.dim < 3:
            return None
        return self.nz_cells + 1

    @property
    def shape_nodes(self):
        """Number of nodes in each direction

        Returns
        -------
        tuple of int
            (nx_nodes, ny_nodes, nz_nodes)
        """
        return
            (x for x in [self.nx_nodes, self.ny_nodes, self.nz_nodes] if x is not None)

    @property
    def shape_edges_x(self):
        """Number of x-edges in each direction

        Returns
        -------
        tuple of int
            (nx_cells, ny_nodes, nz_nodes)
        """
        return (x for x in [self.nx_cells, self.ny_nodes, self.nz_nodes] if x is not None)

    @property
    def shape_edges_y(self):
        """Number of y-edges in each direction

        Returns
        -------
        tuple of int or None
            (nx_nodes, ny_cells, nz_nodes), None if dim < 2
        """
        if self.dim < 2:
            return None
        return (x for x in [self.nx_nodes, self.ny_cells, self.nz_nodes] if x is not None)

    @property
    def shape_edges_z(self):
        """Number of z-edges in each direction

        Returns
        -------
        tuple of int or None
            (nx_nodes, ny_nodes, nz_cells), None if dim < 3
        """
        if self.dim < 3:
            return None
        return (self.nx_nodes, self.ny_nodes, self.nz_cells)

    @property
    def shape_faces_x(self):
        """Number of x-faces in each direction

        Returns
        -------
        tuple of int
            (nx_nodes, ny_cells, nz_cells)
        """
        return (x for x in [self.nx_nodes, self.ny_cells, self.nz_cells] if x is not None)

    @property
    def shape_faces_y(self):
        """Number of y-faces in each direction

        Returns
        -------
        tuple of int or None
            (nx_cells, ny_nodes, nz_cells), None if dim < 2
        """
        if self.dim < 2:
            return None
        return (x for x in [self.nx_cells, self.ny_nodes, self.nz_cells] if x is not None)

    @property
    def shape_faces_z(self):
        """Number of z-faces in each direction

        Returns
        -------
        tuple of int or None
            (nx_cells, ny_cells, nz_nodes), None if dim < 3
        """
        if self.dim < 3:
            return None
        return (self.nx_cells, self.ny_cells, self.nz_nodes)

    # ##################################
    # # Redo the numbering so they are dependent of the vector numbers
    # ##################################
    #
    # @property
    # def nC(self):
    #     """Total number of cells
    #
    #     :rtype: int
    #     :return: nC
    #     """
    #     return int(self.vnC.prod())
    #
    # @property
    # def nN(self):
    #     """Total number of nodes
    #
    #     :rtype: int
    #     :return: nN
    #     """
    #     return int(self.vnN.prod())
    #
    # @property
    # def n_edges_z(self):
    #     """Number of x-edges
    #
    #     :rtype: int
    #     :return: n_edges_z
    #     """
    #     return int(self.shape_edges_z.prod())
    #
    # @property
    # def n_edges_y(self):
    #     """Number of y-edges
    #
    #     :rtype: int
    #     :return: n_edges_y
    #     """
    #     if self.dim < 2:
    #         return
    #     return int(self.shape_edges_y.prod())
    #
    # @property
    # def n_edges_z(self):
    #     """Number of z-edges
    #
    #     :rtype: int
    #     :return: n_edges_z
    #     """
    #     if self.dim < 3:
    #         return
    #     return int(self.shape_edges_z.prod())
    #
    # @property
    # def n_faces_x(self):
    #     """Number of x-faces
    #
    #     :rtype: int
    #     :return: n_faces_x
    #     """
    #     return int(self.shape_faces_x.prod())
    #
    # @property
    # def n_faces_y(self):
    #     """Number of y-faces
    #
    #     :rtype: int
    #     :return: n_faces_y
    #     """
    #     if self.dim < 2:
    #         return
    #     return int(self.shape_faces_y.prod())
    #
    # @property
    # def n_faces_z(self):
    #     """Number of z-faces
    #
    #     :rtype: int
    #     :return: n_faces_z
    #     """
    #     if self.dim < 3:
    #         return
    #     return int(self.shape_faces_z.prod())

    def reshape(self, x, xType='CC', outType='CC', format='V'):
        """`r` is a quick reshape command that will do the best it
        can at giving you what you want.

        For example, you have a face variable, and you want the x
        component of it reshaped to a 3D matrix.

        `r` can fulfil your dreams::

            mesh.r(V, 'F', 'Fx', 'M')
                   |   |     |    |
                   |   |     |    {
                   |   |     |      How: 'M' or ['V'] for a matrix
                   |   |     |      (ndgrid style) or a vector (n x dim)
                   |   |     |    }
                   |   |     {
                   |   |       What you want: ['CC'], 'N',
                   |   |                       'F', 'Fx', 'Fy', 'Fz',
                   |   |                       'E', 'Ex', 'Ey', or 'Ez'
                   |   |     }
                   |   {
                   |     What is it: ['CC'], 'N',
                   |                  'F', 'Fx', 'Fy', 'Fz',
                   |                  'E', 'Ex', 'Ey', or 'Ez'
                   |   }
                   {
                     The input: as a list or ndarray
                   }


        For example::

            # Separates each component of the Ex grid into 3 matrices
            Xex, Yex, Zex = r(mesh.gridEx, 'Ex', 'Ex', 'M')

            # Given an edge vector, return just the x edges as a vector
            XedgeVector = r(edgeVector, 'E', 'Ex', 'V')

            # Separates each component of the edgeVector into 3 vectors
            eX, eY, eZ = r(edgeVector, 'E', 'E', 'V')
        """

        allowed_xType = [
            'CC', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez'
        ]
        if not (
            isinstance(x, list) or isinstance(x, np.ndarray)
        ):
            raise Exception("x must be either a list or a ndarray")
        if xType not in allowed_xType:
            raise Exception (
                "xType must be either "
                "'CC', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', or 'Ez'"
            )
        if outType not in allowed_xType:
            raise Exception(
                "outType must be either "
                "'CC', 'N', 'F', Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', or 'Ez'"
            )
        if format not in ['M', 'V']:
            raise Exception("format must be either 'M' or 'V'")
        if outType[:len(xType)] != xType:
            raise Exception(
                "You cannot change types when reshaping."
            )
        if xType not in outType:
            raise Exception("You cannot change type of components.")

        if isinstance(x, list):
            for i, xi in enumerate(x):
                if not isinstance(x, np.ndarray):
                    raise Exception(
                        "x[{0:d}] must be a numpy array".format(i)
                    )
                if xi.size != x[0].size:
                    raise Exception(
                        "Number of elements in list must not change."
                    )

            x_array = np.ones((x.size, len(x)))
            # Unwrap it and put it in a np array
            for i, xi in enumerate(x):
                x_array[:, i] = mkvc(xi)
            x = x_array

        if not isinstance(x, np.ndarray):
            raise Exception("x must be a numpy array")

        x = x[:]  # make a copy.
        xTypeIsFExyz = (
            len(xType) > 1 and
            xType[0] in ['F', 'E'] and
            xType[1] in ['x', 'y', 'z']
        )

        def outKernal(xx, nn):
            """Returns xx as either a matrix (shape == nn) or a vector."""
            if format == 'M':
                return xx.reshape(nn, order='F')
            elif format == 'V':
                return mkvc(xx)

        def switchKernal(xx):
            """Switches over the different options."""
            if xType in ['CC', 'N']:
                nn = (self._n) if xType == 'CC' else (self._n+1)
                if xx.size != np.prod(nn):
                    raise Exception(
                        "Number of elements must not change."
                    )
                return outKernal(xx, nn)
            elif xType in ['F', 'E']:
                # This will only deal with components of fields,
                # not full 'F' or 'E'
                xx = mkvc(xx)  # unwrap it in case it is a matrix
                nn = self.vnF if xType == 'F' else self.vnE
                nn = np.r_[0, nn]

                nx = [0, 0, 0]
                nx[0] = self.shape_faces_x if xType == 'F' else self.shape_edges_z
                nx[1] = self.shape_faces_y if xType == 'F' else self.shape_edges_y
                nx[2] = self.shape_faces_z if xType == 'F' else self.shape_edges_z

                for dim, dimName in enumerate(['x', 'y', 'z']):
                    if dimName in outType:
                        if self.dim <= dim:
                            raise Exception(
                                "Dimensions of mesh not great enough for "
                                "{}{}".format(xType, dimName)
                            )
                        if xx.size != np.sum(nn):
                            raise Exception(
                                "Vector is not the right size."
                            )
                        start = np.sum(nn[:dim+1])
                        end = np.sum(nn[:dim+2])
                        return outKernal(xx[start:end], nx[dim])

            elif xTypeIsFExyz:
                # This will deal with partial components (x, y or z)
                # lying on edges or faces
                if 'x' in xType:
                    nn = self.shape_faces_x if 'F' in xType else self.shape_edges_z
                elif 'y' in xType:
                    nn = self.shape_faces_y if 'F' in xType else self.shape_edges_y
                elif 'z' in xType:
                    nn = self.shape_faces_z if 'F' in xType else self.shape_edges_z
                if xx.size != np.prod(nn):
                    raise Exception('Vector is not the right size.')
                return outKernal(xx, nn)

        # Check if we are dealing with a vector quantity
        isVectorQuantity = len(x.shape) == 2 and x.shape[1] == self.dim

        if outType in ['F', 'E']:
            if isVectorQuantity:
                raise Exception(
                    'Not sure what to do with a vector vector quantity..'
                )
            outTypeCopy = outType
            out = ()
            for ii, dirName in enumerate(['x', 'y', 'z'][:self.dim]):
                outType = outTypeCopy + dirName
                out += (switchKernal(x),)
            return out
        elif isVectorQuantity:
            out = ()
            for ii in range(x.shape[1]):
                out += (switchKernal(x[:, ii]),)
            return out
        else:
            return switchKernal(x)

    # SHORTHAND
    nCx = nx_cells
    nCy = ny_cells
    nCz = nz_cells
    vnC = shape_cells
    nNx = nx_nodes
    nNy = ny_nodes
    nNz = nz_nodes
    vnN = shape_nodes
    vnEx = shape_edges_x
    vnEy = shape_edges_y
    vnEz = shape_edges_z
    vnFx = shape_faces_x
    vnFy = shape_faces_y
    vnFz = shape_faces_z

    # DEPRECATED
    r  = deprecate_method(reshape, 'r', removal_version="1.0.0")
