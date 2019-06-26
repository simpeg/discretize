"""
Base classes for all discretize meshes
"""

import numpy as np
import properties
import os
import json

from ..utils import mkvc
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
        dtype=float,
        shape=('*',),
        required=True
    )

    # Instantiate the class
    def __init__(self, n, x0=None, **kwargs):
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
    def nC(self):
        """Total number of cells in the mesh.

        Returns
        -------
        int
            number of cells in the mesh

        Example
        -------
        .. plot::
            :include-source:

            import discretize
            import numpy as np
            mesh = discretize.TensorMesh([np.ones(n) for n in [2,3]])
            mesh.plotGrid(centers=True, showIt=True)

            print(mesh.nC)
        """
        return int(self._n.prod())

    @property
    def nN(self):
        """Total number of nodes

        Returns
        -------
        int
            number of nodes in the mesh

        Example
        -------
        .. plot::
            :include-source:

            import discretize
            import numpy as np
            mesh = discretize.TensorMesh([np.ones(n) for n in [2,3]])
            mesh.plotGrid(nodes=True, showIt=True)

            print(mesh.nN)
        """
        return int((self._n+1).prod())

    @property
    def nEx(self):
        """Number of x-edges

        Returns
        -------
        nEx : int

        """
        return int((self._n + np.r_[0, 1, 1][:self.dim]).prod())

    @property
    def nEy(self):
        """Number of y-edges

        Returns
        -------
        nEy : int


        """
        if self.dim < 2:
            return None
        return int((self._n + np.r_[1, 0, 1][:self.dim]).prod())

    @property
    def nEz(self):
        """Number of z-edges

        Returns
        -------
        nEz : int


        """
        if self.dim < 3:
            return None
        return int((self._n + np.r_[1, 1, 0][:self.dim]).prod())

    @property
    def vnE(self):
        """Total number of edges in each direction

        Returns
        -------
        vnE : numpy.ndarray = [nEx, nEy, nEz], (dim, )

        .. plot::
            :include-source:

            import discretize
            import numpy as np
            M = discretize.TensorMesh([np.ones(n) for n in [2,3]])
            M.plotGrid(edges=True, showIt=True)
        """
        return np.array(
            [x for x in [self.nEx, self.nEy, self.nEz] if x is not None],
            dtype=int
        )

    @property
    def nE(self):
        """Total number of edges.

        Returns
        -------
        nE : int = sum([nEx, nEy, nEz])

        """
        return int(self.vnE.sum())

    @property
    def nFx(self):
        """Number of x-faces

        :rtype: int
        :return: nFx
        """
        return int((self._n + np.r_[1, 0, 0][:self.dim]).prod())

    @property
    def nFy(self):
        """Number of y-faces

        :rtype: int
        :return: nFy
        """
        if self.dim < 2:
            return None
        return int((self._n + np.r_[0, 1, 0][:self.dim]).prod())

    @property
    def nFz(self):
        """Number of z-faces

        :rtype: int
        :return: nFz
        """
        if self.dim < 3:
            return None
        return int((self._n + np.r_[0, 0, 1][:self.dim]).prod())

    @property
    def vnF(self):
        """Total number of faces in each direction

        :rtype: numpy.ndarray
        :return: [nFx, nFy, nFz], (dim, )

        .. plot::
            :include-source:

            import discretize
            import numpy as np
            M = discretize.TensorMesh([np.ones(n) for n in [2,3]])
            M.plotGrid(faces=True, showIt=True)
        """
        return np.array(
            [x for x in [self.nFx, self.nFy, self.nFz] if x is not None],
            dtype=int
        )

    @property
    def nF(self):
        """Total number of faces.

        :rtype: int
        :return: sum([nFx, nFy, nFz])

        """
        return int(self.vnF.sum())

    @property
    def normals(self):
        """Face Normals

        :rtype: numpy.ndarray
        :return: normals, (sum(nF), dim)
        """
        if self.dim == 2:
            nX = np.c_[
                np.ones(self.nFx), np.zeros(self.nFx)
            ]
            nY = np.c_[
                np.zeros(self.nFy), np.ones(self.nFy)
            ]
            return np.r_[nX, nY]
        elif self.dim == 3:
            nX = np.c_[
                np.ones(self.nFx), np.zeros(self.nFx), np.zeros(self.nFx)
            ]
            nY = np.c_[
                np.zeros(self.nFy), np.ones(self.nFy), np.zeros(self.nFy)
            ]
            nZ = np.c_[
                np.zeros(self.nFz), np.zeros(self.nFz), np.ones(self.nFz)
            ]
            return np.r_[nX, nY, nZ]

    @property
    def tangents(self):
        """Edge Tangents

        :rtype: numpy.ndarray
        :return: normals, (sum(nE), dim)
        """
        if self.dim == 2:
            tX = np.c_[
                np.ones(self.nEx), np.zeros(self.nEx)
            ]
            tY = np.c_[
                np.zeros(self.nEy), np.ones(self.nEy)
            ]
            return np.r_[tX, tY]
        elif self.dim == 3:
            tX = np.c_[
                np.ones(self.nEx), np.zeros(self.nEx), np.zeros(self.nEx)
            ]
            tY = np.c_[
                np.zeros(self.nEy), np.ones(self.nEy), np.zeros(self.nEy)
            ]
            tZ = np.c_[
                np.zeros(self.nEz), np.zeros(self.nEz), np.ones(self.nEz)
            ]
            return np.r_[tX, tY, tZ]

    def projectFaceVector(self, fV):
        """Given a vector, fV, in cartesian coordinates, this will project
        it onto the mesh using the normals

        :param numpy.ndarray fV: face vector with shape (nF, dim)
        :rtype: numpy.ndarray
        :return: projected face vector, (nF, )

        """
        if not isinstance(fV, np.ndarray):
            raise Exception('fV must be an ndarray')
        if not (
            len(fV.shape) == 2 and
            fV.shape[0] == self.nF and
            fV.shape[1] == self.dim
        ):
            raise Exception('fV must be an ndarray of shape (nF x dim)')
        return np.sum(fV*self.normals, 1)

    def projectEdgeVector(self, eV):
        """Given a vector, eV, in cartesian coordinates, this will project
        it onto the mesh using the tangents

        :param numpy.ndarray eV: edge vector with shape (nE, dim)
        :rtype: numpy.ndarray
        :return: projected edge vector, (nE, )

        """
        if not isinstance(eV, np.ndarray):
            raise Exception('eV must be an ndarray')
        if not (
            len(eV.shape) == 2 and
            eV.shape[0] == self.nE and
            eV.shape[1] == self.dim
        ):
            raise Exception('eV must be an ndarray of shape (nE x dim)')
        return np.sum(eV*self.tangents, 1)

    def save(self, filename='mesh.json', verbose=False):
        """
        Save the mesh to json
        :param str file: filename for saving the casing properties
        :param str directory: working directory for saving the file
        """

        f = os.path.abspath(filename)  # make sure we are working with abs path
        with open(f, 'w') as outfile:
            json.dump(self.serialize(), outfile)

        if verbose is True:
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
        docs for the :class:`~discretize.mixins.vtkModule.InterfaceVTK`.
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


class BaseRectangularMesh(BaseMesh):
    """
    BaseRectangularMesh
    """


    def __init__(self, n, x0=None, **kwargs):
        BaseMesh.__init__(self, n, x0=x0, **kwargs)

    @property
    def nCx(self):
        """Number of cells in the x direction

        :rtype: int
        :return: nCx
        """
        return int(self._n[0])

    @property
    def nCy(self):
        """Number of cells in the y direction

        :rtype: int
        :return: nCy or None if dim < 2
        """
        if self.dim < 2:
            return None
        return int(self._n[1])

    @property
    def nCz(self):
        """Number of cells in the z direction

        :rtype: int
        :return: nCz or None if dim < 3
        """
        if self.dim < 3:
            return None
        return int(self._n[2])

    @property
    def vnC(self):
        """Total number of cells in each direction

        :rtype: numpy.ndarray
        :return: [nCx, nCy, nCz]
        """
        return np.array(
            [x for x in [self.nCx, self.nCy, self.nCz] if x is not None],
            dtype=int
        )

    @property
    def nNx(self):
        """Number of nodes in the x-direction

        :rtype: int
        :return: nNx
        """
        return self.nCx + 1

    @property
    def nNy(self):
        """Number of nodes in the y-direction

        :rtype: int
        :return: nNy or None if dim < 2
        """
        if self.dim < 2:
            return None
        return self.nCy + 1

    @property
    def nNz(self):
        """Number of nodes in the z-direction

        :rtype: int
        :return: nNz or None if dim < 3
        """
        if self.dim < 3:
            return None
        return self.nCz + 1

    @property
    def vnN(self):
        """Total number of nodes in each direction

        :rtype: numpy.ndarray
        :return: [nNx, nNy, nNz]
        """
        return np.array(
            [x for x in [self.nNx, self.nNy, self.nNz] if x is not None],
            dtype=int
        )

    @property
    def vnEx(self):
        """Number of x-edges in each direction

        :rtype: numpy.ndarray
        :return: vnEx
        """
        return np.array(
            [x for x in [self.nCx, self.nNy, self.nNz] if x is not None],
            dtype=int
        )

    @property
    def vnEy(self):
        """Number of y-edges in each direction

        :rtype: numpy.ndarray
        :return: vnEy or None if dim < 2
        """
        if self.dim < 2:
            return None
        return np.array(
            [x for x in [self.nNx, self.nCy, self.nNz] if x is not None],
            dtype=int
        )

    @property
    def vnEz(self):
        """Number of z-edges in each direction

        :rtype: numpy.ndarray
        :return: vnEz or None if dim < 3
        """
        if self.dim < 3:
            return None
        return np.array(
            [x for x in [self.nNx, self.nNy, self.nCz] if x is not None],
            dtype=int
        )

    @property
    def vnFx(self):
        """Number of x-faces in each direction

        :rtype: numpy.ndarray
        :return: vnFx
        """
        return np.array(
            [x for x in [self.nNx, self.nCy, self.nCz] if x is not None],
            dtype=int
        )

    @property
    def vnFy(self):
        """Number of y-faces in each direction

        :rtype: numpy.ndarray
        :return: vnFy or None if dim < 2
        """
        if self.dim < 2:
            return None
        return np.array(
            [x for x in [self.nCx, self.nNy, self.nCz] if x is not None],
            dtype=int
        )

    @property
    def vnFz(self):
        """Number of z-faces in each direction

        :rtype: numpy.ndarray
        :return: vnFz or None if dim < 3
        """
        if self.dim < 3:
            return None
        return np.array(
            [x for x in [self.nCx, self.nCy, self.nNz] if x is not None],
            dtype=int
        )

    ##################################
    # Redo the numbering so they are dependent of the vector numbers
    ##################################

    @property
    def nC(self):
        """Total number of cells

        :rtype: int
        :return: nC
        """
        return int(self.vnC.prod())

    @property
    def nN(self):
        """Total number of nodes

        :rtype: int
        :return: nN
        """
        return int(self.vnN.prod())

    @property
    def nEx(self):
        """Number of x-edges

        :rtype: int
        :return: nEx
        """
        return int(self.vnEx.prod())

    @property
    def nEy(self):
        """Number of y-edges

        :rtype: int
        :return: nEy
        """
        if self.dim < 2:
            return
        return int(self.vnEy.prod())

    @property
    def nEz(self):
        """Number of z-edges

        :rtype: int
        :return: nEz
        """
        if self.dim < 3:
            return
        return int(self.vnEz.prod())

    @property
    def nFx(self):
        """Number of x-faces

        :rtype: int
        :return: nFx
        """
        return int(self.vnFx.prod())

    @property
    def nFy(self):
        """Number of y-faces

        :rtype: int
        :return: nFy
        """
        if self.dim < 2:
            return
        return int(self.vnFy.prod())

    @property
    def nFz(self):
        """Number of z-faces

        :rtype: int
        :return: nFz
        """
        if self.dim < 3:
            return
        return int(self.vnFz.prod())

    def r(self, x, xType='CC', outType='CC', format='V'):
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
                nx[0] = self.vnFx if xType == 'F' else self.vnEx
                nx[1] = self.vnFy if xType == 'F' else self.vnEy
                nx[2] = self.vnFz if xType == 'F' else self.vnEz

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
                    nn = self.vnFx if 'F' in xType else self.vnEx
                elif 'y' in xType:
                    nn = self.vnFy if 'F' in xType else self.vnEy
                elif 'z' in xType:
                    nn = self.vnFz if 'F' in xType else self.vnEz
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
