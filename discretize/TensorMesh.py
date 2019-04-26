from __future__ import print_function
import numpy as np

from discretize import utils

from .base import BaseRectangularMesh, BaseTensorMesh
from .View import TensorView
from .DiffOperators import DiffOperators
from .InnerProducts import InnerProducts
from .MeshIO import TensorMeshIO


class TensorMesh(
    BaseTensorMesh, BaseRectangularMesh, TensorView, DiffOperators,
    InnerProducts, TensorMeshIO
):
    """
    TensorMesh is a mesh class that deals with tensor product meshes.

    Any Mesh that has a constant width along the entire axis
    such that it can defined by a single width vector, called 'h'.

    .. plot::
        :include-source:

        import discretize

        hx = np.array([1, 1, 1])
        hy = np.array([1, 2])
        hz = np.array([1, 1, 1, 1])

        mesh = discretize.TensorMesh([hx, hy, hz])
        mesh.plotGrid()


    Example of a padded tensor mesh using
    :func:`discretize.utils.meshTensor`:

    .. plot::
        :include-source:

        import discretize
        mesh = discretize.TensorMesh([
            [(10, 10, -1.3), (10, 40), (10, 10, 1.3)],
            [(10, 10, -1.3), (10, 20)]
        ])
        mesh.plotGrid()

    For a quick tensor mesh on a (10x12x15) unit cube

    .. code:: python

        import discretize
        mesh = discretize.TensorMesh([10, 12, 15])

    """

    _meshType = 'TENSOR'

    def __init__(self, h=None, x0=None, **kwargs):
        BaseTensorMesh.__init__(self, h=h, x0=x0, **kwargs)

    def __repr__(self):
        """Plain text representation."""
        fmt = "\n  {}: {:,} cells\n\n".format(type(self).__name__, self.nC)
        fmt += 22*" "+"MESH EXTENT"+13*" "+"CELL WIDTH      FACTOR\n"
        fmt += "  dir    nC        min           max         min       max "
        fmt += "     max\n  ---   ---  "+27*"-"+"  "+18*"-"+"  ------\n"

        # Get attributes and put into table.
        attrs = self._repr_attributes()
        for i in range(self.dim):
            name = attrs['names'][i]
            iattr = attrs[name]
            fmt += "   {}".format(name)
            fmt += " {:6}".format(iattr['nC'])
            for p in ['min', 'max']:
                fmt += " {:13,.2f}".format(iattr[p])
            for p in ['h_min', 'h_max']:
                fmt += " {:9,.2f}".format(iattr[p])
            fmt += "{:8,.2f}".format(iattr['max_fact'])
            fmt += "\n"  # End row

        fmt += "\n"
        return fmt

    def _repr_html_(self):
        """HTML representation."""
        style = " style='padding: 5px 20px 5px 20px;'"

        fmt = "<table>\n"
        fmt += "  <tr>\n"
        fmt += "    <td style='font-weight: bold; font-size: 1.2em; text-align"
        fmt += ": center;' colspan='3'>{}</td\n>".format(type(self).__name__)
        fmt += "    <td style='font-size: 1.2em; text-align: center;'"
        fmt += "colspan='4'>{:,} cells</td>\n".format(self.nC)
        fmt += "  </tr>\n"

        fmt += "  <tr>\n"
        fmt += "    <th></th\n>"
        fmt += "    <th></th\n>"
        fmt += "    <th colspan='2'"+style+">MESH EXTENT</th\n>"
        fmt += "    <th colspan='2'"+style+">CELL WIDTH</th\n>"
        fmt += "    <th"+style+">FACTOR</th\n>"
        fmt += "  </tr\n>"

        fmt += "  <tr>\n"
        fmt += "    <th"+style+">dir</th>\n"
        fmt += "    <th"+style+">nC</th>\n"
        fmt += "    <th"+style+">min</th>\n"
        fmt += "    <th"+style+">max</th>\n"
        fmt += "    <th"+style+">min</th>\n"
        fmt += "    <th"+style+">max</th>\n"
        fmt += "    <th"+style+">max</th>\n"
        fmt += "  </tr>\n"

        # Get attributes and put into table.
        attrs = self._repr_attributes()
        for i in range(self.dim):
            name = attrs['names'][i]
            iattr = attrs[name]
            fmt += "  <tr>\n"  # Start row
            fmt += "    <td"+style+">{}</td>\n".format(name)
            fmt += "    <td"+style+">{}</td>\n".format(iattr['nC'])
            for p in ['min', 'max', 'h_min', 'h_max', 'max_fact']:
                fmt += "    <td"+style+">{:,.2f}</td>\n".format(iattr[p])
            fmt += "  </tr>\n"  # End row

        fmt += "</table>\n"
        return fmt

    # --------------- Geometries ---------------------
    @property
    def vol(self):
        """Construct cell volumes of the 3D model as 1d array."""
        if getattr(self, '_vol', None) is None:
            vh = self.h
            # Compute cell volumes
            if self.dim == 1:
                self._vol = utils.mkvc(vh[0])
            elif self.dim == 2:
                # Cell sizes in each direction
                self._vol = utils.mkvc(np.outer(vh[0], vh[1]))
            elif self.dim == 3:
                # Cell sizes in each direction
                self._vol = utils.mkvc(
                    np.outer(utils.mkvc(np.outer(vh[0], vh[1])), vh[2])
                )
        return self._vol

    @property
    def areaFx(self):
        """
        Area of the x-faces
        """
        if getattr(self, '_areaFx', None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute areas of cell faces
            if self.dim == 1:
                areaFx = np.ones(n[0]+1)
            elif self.dim == 2:
                areaFx = np.outer(np.ones(n[0]+1), vh[1])
            elif self.dim == 3:
                areaFx = np.outer(
                    np.ones(n[0]+1), utils.mkvc(np.outer(vh[1], vh[2]))
                )
            self._areaFx = utils.mkvc(areaFx)
        return self._areaFx

    @property
    def areaFy(self):
        """
        Area of the y-faces
        """
        if getattr(self, '_areaFy', None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute areas of cell faces
            if self.dim == 1:
                raise Exception('1D meshes do not have y-Faces')
            elif self.dim == 2:
                areaFy = np.outer(vh[0], np.ones(n[1]+1))
            elif self.dim == 3:
                areaFy = np.outer(
                    vh[0], utils.mkvc(np.outer(np.ones(n[1]+1), vh[2]))
                )
            self._areaFy = utils.mkvc(areaFy)
        return self._areaFy

    @property
    def areaFz(self):
        """
        Area of the z-faces
        """
        if getattr(self, '_areaFz', None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute areas of cell faces
            if self.dim == 1 or self.dim == 2:
                raise Exception(
                    '{}D meshes do not have z-Faces'.format(self.dim)
                )
            elif self.dim == 3:
                areaFz = np.outer(
                    vh[0], utils.mkvc(np.outer(vh[1], np.ones(n[2]+1)))
                )
            self._areaFz = utils.mkvc(areaFz)
        return self._areaFz

    @property
    def area(self):
        """Construct face areas of the 3D model as 1d array."""
        if self.dim == 1:
            return self.areaFx
        elif self.dim == 2:
            return np.r_[self.areaFx, self.areaFy]
        elif self.dim == 3:
            return np.r_[self.areaFx, self.areaFy, self.areaFz]

    @property
    def edgeEx(self):
        """x-edge lengths"""
        if getattr(self, '_edgeEx', None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute edge lengths
            if self.dim == 1:
                edgeEx = vh[0]
            elif self.dim == 2:
                edgeEx = np.outer(vh[0], np.ones(n[1]+1))
            elif self.dim == 3:
                edgeEx = np.outer(
                    vh[0],
                    utils.mkvc(np.outer(np.ones(n[1]+1), np.ones(n[2]+1)))
                )
            self._edgeEx = utils.mkvc(edgeEx)
        return self._edgeEx

    @property
    def edgeEy(self):
        """y-edge lengths"""
        if getattr(self, '_edgeEy', None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute edge lengths
            if self.dim == 1:
                raise Exception('1D meshes do not have y-edges')
            elif self.dim == 2:
                edgeEy = np.outer(np.ones(n[0]+1), vh[1])
            elif self.dim == 3:
                edgeEy = np.outer(
                    np.ones(n[0]+1),
                    utils.mkvc(np.outer(vh[1], np.ones(n[2]+1)))
                )
            self._edgeEy = utils.mkvc(edgeEy)
        return self._edgeEy

    @property
    def edgeEz(self):
        """z-edge lengths"""
        if getattr(self, '_edgeEz', None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute edge lengths
            if self.dim == 1 or self.dim == 2:
                raise Exception(
                    '{}D meshes do not have y-edges'.format(self.dim)
                )
            elif self.dim == 3:
                edgeEz = np.outer(
                    np.ones(n[0]+1),
                    utils.mkvc(np.outer(np.ones(n[1]+1), vh[2]))
                )
            self._edgeEz = utils.mkvc(edgeEz)
        return self._edgeEz

    @property
    def edge(self):
        """Construct edge legnths of the 3D model as 1d array."""
        if self.dim == 1:
            return self.edgeEx
        elif self.dim == 2:
            return np.r_[self.edgeEx, self.edgeEy]
        elif(self.dim == 3):
            return np.r_[self.edgeEx, self.edgeEy, self.edgeEz]
        return self._edge

    @property
    def faceBoundaryInd(self):
        """
        Find indices of boundary faces in each direction
        """
        if self.dim == 1:
            indxd = (self.gridFx == min(self.gridFx))
            indxu = (self.gridFx == max(self.gridFx))
            return indxd, indxu
        elif self.dim == 2:
            indxd = (self.gridFx[:, 0] == min(self.gridFx[:, 0]))
            indxu = (self.gridFx[:, 0] == max(self.gridFx[:, 0]))
            indyd = (self.gridFy[:, 1] == min(self.gridFy[:, 1]))
            indyu = (self.gridFy[:, 1] == max(self.gridFy[:, 1]))
            return indxd, indxu, indyd, indyu
        elif self.dim == 3:
            indxd = (self.gridFx[:, 0] == min(self.gridFx[:, 0]))
            indxu = (self.gridFx[:, 0] == max(self.gridFx[:, 0]))
            indyd = (self.gridFy[:, 1] == min(self.gridFy[:, 1]))
            indyu = (self.gridFy[:, 1] == max(self.gridFy[:, 1]))
            indzd = (self.gridFz[:, 2] == min(self.gridFz[:, 2]))
            indzu = (self.gridFz[:, 2] == max(self.gridFz[:, 2]))
            return indxd, indxu, indyd, indyu, indzd, indzu

    @property
    def cellBoundaryInd(self):
        """
        Find indices of boundary faces in each direction
        """
        if self.dim == 1:
            indxd = (self.gridCC == min(self.gridCC))
            indxu = (self.gridCC == max(self.gridCC))
            return indxd, indxu
        elif self.dim == 2:
            indxd = (self.gridCC[:, 0] == min(self.gridCC[:, 0]))
            indxu = (self.gridCC[:, 0] == max(self.gridCC[:, 0]))
            indyd = (self.gridCC[:, 1] == min(self.gridCC[:, 1]))
            indyu = (self.gridCC[:, 1] == max(self.gridCC[:, 1]))
            return indxd, indxu, indyd, indyu
        elif self.dim == 3:
            indxd = (self.gridCC[:, 0] == min(self.gridCC[:, 0]))
            indxu = (self.gridCC[:, 0] == max(self.gridCC[:, 0]))
            indyd = (self.gridCC[:, 1] == min(self.gridCC[:, 1]))
            indyu = (self.gridCC[:, 1] == max(self.gridCC[:, 1]))
            indzd = (self.gridCC[:, 2] == min(self.gridCC[:, 2]))
            indzu = (self.gridCC[:, 2] == max(self.gridCC[:, 2]))
            return indxd, indxu, indyd, indyu, indzd, indzu

    def _repr_attributes(self):
        """Attributes for the representation of the mesh."""

        attrs = {}
        attrs['names'] = ['x', 'y', 'z'][:self.dim]

        # Loop over dimensions.
        for i in range(self.dim):
            name = attrs['names'][i]  # Name of this dimension
            attrs[name] = {}

            # Get min/max node.
            n_vector = getattr(self, 'vectorN'+name)
            attrs[name]['min'] = np.nanmin(n_vector)
            attrs[name]['max'] = np.nanmax(n_vector)

            # Get min/max cell width.
            h_vector = getattr(self, 'h'+name)
            attrs[name]['h_min'] = np.nanmin(h_vector)
            attrs[name]['h_max'] = np.nanmax(h_vector)

            # Get max stretching factor.
            if len(h_vector) < 2:
                attrs[name]['max_fact'] = 1.0
            else:
                attrs[name]['max_fact'] = np.nanmax(
                    np.r_[h_vector[:-1]/h_vector[1:],
                          h_vector[1:]/h_vector[:-1]]
                )

            # Add number of cells.
            attrs[name]['nC'] = getattr(self, 'nC'+name)

        return attrs
