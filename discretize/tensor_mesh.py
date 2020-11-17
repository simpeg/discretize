import numpy as np


from discretize.base import BaseRectangularMesh, BaseTensorMesh
from discretize.operators import DiffOperators, InnerProducts
from discretize.base.mesh_io import TensorMeshIO
from discretize.utils import mkvc
from discretize.utils.code_utils import deprecate_property


class TensorMesh(
    BaseTensorMesh, BaseRectangularMesh, DiffOperators, InnerProducts, TensorMeshIO
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
        mesh.plot_grid()


    Example of a padded tensor mesh using
    :func:`discretize.utils.unpack_widths`:

    .. plot::
        :include-source:

        import discretize
        mesh = discretize.TensorMesh([
            [(10, 10, -1.3), (10, 40), (10, 10, 1.3)],
            [(10, 10, -1.3), (10, 20)]
        ])
        mesh.plot_grid()

    For a quick tensor mesh on a (10x12x15) unit cube

    .. code:: python

        import discretize
        mesh = discretize.TensorMesh([10, 12, 15])

    """

    _meshType = "TENSOR"
    _aliases = {
        **DiffOperators._aliases,
        **BaseRectangularMesh._aliases,
        **BaseTensorMesh._aliases,
    }

    def __init__(self, h=None, origin=None, **kwargs):
        BaseTensorMesh.__init__(self, h=h, origin=origin, **kwargs)

    def __repr__(self):
        """Plain text representation."""
        fmt = "\n  {}: {:,} cells\n\n".format(type(self).__name__, self.nC)
        fmt += 22 * " " + "MESH EXTENT" + 13 * " " + "CELL WIDTH      FACTOR\n"
        fmt += "  dir    nC        min           max         min       max "
        fmt += "     max\n  ---   ---  " + 27 * "-" + "  " + 18 * "-" + "  ------\n"

        # Get attributes and put into table.
        attrs = self._repr_attributes()
        for i in range(self.dim):
            name = attrs["names"][i]
            iattr = attrs[name]
            fmt += "   {}".format(name)
            fmt += " {:6}".format(iattr["nC"])
            for p in ["min", "max"]:
                fmt += " {:13,.2f}".format(iattr[p])
            for p in ["h_min", "h_max"]:
                fmt += " {:9,.2f}".format(iattr[p])
            fmt += "{:8,.2f}".format(iattr["max_fact"])
            fmt += "\n"  # End row

        fmt += "\n"
        return fmt

    def _repr_html_(self):
        """HTML representation."""
        style = " style='padding: 5px 20px 5px 20px;'"

        fmt = "<table>\n"
        fmt += "  <tr>\n"
        fmt += "    <td style='font-weight: bold; font-size: 1.2em; text-align"
        fmt += ": center;' colspan='3'>{}</td>\n".format(type(self).__name__)
        fmt += "    <td style='font-size: 1.2em; text-align: center;'"
        fmt += "colspan='4'>{:,} cells</td>\n".format(self.nC)
        fmt += "  </tr>\n"

        fmt += "  <tr>\n"
        fmt += "    <th></th>\n"
        fmt += "    <th></th>\n"
        fmt += "    <th colspan='2'" + style + ">MESH EXTENT</th>\n"
        fmt += "    <th colspan='2'" + style + ">CELL WIDTH</th>\n"
        fmt += "    <th" + style + ">FACTOR</th>\n"
        fmt += "  </tr>\n"

        fmt += "  <tr>\n"
        fmt += "    <th" + style + ">dir</th>\n"
        fmt += "    <th" + style + ">nC</th>\n"
        fmt += "    <th" + style + ">min</th>\n"
        fmt += "    <th" + style + ">max</th>\n"
        fmt += "    <th" + style + ">min</th>\n"
        fmt += "    <th" + style + ">max</th>\n"
        fmt += "    <th" + style + ">max</th>\n"
        fmt += "  </tr>\n"

        # Get attributes and put into table.
        attrs = self._repr_attributes()
        for i in range(self.dim):
            name = attrs["names"][i]
            iattr = attrs[name]
            fmt += "  <tr>\n"  # Start row
            fmt += "    <td" + style + ">{}</td>\n".format(name)
            fmt += "    <td" + style + ">{}</td>\n".format(iattr["nC"])
            for p in ["min", "max", "h_min", "h_max", "max_fact"]:
                fmt += "    <td" + style + ">{:,.2f}</td>\n".format(iattr[p])
            fmt += "  </tr>\n"  # End row

        fmt += "</table>\n"
        return fmt

    # --------------- Geometries ---------------------
    @property
    def cell_volumes(self):
        """Construct cell volumes of the 3D model as 1d array."""
        if getattr(self, "_cell_volumes", None) is None:
            vh = self.h
            # Compute cell volumes
            if self.dim == 1:
                self._cell_volumes = mkvc(vh[0])
            elif self.dim == 2:
                # Cell sizes in each direction
                self._cell_volumes = mkvc(np.outer(vh[0], vh[1]))
            elif self.dim == 3:
                # Cell sizes in each direction
                self._cell_volumes = mkvc(np.outer(mkvc(np.outer(vh[0], vh[1])), vh[2]))
        return self._cell_volumes

    @property
    def face_x_areas(self):
        """
        Area of the x-faces
        """
        if getattr(self, "_face_x_areas", None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute areas of cell faces
            if self.dim == 1:
                areaFx = np.ones(n[0] + 1)
            elif self.dim == 2:
                areaFx = np.outer(np.ones(n[0] + 1), vh[1])
            elif self.dim == 3:
                areaFx = np.outer(np.ones(n[0] + 1), mkvc(np.outer(vh[1], vh[2])))
            self._face_x_areas = mkvc(areaFx)
        return self._face_x_areas

    @property
    def face_y_areas(self):
        """
        Area of the y-faces
        """
        if getattr(self, "_face_y_areas", None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute areas of cell faces
            if self.dim == 1:
                raise Exception("1D meshes do not have y-Faces")
            elif self.dim == 2:
                areaFy = np.outer(vh[0], np.ones(n[1] + 1))
            elif self.dim == 3:
                areaFy = np.outer(vh[0], mkvc(np.outer(np.ones(n[1] + 1), vh[2])))
            self._face_y_areas = mkvc(areaFy)
        return self._face_y_areas

    @property
    def face_z_areas(self):
        """
        Area of the z-faces
        """
        if getattr(self, "_face_z_areas", None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute areas of cell faces
            if self.dim == 1 or self.dim == 2:
                raise Exception("{}D meshes do not have z-Faces".format(self.dim))
            elif self.dim == 3:
                areaFz = np.outer(vh[0], mkvc(np.outer(vh[1], np.ones(n[2] + 1))))
            self._face_z_areas = mkvc(areaFz)
        return self._face_z_areas

    @property
    def face_areas(self):
        """Construct face areas of the 3D model as 1d array."""
        if self.dim == 1:
            return self.face_x_areas
        elif self.dim == 2:
            return np.r_[self.face_x_areas, self.face_y_areas]
        elif self.dim == 3:
            return np.r_[self.face_x_areas, self.face_y_areas, self.face_z_areas]

    @property
    def edge_x_lengths(self):
        """x-edge lengths"""
        if getattr(self, "_edge_x_lengths", None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute edge lengths
            if self.dim == 1:
                edgeEx = vh[0]
            elif self.dim == 2:
                edgeEx = np.outer(vh[0], np.ones(n[1] + 1))
            elif self.dim == 3:
                edgeEx = np.outer(
                    vh[0], mkvc(np.outer(np.ones(n[1] + 1), np.ones(n[2] + 1)))
                )
            self._edge_x_lengths = mkvc(edgeEx)
        return self._edge_x_lengths

    @property
    def edge_y_lengths(self):
        """y-edge lengths"""
        if getattr(self, "_edge_y_lengths", None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute edge lengths
            if self.dim == 1:
                raise Exception("1D meshes do not have y-edges")
            elif self.dim == 2:
                edgeEy = np.outer(np.ones(n[0] + 1), vh[1])
            elif self.dim == 3:
                edgeEy = np.outer(
                    np.ones(n[0] + 1), mkvc(np.outer(vh[1], np.ones(n[2] + 1)))
                )
            self._edge_y_lengths = mkvc(edgeEy)
        return self._edge_y_lengths

    @property
    def edge_z_lengths(self):
        """z-edge lengths"""
        if getattr(self, "_edge_z_lengths", None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute edge lengths
            if self.dim == 1 or self.dim == 2:
                raise Exception("{}D meshes do not have y-edges".format(self.dim))
            elif self.dim == 3:
                edgeEz = np.outer(
                    np.ones(n[0] + 1), mkvc(np.outer(np.ones(n[1] + 1), vh[2]))
                )
            self._edge_z_lengths = mkvc(edgeEz)
        return self._edge_z_lengths

    @property
    def edge_lengths(self):
        """Construct edge legnths of the 3D model as 1d array."""
        if self.dim == 1:
            return self.edge_x_lengths
        elif self.dim == 2:
            return np.r_[self.edge_x_lengths, self.edge_y_lengths]
        elif self.dim == 3:
            return np.r_[self.edge_x_lengths, self.edge_y_lengths, self.edge_z_lengths]
        return self._edge

    @property
    def face_boundary_indices(self):
        """
        Find indices of boundary faces in each direction
        """
        if self.dim == 1:
            indxd = self.gridFx == min(self.gridFx)
            indxu = self.gridFx == max(self.gridFx)
            return indxd, indxu
        elif self.dim == 2:
            indxd = self.gridFx[:, 0] == min(self.gridFx[:, 0])
            indxu = self.gridFx[:, 0] == max(self.gridFx[:, 0])
            indyd = self.gridFy[:, 1] == min(self.gridFy[:, 1])
            indyu = self.gridFy[:, 1] == max(self.gridFy[:, 1])
            return indxd, indxu, indyd, indyu
        elif self.dim == 3:
            indxd = self.gridFx[:, 0] == min(self.gridFx[:, 0])
            indxu = self.gridFx[:, 0] == max(self.gridFx[:, 0])
            indyd = self.gridFy[:, 1] == min(self.gridFy[:, 1])
            indyu = self.gridFy[:, 1] == max(self.gridFy[:, 1])
            indzd = self.gridFz[:, 2] == min(self.gridFz[:, 2])
            indzu = self.gridFz[:, 2] == max(self.gridFz[:, 2])
            return indxd, indxu, indyd, indyu, indzd, indzu

    @property
    def cell_boundary_indices(self):
        """
        Find indices of boundary faces in each direction
        """
        if self.dim == 1:
            indxd = self.gridCC == min(self.gridCC)
            indxu = self.gridCC == max(self.gridCC)
            return indxd, indxu
        elif self.dim == 2:
            indxd = self.gridCC[:, 0] == min(self.gridCC[:, 0])
            indxu = self.gridCC[:, 0] == max(self.gridCC[:, 0])
            indyd = self.gridCC[:, 1] == min(self.gridCC[:, 1])
            indyu = self.gridCC[:, 1] == max(self.gridCC[:, 1])
            return indxd, indxu, indyd, indyu
        elif self.dim == 3:
            indxd = self.gridCC[:, 0] == min(self.gridCC[:, 0])
            indxu = self.gridCC[:, 0] == max(self.gridCC[:, 0])
            indyd = self.gridCC[:, 1] == min(self.gridCC[:, 1])
            indyu = self.gridCC[:, 1] == max(self.gridCC[:, 1])
            indzd = self.gridCC[:, 2] == min(self.gridCC[:, 2])
            indzu = self.gridCC[:, 2] == max(self.gridCC[:, 2])
            return indxd, indxu, indyd, indyu, indzd, indzu

    def _repr_attributes(self):
        """Attributes for the representation of the mesh."""

        attrs = {}
        attrs["names"] = ["x", "y", "z"][: self.dim]

        # Loop over dimensions.
        for i in range(self.dim):
            name = attrs["names"][i]  # Name of this dimension
            attrs[name] = {}

            # Get min/max node.
            n_vector = getattr(self, "nodes_" + name)
            attrs[name]["min"] = np.nanmin(n_vector)
            attrs[name]["max"] = np.nanmax(n_vector)

            # Get min/max cell width.
            h_vector = self.h[i]
            attrs[name]["h_min"] = np.nanmin(h_vector)
            attrs[name]["h_max"] = np.nanmax(h_vector)

            # Get max stretching factor.
            if len(h_vector) < 2:
                attrs[name]["max_fact"] = 1.0
            else:
                attrs[name]["max_fact"] = np.nanmax(
                    np.r_[h_vector[:-1] / h_vector[1:], h_vector[1:] / h_vector[:-1]]
                )

            # Add number of cells.
            attrs[name]["nC"] = self.shape_cells[i]

        return attrs

    # DEPRECATIONS
    vol = deprecate_property("cell_volumes", "vol", removal_version="1.0.0")
    areaFx = deprecate_property("face_x_areas", "areaFx", removal_version="1.0.0")
    areaFy = deprecate_property("face_y_areas", "areaFy", removal_version="1.0.0")
    areaFz = deprecate_property("face_z_areas", "areaFz", removal_version="1.0.0")
    area = deprecate_property("face_areas", "area", removal_version="1.0.0")
    edgeEx = deprecate_property("edge_x_lengths", "edgeEx", removal_version="1.0.0")
    edgeEy = deprecate_property("edge_y_lengths", "edgeEy", removal_version="1.0.0")
    edgeEz = deprecate_property("edge_z_lengths", "edgeEz", removal_version="1.0.0")
    edge = deprecate_property("edge_lengths", "edge", removal_version="1.0.0")
    faceBoundaryInd = deprecate_property(
        "face_boundary_indices", "faceBoundaryInd", removal_version="1.0.0"
    )
    cellBoundaryInd = deprecate_property(
        "cell_boundary_indices", "cellBoundaryInd", removal_version="1.0.0"
    )
