"""
A class for converting ``discretize`` meshes to OMF objects
"""

import omf
import numpy as np


import discretize


def ravel_data_array(arr, nx, ny, nz):
    """Ravel's a numpy array into proper order for passing to the OMF
    specification from ``discretize``/UBC formats
    """
    dim = (nz, ny, nx)
    return np.reshape(arr, dim, order='C').ravel(order='F')


def unravel_data_array(arr, nx, ny, nz):
    """Unravel's a numpy array from the OMF specification to
    ``discretize``/UBC formats - the is the inverse of ``ravel_data_array``
    """
    dim = (nz, ny, nx)
    return np.reshape(arr, dim, order='F').ravel(order='C')


class InterfaceOMF(object):


    def _tensor_mesh_to_omf(mesh, models=None):
        """
        Constructs an :class:`omf.VolumeElement` object of this tensor mesh and
        the given models as cell data of that grid.

        Parameters
        ----------

        mesh : discretize.TensorMesh
            The tensor mesh to convert to a :class:`omf.VolumeElement`

        models : dict(numpy.ndarray)
            Name('s) and array('s). Match number of cells

        """
        if models is None:
            models = {}
        # Make the geometry
        geometry = omf.VolumeGridGeometry()
        # Set tensors
        tensors = mesh.h
        if len(tensors) < 1:
            raise RuntimeError("Your mesh is empty... fill it out before converting to OMF")
        elif len(tensors) == 1:
            geometry.tensor_u = tensors[0]
            geometry.tensor_v = np.array([0.0,])
            geometry.tensor_w = np.array([0.0,])
        elif len(tensors) == 2:
            geometry.tensor_u = tensors[0]
            geometry.tensor_v = tensors[1]
            geometry.tensor_w = np.array([0.0,])
        elif len(tensors) == 3:
            geometry.tensor_u = tensors[0]
            geometry.tensor_v = tensors[1]
            geometry.tensor_w = tensors[2]
        else:
            raise RuntimeError("This mesh is too high-dimensional for OMF")
        # Set rotation axes
        geometry.axis_u = mesh.axis_u
        geometry.axis_v = mesh.axis_v
        geometry.axis_w = mesh.axis_w
        # Set the origin
        geometry.origin = mesh.x0
        # Make sure the geometry is built correctly
        geometry.validate()
        # Make the volume elemet (the OMF object)
        omfmesh = omf.VolumeElement(
                geometry=geometry,
            )
        # Add model data arrays onto the cells of the mesh
        omfmesh.data = []
        for name, arr in models.items():
            data = omf.ScalarData(name=name,
                                  array=ravel_data_array(arr, mesh.nCx, mesh.nCy, mesh.nCz),
                                  location='cells')
            omfmesh.data.append(data)
        # Validate to make sure a proper OMF object is returned to the user
        omfmesh.validate()
        return omfmesh


    def _tree_mesh_to_omf(mesh, models=None):
        raise NotImplementedError('Not possible until OMF v2 is released.')


    def _curvilinear_mesh_to_omf(mesh, models=None):
        raise NotImplementedError('Not currently possible.')


    def _cyl_mesh_to_omf(mesh, models=None):
        raise NotImplementedError('Not currently possible.')


    def to_omf(mesh, models=None):
        """Convert this mesh object to it's proper ``omf`` data object with
        the given model dictionary as the cell data of that dataset.

        Parameters
        ----------

        models : dict(numpy.ndarray)
            Name('s) and array('s). Match number of cells

        """
        # TODO: mesh.validate()
        converters = {
            # TODO: 'tree' : InterfaceOMF._tree_mesh_to_omf,
            'tensor' : InterfaceOMF._tensor_mesh_to_omf,
            # TODO: 'curv' : InterfaceOMF._curvilinear_mesh_to_omf,
            # TODO: 'CylMesh' : InterfaceOMF._cyl_mesh_to_omf,
            }
        key = mesh._meshType.lower()
        try:
            convert = converters[key]
        except KeyError:
            raise RuntimeError('Mesh type `{}` is not currently supported for OMF conversion.'.format(key))
        # Convert the data object
        return convert(mesh, models=models)


    @staticmethod
    def _omf_volume_to_tensor(element):
        """Convert an :class:`omf.VolumeElement` to :class:`discretize.TensorMesh`
        """
        geometry = element.geometry
        h = [geometry.tensor_u, geometry.tensor_v, geometry.tensor_w]
        mesh = discretize.TensorMesh(h)
        mesh.axis_u = geometry.axis_u
        mesh.axis_v = geometry.axis_v
        mesh.axis_w = geometry.axis_w
        mesh.x0 = geometry.origin

        data_dict = {}
        for data in element.data:
            # NOTE: this is agnostic about data location - i.e. nodes vs cells
            data_dict[data.name] = unravel_data_array(np.array(data.array), mesh.nCx, mesh.nCy, mesh.nCz)

        # Return TensorMesh and data dictionary
        return mesh, data_dict


    @staticmethod
    def from_omf(element):
        """Convert an OMF element to it's proper ``discretize`` type.
        Automatically determines the output type. Returns both the mesh and a
        dictionary of model arrays.
        """
        element.validate()
        converters = {
            omf.VolumeElement.__name__ : InterfaceOMF._omf_volume_to_tensor,
            }
        key = element.__class__.__name__
        try:
            convert = converters[key]
        except KeyError:
            raise RuntimeError('OMF type `{}` is not currently supported for conversion.'.format(key))
        # Convert the data object
        return convert(element)
