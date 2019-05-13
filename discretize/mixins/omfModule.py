"""
A class for converting ``discretize`` meshes to OMF objects
"""

import omf
import numpy as np


def ravel_data_array(arr, mesh):
    """Ravel's a numpy array into proper order for passing to the OMF
    specification from ``discretize``/UBC formats
    """
    dim = (mesh.nCz, mesh.nCy, mesh.nCx)
    return np.reshape(arr, dim).ravel(order='F')


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
                                  array=ravel_data_array(arr, mesh),
                                  location='cells')
            omfmesh.data.append(data)
        # Validate to make sure a proper OMF object is returned to the user
        omfmesh.validate()
        return omfmesh


    def _tree_mesh_to_omf(mesh, models=None):
        raise NotImplementedError()


    def _curvilinear_mesh_to_omf(mesh, models=None):
        raise NotImplementedError()


    def _cyl_mesh_to_omf(mesh, models=None):
        raise NotImplementedError()


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
