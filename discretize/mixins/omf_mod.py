"""Module for ``omf`` interaction with ``discretize``."""
import numpy as np
import discretize


def omf():
    """Lazy loading omf."""
    import omf

    return omf


def _ravel_data_array(arr, nx, ny, nz):
    """Ravel an array from discretize ordering to omf ordering.

    Converts a 1D numpy array from ``discretize`` ordering (x, y, z)
    to a flattened 1D numpy array with ``OMF`` ordering (z, y, x)

    In ``discretize``, three-dimensional data are frequently organized within a
    1D numpy array whose elements are ordered along the x-axis, then the y-axis,
    then the z-axis. **_ravel_data_array** converts the input array
    (discretize format) to a 1D numpy array ordered according to the open
    mining format; which is ordered along
    the z-axis, then the y-axis, then the x-axis.

    Parameters
    ----------
    arr : numpy.ndarray
        A 1D vector or nD array ordered along the x, then y, then z axes
    nx : int
        Number of cells along the x-axis
    ny : int
        Number of cells along the y-axis
    nz : int
        Number of cells along the z-axis

    Returns
    -------
    numpy.ndarray (n_cells)
        A flattened 1D array ordered according to the open mining format

    Examples
    --------
    To demonstrate the reordering, we design a small 3D tensor mesh.
    We print a numpy array with the xyz locations of cell the centers using the
    original ordering (discretize). We then re-order the cell locations according to OMF.

    >>> from discretize import TensorMesh
    >>> import numpy as np

    >>> hx = np.ones(4)
    >>> hy = 2*np.ones(3)
    >>> hz = 3*np.ones(2)
    >>> mesh = TensorMesh([hx, hy, hz])

    >>> dim = (mesh.nCz, mesh.nCy, mesh.nCx)  # OMF orderting
    >>> xc = np.reshape(mesh.cell_centers[:, 0], dim, order="C").ravel(order="F")
    >>> yc = np.reshape(mesh.cell_centers[:, 1], dim, order="C").ravel(order="F")
    >>> zc = np.reshape(mesh.cell_centers[:, 2], dim, order="C").ravel(order="F")

    .. collapse:: Original ordering. Click to expand

        >>> mesh.cell_centers
        array([[0.5, 1. , 1.5],
               [1.5, 1. , 1.5],
               [2.5, 1. , 1.5],
               [3.5, 1. , 1.5],
               [0.5, 3. , 1.5],
               [1.5, 3. , 1.5],
               [2.5, 3. , 1.5],
               [3.5, 3. , 1.5],
               [0.5, 5. , 1.5],
               [1.5, 5. , 1.5],
               [2.5, 5. , 1.5],
               [3.5, 5. , 1.5],
               [0.5, 1. , 4.5],
               [1.5, 1. , 4.5],
               [2.5, 1. , 4.5],
               [3.5, 1. , 4.5],
               [0.5, 3. , 4.5],
               [1.5, 3. , 4.5],
               [2.5, 3. , 4.5],
               [3.5, 3. , 4.5],
               [0.5, 5. , 4.5],
               [1.5, 5. , 4.5],
               [2.5, 5. , 4.5],
               [3.5, 5. , 4.5]])

    .. collapse:: OMF ordering. Click to expand

        >>> np.c_[xc, yc, zc]
        array([[0.5, 1. , 1.5],
               [0.5, 1. , 4.5],
               [0.5, 3. , 1.5],
               [0.5, 3. , 4.5],
               [0.5, 5. , 1.5],
               [0.5, 5. , 4.5],
               [1.5, 1. , 1.5],
               [1.5, 1. , 4.5],
               [1.5, 3. , 1.5],
               [1.5, 3. , 4.5],
               [1.5, 5. , 1.5],
               [1.5, 5. , 4.5],
               [2.5, 1. , 1.5],
               [2.5, 1. , 4.5],
               [2.5, 3. , 1.5],
               [2.5, 3. , 4.5],
               [2.5, 5. , 1.5],
               [2.5, 5. , 4.5],
               [3.5, 1. , 1.5],
               [3.5, 1. , 4.5],
               [3.5, 3. , 1.5],
               [3.5, 3. , 4.5],
               [3.5, 5. , 1.5],
               [3.5, 5. , 4.5]])
    """
    dim = (nz, ny, nx)
    return np.reshape(arr, dim, order="C").ravel(order="F")


def _unravel_data_array(arr, nx, ny, nz):
    """Unravel an array from omf ordering to discretize ordering.

    Converts a 1D numpy array from ``OMF`` ordering (z, y, x)
    to a flattened 1D numpy array with ``discretize`` ordering (x, y, z)

    In ``OMF``, three-dimensional data are organized within a
    1D numpy array whose elements are ordered along the z-axis, then the y-axis,
    then the x-axis. **_unravel_data_array** converts the input array
    (OMF format) to a 1D numpy array ordered according to ``discretize``;
    which is ordered along the x-axis, then the y-axis, then the y-axis.

    Parameters
    ----------
    arr : numpy.ndarray
        A 1D vector or nD array ordered along the z, then y, then x axes
    nx : int
        Number of cells along the x-axis
    ny : int
        Number of cells along the y-axis
    nz : int
        Number of cells along the z-axis

    Returns
    -------
    (n_cells) numpy.ndarray
        A flattened 1D array ordered according to the discretize format

    """
    dim = (nz, ny, nx)
    return np.reshape(arr, dim, order="F").ravel(order="C")


class InterfaceOMF(object):
    """Convert between ``omf`` and ``discretize`` objects.

    The ``InterfaceOMF`` class was designed for easy conversion between
    ``discretize`` objects and `open mining format <https://www.seequent.com/the-open-mining-format/>`__ (OMF) objects.
    Examples include: meshes, models and data arrays.
    """

    def _tensor_mesh_to_omf(mesh, models=None):
        """Convert a TensorMesh to an omf object.

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
        geometry = omf().VolumeGridGeometry()
        # Set tensors
        tensors = mesh.h
        if len(tensors) < 1:
            raise RuntimeError(
                "Your mesh is empty... fill it out before converting to OMF"
            )
        elif len(tensors) == 1:
            geometry.tensor_u = tensors[0]
            geometry.tensor_v = np.array(
                [
                    0.0,
                ]
            )
            geometry.tensor_w = np.array(
                [
                    0.0,
                ]
            )
        elif len(tensors) == 2:
            geometry.tensor_u = tensors[0]
            geometry.tensor_v = tensors[1]
            geometry.tensor_w = np.array(
                [
                    0.0,
                ]
            )
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
        geometry.origin = mesh.origin
        # Make sure the geometry is built correctly
        geometry.validate()
        # Make the volume elemet (the OMF object)
        omfmesh = omf().VolumeElement(
            geometry=geometry,
        )
        # Add model data arrays onto the cells of the mesh
        omfmesh.data = []
        for name, arr in models.items():
            data = omf().ScalarData(
                name=name,
                array=_ravel_data_array(arr, *mesh.shape_cells),
                location="cells",
            )
            omfmesh.data.append(data)
        # Validate to make sure a proper OMF object is returned to the user
        omfmesh.validate()
        return omfmesh

    def _tree_mesh_to_omf(mesh, models=None):
        raise NotImplementedError("Not possible until OMF v2 is released.")

    def _curvilinear_mesh_to_omf(mesh, models=None):
        raise NotImplementedError("Not currently possible.")

    def _cyl_mesh_to_omf(mesh, models=None):
        raise NotImplementedError("Not currently possible.")

    def to_omf(mesh, models=None):
        """Convert to an ``omf`` data object.

        Convert this mesh object to its proper ``omf`` data object with
        the given model dictionary as the cell data of that dataset.

        Parameters
        ----------
        models : dict of [str, (n_cells) numpy.ndarray], optional
            Name('s) and array('s).

        Returns
        -------
        omf.volume.VolumeElement
        """
        # TODO: mesh.validate()
        converters = {
            # TODO: 'tree' : InterfaceOMF._tree_mesh_to_omf,
            "tensor": InterfaceOMF._tensor_mesh_to_omf,
            # TODO: 'curv' : InterfaceOMF._curvilinear_mesh_to_omf,
            # TODO: 'CylindricalMesh' : InterfaceOMF._cyl_mesh_to_omf,
        }
        key = mesh._meshType.lower()
        try:
            convert = converters[key]
        except KeyError:
            raise RuntimeError(
                "Mesh type `{}` is not currently supported for OMF conversion.".format(
                    key
                )
            )
        # Convert the data object
        return convert(mesh, models=models)

    @staticmethod
    def _omf_volume_to_tensor(element):
        """Convert an :class:`omf.VolumeElement` to :class:`discretize.TensorMesh`."""
        geometry = element.geometry
        h = [geometry.tensor_u, geometry.tensor_v, geometry.tensor_w]
        mesh = discretize.TensorMesh(h)
        mesh.axis_u = geometry.axis_u
        mesh.axis_v = geometry.axis_v
        mesh.axis_w = geometry.axis_w
        mesh.origin = geometry.origin

        data_dict = {}
        for data in element.data:
            # NOTE: this is agnostic about data location - i.e. nodes vs cells
            data_dict[data.name] = _unravel_data_array(
                np.array(data.array), *mesh.shape_cells
            )

        # Return TensorMesh and data dictionary
        return mesh, data_dict

    @staticmethod
    def from_omf(element):
        """Convert an ``omf`` object to a ``discretize`` mesh.

        Convert an OMF element to it's proper ``discretize`` type.
        Automatically determines the output type. Returns both the mesh and a
        dictionary of model arrays.

        Parameters
        ----------
        element : omf.volume.VolumeElement
            The open mining format volume element object

        Returns
        -------
        mesh : discretize.TensorMesh
            The returned mesh type will be appropriately based on the input `element`.
        models : dict of [str, (n_cells) numpy.ndarray]
            The models contained in `element`

        Notes
        -----
        Currently only :class:discretize.TensorMesh is supported.
        """
        element.validate()
        converters = {
            omf().VolumeElement.__name__: InterfaceOMF._omf_volume_to_tensor,
        }
        key = element.__class__.__name__
        try:
            convert = converters[key]
        except KeyError:
            raise RuntimeError(
                "OMF type `{}` is not currently supported for conversion.".format(key)
            )
        # Convert the data object
        return convert(element)
