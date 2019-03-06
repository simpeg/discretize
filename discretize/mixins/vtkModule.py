"""
This module provides a way for ``discretize`` meshes to be
converted to VTK data objects (and back when possible) if the
`VTK Python package`_ is available.
The :class:`discretize.mixins.vtkModule.vtkInterface` class becomes inherrited
by all mesh objects and allows users to directly convert any given mesh by
calling that mesh's ``toVTK()`` method
(note that this method will not be available if VTK is not available).

.. _`VTK Python package`: https://pypi.org/project/vtk/

This functionality was originally developed so that discretize could be
interoperable with PVGeo_, providing a direct interface for discretize meshes
within ParaView and other VTK powered platforms. This interoperablity allows
users to visualize their finite volume meshes and model data from discretize
along side all their other georeferenced datasets in a common rendering
environment.

.. _PVGeo: http://pvgeo.org
.. _vtki: http://www.vtki.org

Another notable VTK powered software platforms is ``vtki`` (see vtki_ docs)
which provides a direct interface to the VTK software library through accesible
Python data structures and NumPy arrays::

    pip install vtki

By default, the ``toVTK()`` method will return a ``vtki`` data object so that
users can immediately start visualizing their data in 3D.

See :ref:`vtki_demo_ref` for an example of the types of integrated
visualizations that are possible leveraging the link between discretize, vtki_,
and PVGeo_:

.. image:: ../images/vtki_laguna_del_maule.png
   :target: http://pvgeo.org
   :alt: PVGeo Example Visualization

.. admonition:: Laguna del Maule Bouguer Gravity
   :class: note

    This data scene is was produced from the `Laguna del Maule Bouguer Gravity`_
    example provided by Craig Miller (see Maule volcanic field, Chile. Refer to
    Miller et al 2016 EPSL for full details.)

    The rendering below shows several data sets and a model integrated together:

    * `Point Data`: the Bouguer gravity anomalies
    * Topography Surface
    * `Inverted Model`: The model has been both sliced and thresholded for low values

.. _`Laguna del Maule Bouguer Gravity`: http://docs.simpeg.xyz/content/examples/04-grav/plot_laguna_del_maule_inversion.html#sphx-glr-content-examples-04-grav-plot-laguna-del-maule-inversion-py

"""
import os
import numpy as np

# from ..utils import cyl2cart

import vtk
import vtk.util.numpy_support as nps
from vtk import VTK_VERSION
from vtk import vtkXMLRectilinearGridWriter
from vtk import vtkXMLUnstructuredGridWriter
from vtk import vtkXMLStructuredGridWriter
from vtk import vtkXMLRectilinearGridReader

import warnings


def assignCellData(vtkDS, models=None):
    """Assign the model(s) to the VTK dataset as CellData

    Input:

    :param vtki.Common vtkDS: - Any given VTK data object that has cell data
    :param dict(numpy.ndarray) models: Name('s) and array('s). Match number of cells

    """
    nc = vtkDS.GetNumberOfCells()
    if models is not None:
        for name, mod in models.items():
            # Convert numpy array
            if mod.size != nc:
                raise RuntimeError('Number of model cells ({}) does not match number of mesh cells ({}).'.format(mod.size, nc))
            vtkDoubleArr = nps.numpy_to_vtk(mod, deep=1)
            vtkDoubleArr.SetName(name)
            vtkDS.GetCellData().AddArray(vtkDoubleArr)
    return vtkDS



class vtkInterface(object):
    """This class is full of methods that enable ``discretize`` meshes to
    be converted to VTK data objects (and back when possible). This is
    inherritted by the :class:`discretize.BaseMesh.BaseMesh` class so all these
    methods are available to any mesh object!

    ``CurvilinearMesh``, ``TreeMesh``, and ``TensorMesh`` are all currently
    implemented. The ``CylMesh`` is not implemeted and will raise and excpetion.
    The following is an example of how to use the VTK interface to construct
    VTK data objects or write VTK files.

    .. code-block:: python
       :emphasize-lines: 8,11

       import discretize
       import numpy as np
       h1 = np.linspace(.1, .5, 3)
       h2 = np.linspace(.1, .5, 5)
       h3 = np.linspace(.1, .8, 3)
       mesh = discretize.TensorMesh([h1, h2, h3])

       # Get a VTK data object
       dataset = mesh.toVTK()

       # Save this mesh to a VTK file
       mesh.writeVTK('sample_mesh')

    Note that if your mesh is defined on a reference frame that is not the
    traditional <X,Y,Z> system with vectors of :math:`(1,0,0)`, :math:`(0,1,0)`,
    and :math:`(0,0,1)`, then the mesh will be rotated to be on the traditional
    reference frame. The previous example snippet provides a
    :class:`vtki.RectilinearGrid` object because that tensor mesh lies on the
    traditional reference frame. If we alter the reference frame, then we yield
    a :class:`vtki.StructuredGrid` that is the same mesh rotated in space.

    .. code-block:: python

       # Defined a rotated reference frame
       mesh.axis_u = (1,-1,0)
       mesh.axis_v = (-1,-1,0)
       mesh.axis_w = (0,0,1)
       # Check that the referenc fram is valid
       mesh._validate_orientation()

       # Yield the rotated vtkStructuredGrid
       dataset_r = mesh.toVTK()

       # or write it out to a VTK format
       mesh.writeVTK('sample_rotated')

    The two above code snippets produce a :class:`vtki.RectilinearGrid` and a
    :class:`vtki.StructuredGrid` respecitvely. To demonstarte the difference, we
    have plotted the two datasets next to eachother where the first mesh is in
    green and its data axes are parrallel to the traditional cartesian reference
    frame. The second, rotated mesh is shown in red and its data axii are
    rotated from the traditional cartesian refence frame as specified by the
    ``axis_u``, ``axis_v``, and ``axis_w`` properties.

    .. code-block:: python

        import vtki
        vtki.set_plot_theme('document')

        p = vtki.BackgroundPlotter()
        p.add_mesh(dataset, color='green', show_edges=True)
        p.add_mesh(dataset_r, color='maroon', show_edges=True)
        p.show_grid()
        p.screenshot('vtk-rotated-example.png')

    .. image:: ../images/vtk-rotated-example.png

    """

    def __treeMeshToVTK(mesh, models=None):
        """
        Constructs a :class:`vtki.UnstructuredGrid` object of this tree mesh and
        the given models as ``cell_arrays`` of that ``vtki`` dataset.

        Input:

        :param discretize.TreeMesh mesh: The tree mesh to convert to a :class:`vtki.UnstructuredGrid`
        :param dict(numpy.ndarray) models: Name('s) and array('s). Match number of cells

        """
        # Make the data parts for the vtu object
        # Points
        ptsMat = np.vstack((mesh.gridN, mesh.gridhN))

        # Adjust if result was 2D (voxels are pixels in 2D):
        VTK_CELL_TYPE = vtk.VTK_VOXEL
        if ptsMat.shape[1] == 2:
            # Add Z values of 0.0 if 2D
            ptsMat = np.c_[ptsMat, np.zeros(ptsMat.shape[0])]
            VTK_CELL_TYPE = vtk.VTK_PIXEL
        if ptsMat.shape[1] != 3:
            raise RuntimeError('Points of the mesh are improperly defined.')
        # Rotate the points to the cartesian system
        ptsMat = np.dot(ptsMat, mesh.rotation_matrix)
        # Grab the points
        vtkPts = vtk.vtkPoints()
        vtkPts.SetData(nps.numpy_to_vtk(ptsMat, deep=True))
        # Cells
        cellArray = [c for c in mesh]
        cellConn = np.array([cell.nodes for cell in cellArray])
        cellsMat = np.concatenate((np.ones((cellConn.shape[0], 1), dtype=int)*cellConn.shape[1], cellConn), axis=1).ravel()
        cellsArr = vtk.vtkCellArray()
        cellsArr.SetNumberOfCells(cellConn.shape[0])
        cellsArr.SetCells(cellConn.shape[0], nps.numpy_to_vtk(cellsMat, deep=True, array_type=vtk.VTK_ID_TYPE))
        # Make the object
        output = vtk.vtkUnstructuredGrid()
        output.SetPoints(vtkPts)
        output.SetCells(VTK_CELL_TYPE, cellsArr)
        # Add the level of refinement as a cell array
        cell_levels = np.array([cell._level for cell in cellArray])
        refineLevelArr = nps.numpy_to_vtk(cell_levels, deep=1)
        refineLevelArr.SetName('octreeLevel')
        output.GetCellData().AddArray(refineLevelArr)
        ubc_order = mesh._ubc_order
        # order_ubc will re-order from treemesh ordering to UBC ordering
        # need the opposite operation
        un_order = np.empty_like(ubc_order)
        un_order[ubc_order] = np.arange(len(ubc_order))
        order = nps.numpy_to_vtk(un_order)
        order.SetName('index_cell_corner')
        output.GetCellData().AddArray(order)
        # Assign the model('s) to the object
        return assignCellData(output, models=models)

    @staticmethod
    def __createStructGrid(ptsMat, dims, models=None):
        """An internal helper to build out structured grids"""
        # Adjust if result was 2D:
        if ptsMat.shape[1] == 2:
            # Figure out which dim is null
            nullDim = dims.index(None)
            ptsMat = np.insert(ptsMat, nullDim, np.zeros(ptsMat.shape[0]), axis=1)
        if ptsMat.shape[1] != 3:
            raise RuntimeError('Points of the mesh are improperly defined.')
        # Convert the points
        vtkPts = vtk.vtkPoints()
        vtkPts.SetData(nps.numpy_to_vtk(ptsMat, deep=True))
        # Uncover hidden dimension
        for i, d in enumerate(dims):
            if d is None:
                dims[i] = 0
            dims[i] = dims[i] + 1
        output = vtk.vtkStructuredGrid()
        output.SetDimensions(dims[0], dims[1], dims[2]) # note this subtracts 1
        output.SetPoints(vtkPts)
        # Assign the model('s) to the object
        return assignCellData(output, models=models)

    def __getRotatedNodes(mesh):
        """A helper to get the nodes of a mesh rotated by specified axes"""
        nodes = mesh.gridN
        if mesh.dim == 1:
            nodes = np.c_[mesh.gridN, np.zeros((mesh.nN, 2))]
        elif mesh.dim == 2:
            nodes = np.c_[mesh.gridN, np.zeros((mesh.nN, 1))]
        # Now garuntee nodes are correct
        if nodes.shape != (mesh.nN, 3):
            raise RuntimeError('Nodes of the grid are improperly defined.')
        # Rotate the points based on the axis orientations
        mesh._validate_orientation()
        return np.dot(nodes, mesh.rotation_matrix)

    def __tensorMeshToVTK(mesh, models=None):
        """
        Constructs a :class:`vtki.RectilinearGrid`
        (or a :class:`vtki.StructuredGrid`) object of this tensor mesh and the
        given models as ``cell_arrays`` of that grid.
        If the mesh is defined on a normal cartesian system then a rectilinear
        grid is generated. Otherwise, a structured grid is generated.

        Input:

        :param discretize.TensorMesh mesh: The tensor mesh to convert to a :class:`vtki.RectilinearGrid`
        :param dict(numpy.ndarray) models: Name('s) and array('s). Match number of cells

        """
        # Deal with dimensionalities
        if mesh.dim >= 1:
            vX = mesh.vectorNx
            xD = mesh.nNx
            yD, zD = 1, 1
            vY, vZ = np.array([0, 0])
        if mesh.dim >= 2:
            vY = mesh.vectorNy
            yD = mesh.nNy
        if mesh.dim == 3:
            vZ = mesh.vectorNz
            zD = mesh.nNz
        # If axis orientations are standard then use a vtkRectilinearGrid
        if not mesh.reference_is_rotated:
            # Use rectilinear VTK grid.
            # Assign the spatial information.
            output = vtk.vtkRectilinearGrid()
            output.SetDimensions(xD, yD, zD)
            output.SetXCoordinates(nps.numpy_to_vtk(vX, deep=1))
            output.SetYCoordinates(nps.numpy_to_vtk(vY, deep=1))
            output.SetZCoordinates(nps.numpy_to_vtk(vZ, deep=1))
            return assignCellData(output, models=models)
        # Use a structured grid where points are rotated to the cartesian system
        ptsMat = vtkInterface.__getRotatedNodes(mesh)
        dims = [mesh.nCx, mesh.nCy, mesh.nCz]
        # Assign the model('s) to the object
        return vtkInterface.__createStructGrid(ptsMat, dims, models=models)


    def __curvilinearMeshToVTK(mesh, models=None):
        """
        Constructs a :class:`vtki.StructuredGrid` of this mesh and the given
        models as ``cell_arrays`` of that object.

        Input:

        :param discretize.CurvilinearMesh mesh: The curvilinear mesh to convert to a :class:`vtki.StructuredGrid`
        :param dict(numpy.ndarray) models: Name('s) and array('s). Match number of cells

        """
        ptsMat = vtkInterface.__getRotatedNodes(mesh)
        dims = [mesh.nCx, mesh.nCy, mesh.nCz]
        return vtkInterface.__createStructGrid(ptsMat, dims, models=models)


    def __cylMeshToVTK(mesh, models=None):
        """This treats the CylindricalMesh defined in cylindrical coordinates
        :math:`(r, \theta, z)` and transforms it to cartesian coordinates.
        """
        # # Points
        # ptsMat = cyl2cart(mesh.gridN)
        # dims = [mesh.nCx, mesh.nCy, mesh.nCz]
        # return vtkInterface.__createStructGrid(ptsMat, dims, models=models)
        raise NotImplementedError('`CylMesh`s are not currently supported for VTK conversion.')


    def toVTK(mesh, models=None):
        """Convert this mesh object to it's proper ``vtki`` data object with
        the given model dictionary as the cell data of that dataset.

        Input:

        :param dict(numpy.ndarray) models: Name('s) and array('s). Match number of cells
        """
        # TODO: mesh.validate()
        converters = {
            'tree' : vtkInterface.__treeMeshToVTK,
            'tensor' : vtkInterface.__tensorMeshToVTK,
            'curv' : vtkInterface.__curvilinearMeshToVTK,
            #TODO: 'CylMesh' : vtkInterface.__cylMeshToVTK,
            }
        key = mesh._meshType.lower()
        try:
            convert = converters[key]
        except KeyError:
            raise RuntimeError('Mesh type `{}` is not currently supported for VTK conversion.'.format(key))
        # Convert the data object then attempt a wrapping with `vtki`
        cvtd = convert(mesh, models=models)
        try:
            import vtki
            cvtd = vtki.wrap(cvtd)
        except ImportError:
            warnings.warn('For easier use of VTK objects, you should install `vtki` (the VTK interface): pip install vtki')
        return cvtd

    @staticmethod
    def _saveUnstructuredGrid(fileName, vtkUnstructGrid, directory=''):
        """Saves a VTK unstructured grid file (vtu) for an already generated
        :class:`vtki.UnstructuredGrid` object.

        Input:

        :param str fileName: path to the output vtk file or just its name if directory is specified
        :param str directory: directory where the UBC GIF file lives
        """
        if not isinstance(vtkUnstructGrid, vtk.vtkUnstructuredGrid):
            raise RuntimeError('`_saveUnstructuredGrid` can only handle `vtkUnstructuredGrid` objects. `%s` is not supported.' % vtkUnstructGrid.__class__)
        # Check the extension of the fileName
        fname = os.path.join(directory, fileName)
        ext = os.path.splitext(fname)[1]
        if ext is '':
            fname = fname + '.vtu'
        elif ext not in '.vtu':
            raise IOError('{:s} is an incorrect extension, has to be .vtu'.format(ext))
        # Make the writer
        vtuWriteFilter = vtkXMLUnstructuredGridWriter()
        if float(VTK_VERSION.split('.')[0]) >= 6:
            vtuWriteFilter.SetInputDataObject(vtkUnstructGrid)
        else:
            vtuWriteFilter.SetInput(vtkUnstructGrid)
        vtuWriteFilter.SetFileName(fname)
        # Write the file
        vtuWriteFilter.Update()

    @staticmethod
    def _saveStructuredGrid(fileName, vtkStructGrid, directory=''):
        """Saves a VTK structured grid file (vtk) for an already generated
        :class:`vtki.StructuredGrid` object.

        Input:

        :param str fileName: path to the output vtk file or just its name if directory is specified
        :param str directory: directory where the UBC GIF file lives
        """
        if not isinstance(vtkStructGrid, vtk.vtkStructuredGrid):
            raise RuntimeError('`_saveStructuredGrid` can only handle `vtkStructuredGrid` objects. `{}` is not supported.'.format(vtkStructGrid.__class__))
        # Check the extension of the fileName
        fname = os.path.join(directory, fileName)
        ext = os.path.splitext(fname)[1]
        if ext is '':
            fname = fname + '.vts'
        elif ext not in '.vts':
            raise IOError('{:s} is an incorrect extension, has to be .vts'.format(ext))
        # Make the writer
        writer = vtkXMLStructuredGridWriter()
        if float(VTK_VERSION.split('.')[0]) >= 6:
            writer.SetInputDataObject(vtkStructGrid)
        else:
            writer.SetInput(vtkStructGrid)
        writer.SetFileName(fname)
        # Write the file
        writer.Update()

    @staticmethod
    def _saveRectilinearGrid(fileName, vtkRectGrid, directory=''):
        """Saves a VTK rectilinear file (vtr) ffor an already generated
        :class:`vtki.RectilinearGrid` object.

        Input:

        :param str fileName: path to the output vtk file or just its name if directory is specified
        :param str directory: directory where the UBC GIF file lives
        """
        if not isinstance(vtkRectGrid, vtk.vtkRectilinearGrid):
            raise RuntimeError('`_saveRectilinearGrid` can only handle `vtkRectilinearGrid` objects. `{}` is not supported.'.format(vtkRectGrid.__class__))
        # Check the extension of the fileName
        fname = os.path.join(directory, fileName)
        ext = os.path.splitext(fname)[1]
        if ext is '':
            fname = fname + '.vtr'
        elif ext not in '.vtr':
            raise IOError('{:s} is an incorrect extension, has to be .vtr'.format(ext))
        # Write the file.
        vtrWriteFilter = vtkXMLRectilinearGridWriter()
        if float(VTK_VERSION.split('.')[0]) >= 6:
            vtrWriteFilter.SetInputDataObject(vtkRectGrid)
        else:
            vtuWriteFilter.SetInput(vtuObj)
        vtrWriteFilter.SetFileName(fname)
        vtrWriteFilter.Update()

    def writeVTK(mesh, fileName, models=None, directory=''):
        """Makes and saves a VTK object from this mesh and given models

        Input:

        :param str fileName:  path to the output vtk file or just its name if directory is specified
        :param dict models: dictionary of numpy.array - Name('s) and array('s). Match number of cells
        :param str directory: directory where the UBC GIF file lives

        """
        vtkObj = vtkInterface.toVTK(mesh, models=models)
        writers = {
            'vtkUnstructuredGrid' : vtkInterface._saveUnstructuredGrid,
            'vtkRectilinearGrid' : vtkInterface._saveRectilinearGrid,
            'vtkStructuredGrid' : vtkInterface._saveStructuredGrid,
            }
        key = vtkObj.GetClassName()
        try:
            write = writers[key]
        except:
            raise RuntimeError('VTK data type `%s` is not currently supported.' % key)
        return write(fileName, vtkObj, directory=directory)


class vtkTensorRead(object):
    """Provides a convienance method for reading VTK Rectilinear Grid files
    as ``TensorMesh`` objects."""


    @classmethod
    def vtkToTensorMesh(TensorMesh, vtrGrid):
        """Converts a ``vtkRectilinearGrid`` or :class:`vtki.RectilinearGrid`
        to a :class:`discretize.TensorMesh` object.

        Output:

        :rtype: tuple
        :return: (TensorMesh, modelDictionary)
        """
        # Sort information
        hx = np.abs(np.diff(nps.vtk_to_numpy(vtrGrid.GetXCoordinates())))
        xR = nps.vtk_to_numpy(vtrGrid.GetXCoordinates())[0]
        hy = np.abs(np.diff(nps.vtk_to_numpy(vtrGrid.GetYCoordinates())))
        yR = nps.vtk_to_numpy(vtrGrid.GetYCoordinates())[0]
        zD = np.diff(nps.vtk_to_numpy(vtrGrid.GetZCoordinates()))
        # Check the direction of hz
        if np.all(zD < 0):
            hz = np.abs(zD[::-1])
            zR = nps.vtk_to_numpy(vtrGrid.GetZCoordinates())[-1]
        else:
            hz = np.abs(zD)
            zR = nps.vtk_to_numpy(vtrGrid.GetZCoordinates())[0]
        x0 = np.array([xR, yR, zR])

        # Make the object
        tensMsh = TensorMesh([hx, hy, hz], x0=x0)

        # Grap the models
        models = {}
        for i in np.arange(vtrGrid.GetCellData().GetNumberOfArrays()):
            modelName = vtrGrid.GetCellData().GetArrayName(i)
            if np.all(zD < 0):
                modFlip = nps.vtk_to_numpy(vtrGrid.GetCellData().GetArray(i))
                tM = tensMsh.r(modFlip, 'CC', 'CC', 'M')
                modArr = tensMsh.r(tM[:, :, ::-1], 'CC', 'CC', 'V')
            else:
                modArr = nps.vtk_to_numpy(vtrGrid.GetCellData().GetArray(i))
            models[modelName] = modArr

        # Return the data
        return tensMsh, models

    @classmethod
    def readVTK(TensorMesh, fileName, directory=''):
        """Read VTK Rectilinear (vtr xml file) and return Tensor mesh and model

        Input:

        :param str fileName: path to the vtr model file to read or just its name if directory is specified
        :param str directory: directory where the UBC GIF file lives

        Output:

        :rtype: tuple
        :return: (TensorMesh, modelDictionary)
        """
        fname = os.path.join(directory, fileName)
        # Read the file
        vtrReader = vtkXMLRectilinearGridReader()
        vtrReader.SetFileName(fname)
        vtrReader.Update()
        vtrGrid = vtrReader.GetOutput()
        return TensorMesh.vtkToTensorMesh(vtrGrid)
