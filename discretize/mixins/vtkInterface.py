"""
The ``vtkInterface`` module provides an way for ``discretize`` meshes to be
converted to VTK data objects (and back when possible).

"""
import os
import numpy as np
import six

import vtk
import vtk.util.numpy_support as nps
from vtk import VTK_VERSION
from vtk import vtkXMLRectilinearGridWriter
from vtk import vtkXMLUnstructuredGridWriter
from vtk import vtkStructuredGridWriter

# TODO: WHY CANT I DO LOCAL IMPORTS?
# from ..TensorMesh import TensorMesh
# from ..TreeMesh import TreeMesh
# from ..CurvilinearMesh import CurvilinearMesh
# from ..CylMesh import CylMesh


class vtkInterface(object):
    """This class is full of static methods that enable ``discretize`` meshes to
    be converted to VTK data objects (and back when possible).
    """


    @staticmethod
    def _treeMeshToVTK(mesh, models=None):
        """
        Constructs a ``vtkUnstructuredGrid`` object of this tree mesh and the
        given models as ``CellData`` of that VTK dataset.

        Input:
        :param mesh, discretize.TreeMesh - The tree mesh to convert to a ``vtkUnstructuredGrid``
        :param models, dictionary of numpy.array - Name('s) and array('s). Match number of cells

        """
        # Make the data parts for the vtu object
        # Points
        ptsMat = np.vstack((mesh.gridN, mesh.gridhN))

        # Adjust if result was 2D (voxels are pixels in 2D):
        VTK_CELL_TYPE = vtk.VTK_VOXEL
        if ptsMat.shape[1] == 2:
            # TODO: assumes any 2D tree meshes are on the XY Plane
            # Add Z values of 0.0 if 2D
            ptsMat = np.c_[ptsMat, np.zeros(ptsMat.shape[0])]
            VTK_CELL_TYPE = vtk.VTK_PIXEL
        if ptsMat.shape[1] != 3:
            raise RuntimeError('Points of the mesh are improperly defined.')

        vtkPts = vtk.vtkPoints()
        vtkPts.SetData(nps.numpy_to_vtk(ptsMat, deep=True))

        # Cells
        cellArray = [c for c in mesh]
        cellConn = np.array([cell.nodes for cell in cellArray])

        cellsMat = np.concatenate((np.ones((cellConn.shape[0], 1), dtype=np.int64)*cellConn.shape[1], cellConn), axis=1).ravel()
        cellsArr = vtk.vtkCellArray()
        cellsArr.SetNumberOfCells(cellConn.shape[0])
        cellsArr.SetCells(cellConn.shape[0], nps.numpy_to_vtkIdTypeArray(cellsMat, deep=True))

        # Make the object
        vtuObj = vtk.vtkUnstructuredGrid()
        vtuObj.SetPoints(vtkPts)
        vtuObj.SetCells(VTK_CELL_TYPE, cellsArr)
        # Add the level of refinement as a cell array
        cell_levels = np.array([cell._level for cell in cellArray])
        refineLevelArr = nps.numpy_to_vtk(cell_levels, deep=1)
        refineLevelArr.SetName('octreeLevel')
        vtuObj.GetCellData().AddArray(refineLevelArr)
        # Assign the model('s) to the object
        if models is not None:
            for item in six.iteritems(models):
                # Convert numpy array
                vtkDoubleArr = nps.numpy_to_vtk(item[1], deep=1)
                vtkDoubleArr.SetName(item[0])
                vtuObj.GetCellData().AddArray(vtkDoubleArr)

        return vtuObj

    @staticmethod
    def _tensorMeshToVTK(mesh, models=None):
        """
        Constructs a ``vtkRectilinearGrid`` object of this tensor mesh and the
        given models as ``CellData`` of that grid.

        Input:
        :param mesh, discretize.TensorMesh - The tensor mesh to convert to a ``vtkRectilinearGrid``
        :param models, dictionary of numpy.array - Name('s) and array('s). Match number of cells

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
        # Use rectilinear VTK grid.
        # Assign the spatial information.
        vtkObj = vtk.vtkRectilinearGrid()
        vtkObj.SetDimensions(xD, yD, zD)
        vtkObj.SetXCoordinates(nps.numpy_to_vtk(vX, deep=1))
        vtkObj.SetYCoordinates(nps.numpy_to_vtk(vY, deep=1))
        vtkObj.SetZCoordinates(nps.numpy_to_vtk(vZ, deep=1))

        # Assign the model('s) to the object
        if models is not None:
            for item in models.items():
                # Convert numpy array
                vtkDoubleArr = nps.numpy_to_vtk(item[1], deep=1)
                vtkDoubleArr.SetName(item[0])
                vtkObj.GetCellData().AddArray(vtkDoubleArr)
            # Set the active scalar
            vtkObj.GetCellData().SetActiveScalars(list(models.keys())[0])
        return vtkObj


    @staticmethod
    def _curvilinearMeshToVTK(mesh, models=None):
        """
        Constructs a ``vtkStructuredGrid`` of this mesh and the given
        models as ``CellData`` of that object.

        Input:
        :param mesh, discretize.CurvilinearMesh - The curvilinear mesh to convert to a ``vtkStructuredGrid``
        :param models, dictionary of numpy.array - Name('s) and array('s). Match number of cells

        """
        # Make the data parts for the vtu object
        # Points
        ptsMat = mesh.gridN

        dims = [mesh.nCx, mesh.nCy, mesh.nCz]
        # Adjust if result was 2D:
        if ptsMat.shape[1] == 2:
            # Figure out which dim is null
            nullDim = dims.index(None)
            ptsMat = np.insert(ptsMat, nullDim, np.zeros(ptsMat.shape[0]), axis=1)
        if ptsMat.shape[1] != 3:
            raise RuntimeError('Points of the mesh are improperly defined.')

        vtkPts = vtk.vtkPoints()
        vtkPts.SetData(nps.numpy_to_vtk(ptsMat, deep=True))

        dims = [mesh.nCx, mesh.nCy, mesh.nCz]
        for i, d in enumerate(dims):
            if d is None:
                dims[i] = 0
            dims[i] = dims[i] + 1

        output = vtk.vtkStructuredGrid()
        output.SetDimensions(dims[0], dims[1], dims[2]) # note this subtracts 1
        output.SetPoints(vtkPts)

        # Assign the model('s) to the object
        if models is not None:
            for item in six.iteritems(models):
                # Convert numpy array
                vtkDoubleArr = nps.numpy_to_vtk(item[1], deep=1)
                vtkDoubleArr.SetName(item[0])
                output.GetCellData().AddArray(vtkDoubleArr)

        return output

    @staticmethod
    def _cylMeshToVTK(mesh, models=None):
        # TODO: implement this!
        pass

    @staticmethod
    def toVTK(mesh, models=None):
        """Convert any mesh object to it's proper VTK data object."""
        converters = {
            'TreeMesh' : vtkInterface._treeMeshToVTK,
            'TensorMesh' : vtkInterface._tensorMeshToVTK,
            'CurvilinearMesh' : vtkInterface._curvilinearMeshToVTK,
            #TODO: 'CylMesh' : vtkInterface._cylMeshToVTK,
            }
        key = type(mesh).__name__
        try:
            convert = converters[key]
        except:
            raise RuntimeError('Mesh type `%s` is not currently supported for VTK conversion.' % key)
        return convert(mesh, models=models)

    @staticmethod
    def _saveUnstructuredGrid(fileName, vtkUnstructGrid, directory=''):
        """Saves a VTK unstructured grid file (vtu) for an already generated
        ``vtkUnstructuredGrid`` object.

        Input:
        :param str fileName:  path to the output vtk file or just its name if directory is specified
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
            raise IOError('{:s} is an incorrect extension, has to be .vtu')
        # Make the writer
        vtuWriteFilter = vtkXMLUnstructuredGridWriter()
        if float(VTK_VERSION.split('.')[0]) >= 6:
            vtuWriteFilter.SetInputDataObject(vtkUnstructGrid)
        else:
            vtuWriteFilter.SetInput(vtkUnstructGrid)
        vtuWriteFilter.SetFileName(fileName)
        # Write the file
        vtuWriteFilter.Update()

    @staticmethod
    def _saveStructuredGrid(fileName, vtkStructGrid, directory=''):
        """Saves a VTK structured grid file (vtk) for an already generated
        ``vtkStructuredGrid`` object.

        Input:
        :param str fileName:  path to the output vtk file or just its name if directory is specified
        :param str directory: directory where the UBC GIF file lives
        """
        if not isinstance(vtkStructGrid, vtk.vtkStructuredGrid):
            raise RuntimeError('`_saveStructuredGrid` can only handle `vtkStructuredGrid` objects. `%s` is not supported.' % vtkStructGrid.__class__)
        # Check the extension of the fileName
        fname = os.path.join(directory, fileName)
        ext = os.path.splitext(fname)[1]
        if ext is '':
            fname = fname + '.vtk'
        elif ext not in '.vtk':
            raise IOError('{:s} is an incorrect extension, has to be .vtk')
        # Make the writer
        writer = vtkStructuredGridWriter()
        if float(VTK_VERSION.split('.')[0]) >= 6:
            writer.SetInputDataObject(vtkStructGrid)
        else:
            writer.SetInput(vtkStructGrid)
        writer.SetFileName(fileName)
        # Write the file
        writer.Update()

    @staticmethod
    def _saveRectilinearGrid(fileName, vtkRectGrid, directory=''):
        """Saves a VTK rectilinear file (vtr) ffor an already generated
        ``vtkRectilinearGrid`` object.

        Input:
        :param str fileName:  path to the output vtk file or just its name if directory is specified
        :param str directory: directory where the UBC GIF file lives
        """
        if not isinstance(vtkRectGrid, vtk.vtkRectilinearGrid):
            raise RuntimeError('`_saveRectilinearGrid` can only handle `vtkRectilinearGrid` objects. `%s` is not supported.' % vtkRectGrid.__class__)
        # Check the extension of the fileName
        fname = os.path.join(directory, fileName)
        ext = os.path.splitext(fname)[1]
        if ext is '':
            fname = fname + '.vtr'
        elif ext not in '.vtr':
            raise IOError('{:s} is an incorrect extension, has to be .vtr')
        # Write the file.
        vtrWriteFilter = vtkXMLRectilinearGridWriter()
        if float(VTK_VERSION.split('.')[0]) >= 6:
            vtrWriteFilter.SetInputDataObject(vtkRectGrid)
        else:
            vtuWriteFilter.SetInput(vtuObj)
        vtrWriteFilter.SetFileName(fname)
        vtrWriteFilter.Update()

    @staticmethod
    def saveVTK(fileName, mesh, models=None, directory=''):
        """Save any mesh object to its corresponding VTK data format."""
        vtkObj = vtkInterface.toVTK(mesh, models=models)
        writers = {
            'vtkUnstructuredGrid' : vtkInterface._saveUnstructuredGrid,
            'vtkRectilinearGrid' : vtkInterface._saveRectilinearGrid,
            'vtkStructuredGrid' : vtkInterface._saveStructuredGrid,
            #TODO: 'CylMesh' : vtkInterface.???,
            }
        key = type(vtkObj).__name__
        try:
            write = writers[key]
        except:
            raise RuntimeError('VTK data type `%s` is not currently supported.' % key)
        return write(fileName, vtkObj, directory=directory)
