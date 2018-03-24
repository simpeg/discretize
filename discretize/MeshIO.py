import os
import json
import properties
import numpy as np
import six

from . import utils
from .BaseMesh import BaseMesh


def load_mesh(filename):
    """
    Open a json file and load the mesh into the target class

    As long as there are no namespace conflicts, the target __class__
    will be stored on the properties.HasProperties registry and may be
    fetched from there.

    :param str filename: name of file to read in
    """
    with open(filename, 'r') as outfile:
        jsondict = json.load(outfile)
        data = BaseMesh.deserialize(jsondict, trusted=True)
    return data


class TensorMeshIO(object):

    @classmethod
    def _readUBC_3DMesh(TensorMesh, fileName):
        """Read UBC GIF 3D tensor mesh and generate same dimension TensorMesh.

        Input:
        :param string fileName: path to the UBC GIF mesh file

        Output:
        :rtype: TensorMesh
        :return: The tensor mesh for the fileName.
        """

        # Interal function to read cell size lines for the UBC mesh files.
        def readCellLine(line):
            line_list = []
            for seg in line.split():
                if '*' in seg:
                    sp = seg.split('*')
                    seg_arr = np.ones((int(sp[0]),)) * float(sp[1])
                else:
                    seg_arr = np.array([float(seg)], float)
                line_list.append(seg_arr)
            return np.concatenate(line_list)

        # Read the file as line strings, remove lines with comment = !
        msh = np.genfromtxt(fileName, delimiter='\n', dtype=np.str, comments='!')
        # Fist line is the size of the model
        sizeM = np.array(msh[0].split(), dtype=float)
        # Second line is the South-West-Top corner coordinates.
        x0 = np.array(msh[1].split(), dtype=float)
        # Read the cell sizes
        h1 = readCellLine(msh[2])
        h2 = readCellLine(msh[3])
        h3temp = readCellLine(msh[4])
        # Invert the indexing of the vector to start from the bottom.
        h3 = h3temp[::-1]
        # Adjust the reference point to the bottom south west corner
        x0[2] = x0[2] - np.sum(h3)
        # Make the mesh
        tensMsh = TensorMesh([h1, h2, h3], x0=x0)
        return tensMsh

    @classmethod
    def _readUBC_2DMesh(TensorMesh, fileName):
        """Read UBC GIF 2DTensor mesh and generate 2D Tensor mesh in simpeg

        Input:
        :param string fileName: path to the UBC GIF mesh file

        Output:
        :rtype: TensorMesh
        :return: SimPEG TensorMesh 2D object
        """

        fopen = open(fileName, 'r')

        # Read down the file and unpack dx vector
        def unpackdx(fid, nrows):
            for ii in range(nrows):
                line = fid.readline()
                var = np.array(line.split(), dtype=float)
                if ii == 0:
                    x0 = var[0]
                    xvec = np.ones(int(var[2])) * (var[1] - var[0]) / int(var[2])
                    xend = var[1]
                else:
                    xvec = np.hstack((xvec, np.ones(int(var[1])) * (var[0] - xend) / int(var[1])))
                    xend = var[0]
            return x0, xvec

        # Start with dx block
        # First line specifies the number of rows for x-cells
        line = fopen.readline()
        # Strip comments lines
        while line.startswith("!"):
            line = fopen.readline()
        nl = np.array(line.split(), dtype=int)
        [x0, dx] = unpackdx(fopen, nl[0])
        # Move down the file until reaching the z-block
        line = fopen.readline()
        if not line:
            line = fopen.readline()
        # End with dz block
        # First line specifies the number of rows for z-cells
        line = fopen.readline()
        nl = np.array(line.split(), dtype=int)
        [z0, dz] = unpackdx(fopen, nl[0])
        # Flip z0 to be the bottom of the mesh for SimPEG
        z0 = z0 - sum(dz)
        dz = dz[::-1]
        # Make the mesh
        tensMsh = TensorMesh([dx, dz], x0=(x0, z0))

        fopen.close()

        return tensMsh

    @classmethod
    def readUBC(TensorMesh, fileName, directory=''):
        """Wrapper to Read UBC GIF 2D  and 3D tensor mesh and generate same dimension TensorMesh.

        Input:
        :param str fileName: path to the UBC GIF mesh file or just its name if directory is specified
        :param str directory: directory where the UBC GIF file lives

        Output:
        :rtype: TensorMesh
        :return: The tensor mesh for the fileName.
        """
        # Check the expected mesh dimensions
        fname = os.path.join(directory, fileName)
        # Read the file as line strings, remove lines with comment = !
        msh = np.genfromtxt(
            fname, delimiter='\n', dtype=np.str,
            comments='!', max_rows=1
        )
        # Fist line is the size of the model
        sizeM = np.array(msh.ravel()[0].split(), dtype=float)
        # Check if the mesh is a UBC 2D mesh
        if sizeM.shape[0] == 1:
            Tnsmsh = TensorMesh._readUBC_2DMesh(fname)
        # Check if the mesh is a UBC 3D mesh
        elif sizeM.shape[0] == 3:
            Tnsmsh = TensorMesh._readUBC_3DMesh(fname)
        else:
            raise Exception('File format not recognized')
        return Tnsmsh

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
        from vtk import vtkXMLRectilinearGridReader as vtrFileReader
        from vtk.util.numpy_support import vtk_to_numpy

        fname = os.path.join(directory, fileName)
        # Read the file
        vtrReader = vtrFileReader()
        vtrReader.SetFileName(fname)
        vtrReader.Update()
        vtrGrid = vtrReader.GetOutput()
        # Sort information
        hx = np.abs(np.diff(vtk_to_numpy(vtrGrid.GetXCoordinates())))
        xR = vtk_to_numpy(vtrGrid.GetXCoordinates())[0]
        hy = np.abs(np.diff(vtk_to_numpy(vtrGrid.GetYCoordinates())))
        yR = vtk_to_numpy(vtrGrid.GetYCoordinates())[0]
        zD = np.diff(vtk_to_numpy(vtrGrid.GetZCoordinates()))
        # Check the direction of hz
        if np.all(zD < 0):
            hz = np.abs(zD[::-1])
            zR = vtk_to_numpy(vtrGrid.GetZCoordinates())[-1]
        else:
            hz = np.abs(zD)
            zR = vtk_to_numpy(vtrGrid.GetZCoordinates())[0]
        x0 = np.array([xR, yR, zR])

        # Make the object
        tensMsh = TensorMesh([hx, hy, hz], x0=x0)

        # Grap the models
        models = {}
        for i in np.arange(vtrGrid.GetCellData().GetNumberOfArrays()):
            modelName = vtrGrid.GetCellData().GetArrayName(i)
            if np.all(zD < 0):
                modFlip = vtk_to_numpy(vtrGrid.GetCellData().GetArray(i))
                tM = tensMsh.r(modFlip, 'CC', 'CC', 'M')
                modArr = tensMsh.r(tM[:, :, ::-1], 'CC', 'CC', 'V')
            else:
                modArr = vtk_to_numpy(vtrGrid.GetCellData().GetArray(i))
            models[modelName] = modArr

        # Return the data
        return tensMsh, models

    def writeVTK(mesh, fileName, models=None, directory=''):
        """Makes and saves a VTK rectilinear file (vtr)
        for a Tensor mesh and model.

        Input:
        :param str fileName:  path to the output vtk file or just its name if directory is specified
        :param str directory: directory where the UBC GIF file lives
        :param dict models: dictionary of numpy.array - Name('s) and array('s).
        Match number of cells
        """
        from vtk import vtkRectilinearGrid as rectGrid, vtkXMLRectilinearGridWriter as rectWriter, VTK_VERSION
        from vtk.util.numpy_support import numpy_to_vtk

        fname = os.path.join(directory, fileName)
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
        vtkObj = rectGrid()
        vtkObj.SetDimensions(xD, yD, zD)
        vtkObj.SetXCoordinates(numpy_to_vtk(vX, deep=1))
        vtkObj.SetYCoordinates(numpy_to_vtk(vY, deep=1))
        vtkObj.SetZCoordinates(numpy_to_vtk(vZ, deep=1))

        # Assign the model('s) to the object
        if models is not None:
            for key in models:
                # Convert numpy array
                vtkDoubleArr = numpy_to_vtk(models[key], deep=1)
                vtkDoubleArr.SetName(key)
                vtkObj.GetCellData().AddArray(vtkDoubleArr)
            # Set the active scalar
            vtkObj.GetCellData().SetActiveScalars(key)

        # Check the extension of the fileName
        ext = os.path.splitext(fname)[1]
        if ext is '':
            fname = fname + '.vtr'
        elif ext not in '.vtr':
            raise IOError('{:s} is an incorrect extension, has to be .vtr')
        # Write the file.
        vtrWriteFilter = rectWriter()
        if float(VTK_VERSION.split('.')[0]) >= 6:
            vtrWriteFilter.SetInputData(vtkObj)
        else:
            vtuWriteFilter.SetInput(vtuObj)
        vtrWriteFilter.SetFileName(fname)
        vtrWriteFilter.Update()

    def _toVTRObj(mesh, models=None):
        """
        Makes and saves a VTK rectilinear file (vtr) for a
        Tensor mesh and model.

        Input:
        :param str, path to the output vtk file
        :param mesh, TensorMesh object - mesh to be transfer to VTK
        :param models, dictionary of numpy.array - Name('s) and array('s). Match number of cells

        """
        # Import
        from vtk import vtkRectilinearGrid as rectGrid, VTK_VERSION
        from vtk.util.numpy_support import numpy_to_vtk

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
        vtkObj = rectGrid()
        vtkObj.SetDimensions(xD, yD, zD)
        vtkObj.SetXCoordinates(numpy_to_vtk(vX, deep=1))
        vtkObj.SetYCoordinates(numpy_to_vtk(vY, deep=1))
        vtkObj.SetZCoordinates(numpy_to_vtk(vZ, deep=1))

        # Assign the model('s) to the object
        if models is not None:
            for item in models.iteritems():
                # Convert numpy array
                vtkDoubleArr = numpy_to_vtk(item[1], deep=1)
                vtkDoubleArr.SetName(item[0])
                vtkObj.GetCellData().AddArray(vtkDoubleArr)
            # Set the active scalar
            vtkObj.GetCellData().SetActiveScalars(list(models.keys())[0])
        return vtkObj

    def _readModelUBC_2D(mesh, fileName):
        """
        Read UBC GIF 2DTensor model and generate 2D Tensor model in simpeg

        Input:
        :param string fileName: path to the UBC GIF 2D model file

        Output:
        :rtype: numpy.ndarray
        :return: model with TensorMesh ordered
        """

        # Open fileand skip header... assume that we know the mesh already
        obsfile = np.genfromtxt(
            fileName, delimiter=' \n',
            dtype=np.str, comments='!'
        )

        dim = np.array(obsfile[0].split(), dtype=int)
        if not np.all([mesh.nCx, mesh.nCy] == dim):
            raise Exception('Dimension of the model and mesh mismatch')

        # Make a list of the lines
        model = [line.split() for line in obsfile[1:]]
        # Address the case where lines are not split equally
        model = [cellvalue for sublist in model[::-1] for cellvalue in sublist]
        # Make the vector
        model = utils.mkvc(np.array(model, dtype=float).T)
        if not len(model) == mesh.nC:
            raise Exception(
                """Something is not right, expected size is {:d}
                but unwrap vector is size {:d}""".format(mesh.nC, len(model))
            )

        return model

    def _readModelUBC_3D(mesh, fileName):
        """Read UBC 3DTensor mesh model and generate 3D Tensor mesh model

        Input:
        :param string fileName: path to the UBC GIF mesh file to read

        Output:
        :rtype: numpy.ndarray
        :return: model with TensorMesh ordered
        """
        f = open(fileName, 'r')
        model = np.array(list(map(float, f.readlines())))
        f.close()
        model = np.reshape(model, (mesh.nCz, mesh.nCx, mesh.nCy), order='F')
        model = model[::-1, :, :]
        model = np.transpose(model, (1, 2, 0))
        model = utils.mkvc(model)
        return model

    def readModelUBC(mesh, fileName, directory=''):
        """Read UBC 2D or 3D Tensor mesh model
            and generate Tensor mesh model

        Input:
        :param str fileName:  path to the UBC GIF mesh file to read
        or just its name if directory is specified
        :param str directory: directory where the UBC GIF file lives

        Output:
        :rtype: numpy.ndarray
        :return: model with TensorMesh ordered
        """
        fname = os.path.join(directory, fileName)
        if mesh.dim == 3:
            model = mesh._readModelUBC_3D(fname)
        elif mesh.dim == 2:
            model = mesh._readModelUBC_2D(fname)
        else:
            raise Exception('mesh must be a Tensor Mesh 2D or 3D')
        return model

    def writeModelUBC(mesh, fileName, model, directory=''):
        """Writes a model associated with a TensorMesh
        to a UBC-GIF format model file.

        Input:
        :param str fileName:  File to write to
        or just its name if directory is specified
        :param str directory: directory where the UBC GIF file lives
        :param numpy.ndarray model: The model
        """
        fname = os.path.join(directory, fileName)
        if mesh.dim == 3:
            # Reshape model to a matrix
            modelMat = mesh.r(model, 'CC', 'CC', 'M')
            # Transpose the axes
            modelMatT = modelMat.transpose((2, 0, 1))
            # Flip z to positive down
            modelMatTR = utils.mkvc(modelMatT[::-1, :, :])
            np.savetxt(fname, modelMatTR.ravel())

        elif mesh.dim == 2:
            modelMat = mesh.r(model, 'CC', 'CC', 'M').T[::-1]
            f = open(fname, 'w')
            f.write('{:d} {:d}\n'.format(mesh.nCx, mesh.nCy))
            f.close()
            f = open(fname, 'ab')
            np.savetxt(f, modelMat)
            f.close()

        else:
            raise Exception('mesh must be a Tensor Mesh 2D or 3D')

    def _writeUBC_3DMesh(mesh, fileName, comment_lines=''):
        """Writes a TensorMesh to a UBC-GIF format mesh file.

        Input:
        :param string fileName: File to write to
        :param dict models: A dictionary of the models
        """
        if not mesh.dim == 3:
            raise Exception('Mesh must be 3D')

        s = comment_lines
        s += '{0:d} {1:d} {2:d}\n'.format(*tuple(mesh.vnC))
        # Have to it in the same operation or use mesh.x0.copy(),
        # otherwise the mesh.x0 is updated.
        origin = mesh.x0 + np.array([0, 0, mesh.hz.sum()])
        origin.dtype = float

        s += '{0:.6f} {1:.6f} {2:.6f}\n'.format(*tuple(origin))
        s += ('%.6f '*mesh.nCx+'\n') % tuple(mesh.hx)
        s += ('%.6f '*mesh.nCy+'\n') % tuple(mesh.hy)
        s += ('%.6f '*mesh.nCz+'\n') % tuple(mesh.hz[::-1])
        f = open(fileName, 'w')
        f.write(s)
        f.close()

    def _writeUBC_2DMesh(mesh, fileName, comment_lines=''):
        """Writes a TensorMesh to a UBC-GIF format mesh file.

        Input:
        :param string fileName: File to write to
        :param dict models: A dictionary of the models

        """
        if not mesh.dim == 2:
            raise Exception('Mesh must be 2D')

        def writeF(fx, outStr=''):
            # Init
            i = 0
            origin = True
            x0 = fx[i]
            f = fx[i]
            number_segment = 0
            auxStr = ''

            while True:
                i = i + 1
                if i >= fx.size:
                    break
                dx = -f + fx[i]
                f = fx[i]
                n = 1

                for j in range(i+1, fx.size):
                    if -f + fx[j] == dx:
                        n += 1
                        i += 1
                        f = fx[j]
                    else:
                        break

                number_segment += 1
                if origin:
                    auxStr += '{:.10f} {:.10f} {:d} \n'.format(x0, f, n)
                    origin = False
                else:
                    auxStr += '{:.10f} {:d} \n'.format(f, n)

            auxStr = '{:d}\n'.format(number_segment) + auxStr
            outStr += auxStr

            return outStr

        # Grab face coordinates
        fx = mesh.vectorNx
        fz = -mesh.vectorNy[::-1]

        # Create the string
        outStr = comment_lines
        outStr = writeF(fx, outStr=outStr)
        outStr += '\n'
        outStr = writeF(fz, outStr=outStr)

        # Write file
        f = open(fileName, 'w')
        f.write(outStr)
        f.close()

    def writeUBC(mesh, fileName, models=None, directory='', comment_lines=''):
        """Writes a TensorMesh to a UBC-GIF format mesh file.

        Input:
        :param str fileName: File to write to
        :param str directory: directory where to save model
        :param dict models: A dictionary of the models
        :param str comment_lines: comment lines preceded with '!' to add
        """
        fname = os.path.join(directory, fileName)
        if mesh.dim == 3:
            mesh._writeUBC_3DMesh(fname, comment_lines=comment_lines)
        elif mesh.dim == 2:
            mesh._writeUBC_2DMesh(fname, comment_lines=comment_lines)
        else:
            raise Exception('mesh must be a Tensor Mesh 2D or 3D')

        if models is None:
            return
        assert type(models) is dict, 'models must be a dict'
        for key in models:
            assert type(key) is str, 'The dict key is a file name'
            mesh.writeModelUBC(key, models[key], directory=directory)


class TreeMeshIO(object):

    def writeUBC(mesh, fileName, models=None):
        """Write UBC ocTree mesh and model files from a
        octree mesh and model.

        :param string fileName: File to write to
        :param dict models: Models in a dict, where each key is the filename
        """

        # Calculate information to write in the file.
        # Number of cells in the underlying mesh
        nCunderMesh = np.array([h.size for h in mesh.h], dtype=np.int64)
        # The top-south-west most corner of the mesh
        tswCorn = mesh.x0 + np.array([0, 0, np.sum(mesh.h[2])])
        # Smallest cell size
        smallCell = np.array([h.min() for h in mesh.h])
        # Number of cells
        nrCells = mesh.nC

        # Extract information about the cells.
        cellPointers = np.array([c._pointer for c in mesh])
        cellW = np.array([mesh._levelWidth(i) for i in cellPointers[:, -1]])
        # Need to shift the pointers to work with UBC indexing
        # UBC Octree indexes always the top-left-close (top-south-west) corner
        # first and orders the cells in z(top-down), x, y vs x, y, z(bottom-up)
        # Shift index up by 1
        ubcCellPt = cellPointers[:, 0:-1].copy() + np.array([1., 1., 1.])
        # Need re-index the z index to be from the top-left-close corner and to
        # be from the global top.
        ubcCellPt[:, 2] = (nCunderMesh[-1] + 2) - (ubcCellPt[:, 2] + cellW)

        # Reorder the ubcCellPt
        ubcReorder = np.argsort(
            ubcCellPt.view(', '.join(3*['float'])),
            axis=0,
            order=['f2', 'f1', 'f0']
        )[:, 0]
        # Make a array with the pointers and the withs,
        # that are order in the ubc ordering
        indArr = np.concatenate(
            (ubcCellPt[ubcReorder, :], cellW[ubcReorder].reshape((-1, 1))),
            axis=1
        )
        # Write the UBC octree mesh file
        head = (
            '{:.0f} {:.0f} {:.0f}\n'.format(
                nCunderMesh[0], nCunderMesh[1], nCunderMesh[2]
            ) +
            '{:.4f} {:.4f} {:.4f}\n'.format(
                tswCorn[0], tswCorn[1], tswCorn[2]
            ) +
            '{:.3f} {:.3f} {:.3f}\n'.format(
                smallCell[0], smallCell[1], smallCell[2]
            ) +
            '{:.0f}'.format(nrCells)
        )
        np.savetxt(fileName, indArr, fmt='%i', header=head, comments='')

        # Print the models
        # Assign the model('s) to the object
        if models is not None:
            for item in six.iteritems(models):
                # Save the data
                np.savetxt(item[0], item[1][ubcReorder], fmt='%3.5e')

    @classmethod
    def readUBC(TreeMesh, meshFile):
        """Read UBC 3D OcTree mesh and/or modelFiles

        Input:
        :param str meshFile: path to the UBC GIF OcTree mesh file to read
        :rtype: discretize.TreeMesh
        :return: The octree mesh
        """

        # Read the file lines
        fileLines = np.genfromtxt(meshFile, dtype=str,
            delimiter='\n', comments='!')
        # Extract the data
        nCunderMesh = np.array(fileLines[0].
            split('!')[0].split(), dtype=int)
        # I think this is the case?
        # Format of file changed... First 3 values are the # of cells in the
        # underlying mesh and remaining 6 values are padding for the core region.
        nCunderMesh  = nCunderMesh[0:3]

        if np.unique(nCunderMesh).size >1:
            raise Exception('TreeMeshes have the same number of cell in all directions')
        tswCorn = np.array(
            fileLines[1].split('!')[0].split(),
            dtype=float
        )
        smallCell = np.array(
            fileLines[2].split('!')[0].split(),
            dtype=float
        )
        nrCells = np.array(
            fileLines[3].split('!')[0].split(),
            dtype=float
        )
        # Read the index array
        indArr = np.genfromtxt((line.encode('utf8') for line in fileLines[4::]), dtype=np.int)

        # Calculate simpeg parameters
        h1, h2, h3 = [np.ones(nr)*sz for nr, sz in zip(nCunderMesh, smallCell)]
        x0 = tswCorn - np.array([0, 0, np.sum(h3)])
        # Convert the index array to a points list that complies with TreeMesh
        # Shift to start at 0
        simpegCellPt = indArr[:, 0:-1].copy()
        simpegCellPt[:, 2] = ( nCunderMesh[-1] + 2) - (simpegCellPt[:, 2] + indArr[:, 3])
        # Need reindex the z index to be from the bottom-left-close corner
        # and to be from the global bottom.
        simpegCellPt = simpegCellPt - np.array([1., 1., 1.])

        # Calculate the cell level
        simpegLevel = np.log2(np.min(nCunderMesh)) - np.log2(indArr[:, 3])
        # Make a pointer matrix
        simpegPointers = np.concatenate((simpegCellPt, simpegLevel.reshape((-1, 1))), axis=1)

        # Make the tree mesh
        mesh = TreeMesh([h1, h2, h3], x0=x0)
        mesh._cells = set([mesh._index(p) for p in simpegPointers.tolist()])

        # Figure out the reordering
        mesh._simpegReorderUBC = np.argsort(
            np.array([mesh._index(i) for i in simpegPointers.tolist()])
        )
        # mesh._simpegReorderUBC = np.argsort((np.array([[1, 1, 1, -1]])*simpegPointers).view(', '.join(4*['float'])), axis=0, order=['f3', 'f2', 'f1', 'f0'])[:, 0]

        return mesh

    def readModelUBC(mesh, fileName):
        """Read UBC OcTree model and get vector

        :param string fileName: path to the UBC GIF model file to read
        :rtype: numpy.ndarray
        :return: OcTree model
        """

        if type(fileName) is list:
            out = {}
            for f in fileName:
                out[f] = mesh.readModelUBC(f)
            return out

        assert hasattr(mesh, '_simpegReorderUBC'), 'The file must have been loaded from a UBC format.'
        assert mesh.dim == 3

        modList = []
        modArr = np.loadtxt(fileName)
        if len(modArr.shape) == 1:
            modList.append(modArr[mesh._simpegReorderUBC])
        else:
            modList.append(modArr[mesh._simpegReorderUBC, :])
        return modList

    def writeVTK(mesh, fileName, models=None):
        """Function to write a VTU file from a TreeMesh and model."""
        import vtk
        from vtk import vtkXMLUnstructuredGridWriter as Writer, VTK_VERSION
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        if str(type(mesh)).split()[-1][1:-2] not in 'discretize.TreeMesh.TreeMesh':
            raise IOError('mesh is not a TreeMesh.')

        # Make the data parts for the vtu object
        # Points
        mesh.number()
        ptsMat = mesh._gridN + mesh.x0

        vtkPts = vtk.vtkPoints()
        vtkPts.SetData(numpy_to_vtk(ptsMat, deep=True))
        # Cells
        cellConn = np.array([c.nodes for c in mesh], dtype=np.int64)

        cellsMat = np.concatenate((np.ones((cellConn.shape[0], 1), dtype=np.int64)*cellConn.shape[1], cellConn), axis=1).ravel()
        cellsArr = vtk.vtkCellArray()
        cellsArr.SetNumberOfCells(cellConn.shape[0])
        cellsArr.SetCells(cellConn.shape[0], numpy_to_vtkIdTypeArray(cellsMat, deep =True))

        # Make the object
        vtuObj = vtk.vtkUnstructuredGrid()
        vtuObj.SetPoints(vtkPts)
        vtuObj.SetCells(vtk.VTK_VOXEL, cellsArr)
        # Add the level of refinement as a cell array
        cellSides = np.array([np.array(vtuObj.GetCell(i).GetBounds()).reshape((3, 2)).dot(np.array([-1, 1])) for i in np.arange(vtuObj.GetNumberOfCells())])
        uniqueLevel, indLevel = np.unique(np.prod(cellSides, axis=1), return_inverse=True)
        refineLevelArr = numpy_to_vtk(indLevel.max() - indLevel, deep=1)
        refineLevelArr.SetName('octreeLevel')
        vtuObj.GetCellData().AddArray(refineLevelArr)
        # Assign the model('s) to the object
        if models is not None:
            for item in six.iteritems(models):
                # Convert numpy array
                vtkDoubleArr = numpy_to_vtk(item[1], deep=1)
                vtkDoubleArr.SetName(item[0])
                vtuObj.GetCellData().AddArray(vtkDoubleArr)

        # Make the writer
        vtuWriteFilter = Writer()
        if float(VTK_VERSION.split('.')[0]) >= 6:
            vtuWriteFilter.SetInputData(vtuObj)
        else:
            vtuWriteFilter.SetInput(vtuObj)
        vtuWriteFilter.SetFileName(fileName)
        # Write the file
        vtuWriteFilter.Update()
