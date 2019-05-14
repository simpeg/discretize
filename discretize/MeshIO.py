import os
import json
import properties
import numpy as np
import six

from . import utils
from .base import BaseMesh

try:
    from .mixins import InterfaceTensorread_vtk
except ImportError as err:
    InterfaceTensorread_vtk = object


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


class TensorMeshIO(InterfaceTensorread_vtk):

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

    @classmethod
    def readUBC(TreeMesh, meshFile):
        """Read UBC 3D OcTree mesh file
        Input:
        :param str meshFile: path to the UBC GIF OcTree mesh file to read
        :rtype: discretize.TreeMesh
        :return: The octree mesh
        """
        fileLines = np.genfromtxt(meshFile, dtype=str,
                                  delimiter='\n', comments='!')
        nCunderMesh = np.array(fileLines[0].split('!')[0].split(), dtype=int)
        nCunderMesh = nCunderMesh[0:3]

        tswCorn = np.array(
            fileLines[1].split('!')[0].split(),
            dtype=float
        )
        smallCell = np.array(
            fileLines[2].split('!')[0].split(),
            dtype=float
        )
        # Read the index array
        indArr = np.genfromtxt((line.encode('utf8') for line in fileLines[4::]),
                               dtype=np.int)

        h1, h2, h3 = [np.ones(nr)*sz for nr, sz in zip(nCunderMesh, smallCell)]
        x0 = tswCorn - np.array([0, 0, np.sum(h3)])

        ls = np.log2(nCunderMesh).astype(int)
        if ls[0] == ls[1] and ls[1] == ls[2]:
            max_level = ls[0]
        else:
            max_level = min(ls)+1

        mesh = TreeMesh([h1, h2, h3], x0=x0)

        # Convert indArr to points in coordinates of underlying cpp tree
        # indArr is ix, iy, iz(top-down) need it in ix, iy, iz (bottom-up)

        levels = indArr[:, -1]
        indArr = indArr[:, :-1]

        indArr -= 1  # shift by 1....
        indArr = 2*indArr + levels[:, None]  # get cell center index
        indArr[:, 2] = 2*nCunderMesh[2] - indArr[:, 2]  # switch direction of iz
        levels = max_level-np.log2(levels)  # calculate level

        mesh.__setstate__((indArr, levels))
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

        assert mesh.dim == 3

        modList = []
        modArr = np.loadtxt(fileName)

        ubc_order = mesh._ubc_order
        # order_ubc will re-order from treemesh ordering to UBC ordering
        # need the opposite operation
        un_order = np.empty_like(ubc_order)
        un_order[ubc_order] = np.arange(len(ubc_order))

        model = modArr[un_order].copy() # ensure a contiguous array
        return model

    def writeUBC(mesh, fileName, models=None):
        """Write UBC ocTree mesh and model files from a
        octree mesh and model.
        :param string fileName: File to write to
        :param dict models: Models in a dict, where each key is the filename
        """
        uniform_hs = np.array([np.allclose(h, h[0]) for h in mesh.h])
        if np.any(~uniform_hs):
            raise Exception('UBC form does not support variable cell widths')
        nCunderMesh = np.array([h.size for h in mesh.h], dtype=np.int64)
        tswCorn = mesh.x0 + np.array([0, 0, np.sum(mesh.h[2])])
        smallCell = np.array([h[0] for h in mesh.h])
        nrCells = mesh.nC

        indArr, levels = mesh._ubc_indArr
        ubc_order = mesh._ubc_order

        indArr = indArr[ubc_order]
        levels = levels[ubc_order]

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
        np.savetxt(fileName, np.c_[indArr, levels], fmt='%i', header=head, comments='')

        # Print the models
        # Assign the model('s) to the object
        if models is not None:
            for item in six.iteritems(models):
                # Save the data
                np.savetxt(item[0], item[1][ubc_order], fmt='%3.5e')
