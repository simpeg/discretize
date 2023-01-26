"""Module for reading and writing meshes to text files.

The text files representing meshes are often in the `UBC` format.
"""
import os
import numpy as np

from discretize.utils import mkvc
from discretize.utils.code_utils import deprecate_method
import warnings

try:
    from discretize.mixins.vtk_mod import (
        InterfaceTensorread_vtk,
        InterfaceSimplexReadVTK,
    )
except ImportError:
    InterfaceSimplexReadVTK = InterfaceTensorread_vtk = object


class TensorMeshIO(InterfaceTensorread_vtk):
    """Class for managing the input/output of tensor meshes and models.

    The ``TensorMeshIO`` class contains a set of class methods specifically
    for the :class:`~discretize.TensorMesh` class. These include:

        - Read/write tensor meshes to file
        - Read/write models defined on tensor meshes

    """

    @classmethod
    def _readUBC_3DMesh(cls, file_name):
        """Read 3D tensor mesh from UBC-GIF formatted file.

        Parameters
        ----------
        file_name : str or file name
            full path to the UBC-GIF formatted mesh file

        Returns
        -------
        discretize.TensorMesh
            The tensor mesh
        """
        # Read the file as line strings, remove lines with comment = !
        msh = np.genfromtxt(file_name, delimiter="\n", dtype=str, comments="!")

        # Interal function to read cell size lines for the UBC mesh files.
        def readCellLine(line):
            line_list = []
            for seg in line.split():
                if "*" in seg:
                    sp = seg.split("*")
                    seg_arr = np.ones((int(sp[0]),)) * float(sp[1])
                else:
                    seg_arr = np.array([float(seg)], float)
                line_list.append(seg_arr)
            return np.concatenate(line_list)

        # Fist line is the size of the model
        # sizeM = np.array(msh[0].split(), dtype=float)
        # Second line is the South-West-Top corner coordinates.
        origin = np.array(msh[1].split(), dtype=float)
        # Read the cell sizes
        h1 = readCellLine(msh[2])
        h2 = readCellLine(msh[3])
        h3temp = readCellLine(msh[4])
        # Invert the indexing of the vector to start from the bottom.
        h3 = h3temp[::-1]
        # Adjust the reference point to the bottom south west corner
        origin[2] = origin[2] - np.sum(h3)
        # Make the mesh
        tensMsh = cls([h1, h2, h3], origin=origin)
        return tensMsh

    @classmethod
    def _readUBC_2DMesh(cls, file_name):
        """Read 2D tensor mesh from UBC-GIF formatted file.

        Parameters
        ----------
        file_name : str or file name
            full path to the UBC-GIF formatted mesh file

        Returns
        -------
        discretize.TensorMesh
            The tensor mesh
        """
        fopen = open(file_name, "r")

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
                    xvec = np.hstack(
                        (xvec, np.ones(int(var[1])) * (var[0] - xend) / int(var[1]))
                    )
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
        z0 = -(z0 + sum(dz))
        dz = dz[::-1]
        # Make the mesh
        tensMsh = cls([dx, dz], origin=(x0, z0))

        fopen.close()

        return tensMsh

    @classmethod
    def read_UBC(cls, file_name, directory=""):
        """Read 2D or 3D tensor mesh from UBC-GIF formatted file.

        Parameters
        ----------
        file_name : str or file name
            full path to the UBC-GIF formatted mesh file or just its name if directory is specified
        directory : str, optional
            directory where the UBC-GIF file lives

        Returns
        -------
        discretize.TensorMesh
            The tensor mesh
        """
        # Check the expected mesh dimensions
        fname = os.path.join(directory, file_name)
        # Read the file as line strings, remove lines with comment = !
        msh = np.genfromtxt(fname, delimiter="\n", dtype=str, comments="!", max_rows=1)
        # Fist line is the size of the model
        sizeM = np.array(msh.ravel()[0].split(), dtype=float)
        # Check if the mesh is a UBC 2D mesh
        if sizeM.shape[0] == 1:
            Tnsmsh = cls._readUBC_2DMesh(fname)
        # Check if the mesh is a UBC 3D mesh
        elif sizeM.shape[0] == 3:
            Tnsmsh = cls._readUBC_3DMesh(fname)
        else:
            raise Exception("File format not recognized")
        return Tnsmsh

    def _readModelUBC_2D(mesh, file_name):
        """Read UBC-GIF formatted model file for 2D tensor mesh.

        Parameters
        ----------
        file_name : str or file name
            full path to the UBC-GIF formatted model file

        Returns
        -------
        (n_cells) numpy.ndarray
            The model defined on the 2D tensor mesh
        """
        # Open file and skip header... assume that we know the mesh already
        obsfile = np.genfromtxt(file_name, delimiter=" \n", dtype=str, comments="!")

        dim = tuple(np.array(obsfile[0].split(), dtype=int))
        if mesh.shape_cells != dim:
            raise Exception("Dimension of the model and mesh mismatch")

        model = []
        for line in obsfile[1:]:
            model.extend([float(val) for val in line.split()])
        model = np.asarray(model)
        if not len(model) == mesh.nC:
            raise Exception(
                """Something is not right, expected size is {:d}
                but unwrap vector is size {:d}""".format(
                    mesh.nC, len(model)
                )
            )

        return model.reshape(mesh.vnC, order="F")[:, ::-1].reshape(-1, order="F")

    def _readModelUBC_3D(mesh, file_name):
        """Read UBC-GIF formatted model file for 3D tensor mesh.

        Parameters
        ----------
        file_name : str or file name
            full path to the UBC-GIF formatted model file

        Returns
        -------
        (n_cells) numpy.ndarray
            The model defined on the 3D tensor mesh
        """
        f = open(file_name, "r")
        model = np.array(list(map(float, f.readlines())))
        f.close()
        nCx, nCy, nCz = mesh.shape_cells
        model = np.reshape(model, (nCz, nCx, nCy), order="F")
        model = model[::-1, :, :]
        model = np.transpose(model, (1, 2, 0))
        model = mkvc(model)
        return model

    def read_model_UBC(mesh, file_name, directory=""):
        """Read UBC-GIF formatted model file for 2D or 3D tensor mesh.

        Parameters
        ----------
        file_name : str or file name
            full path to the UBC-GIF formatted model file or just its name if directory is specified
        directory : str, optional
            directory where the UBC-GIF file lives

        Returns
        -------
        (n_cells) numpy.ndarray
            The model defined on the mesh
        """
        fname = os.path.join(directory, file_name)
        if mesh.dim == 3:
            model = mesh._readModelUBC_3D(fname)
        elif mesh.dim == 2:
            model = mesh._readModelUBC_2D(fname)
        else:
            raise Exception("mesh must be a Tensor Mesh 2D or 3D")
        return model

    def write_model_UBC(mesh, file_name, model, directory=""):
        """Write 2D or 3D tensor model to UBC-GIF formatted file.

        Parameters
        ----------
        file_name : str or file name
            full path for the output mesh file or just its name if directory is specified
        model : (n_cells) numpy.ndarray
            The model to write out.
        directory : str, optional
            output directory
        """
        fname = os.path.join(directory, file_name)
        if mesh.dim == 3:
            # Reshape model to a matrix
            modelMat = mesh.reshape(model, "CC", "CC", "M")
            # Transpose the axes
            modelMatT = modelMat.transpose((2, 0, 1))
            # Flip z to positive down
            modelMatTR = mkvc(modelMatT[::-1, :, :])
            np.savetxt(fname, modelMatTR.ravel())

        elif mesh.dim == 2:
            modelMat = mesh.reshape(model, "CC", "CC", "M").T[::-1]
            f = open(fname, "w")
            f.write("{:d} {:d}\n".format(*mesh.shape_cells))
            f.close()
            f = open(fname, "ab")
            np.savetxt(f, modelMat)
            f.close()

        else:
            raise Exception("mesh must be a Tensor Mesh 2D or 3D")

    def _writeUBC_3DMesh(mesh, file_name, comment_lines=""):
        """Write 3D tensor mesh to UBC-GIF formatted file.

        Parameters
        ----------
        file_name : str or file name
            full path for the output mesh file
        comment_lines : str, optional
            comment lines preceded are preceeded with '!'
        """
        if not mesh.dim == 3:
            raise Exception("Mesh must be 3D")

        s = comment_lines
        s += "{0:d} {1:d} {2:d}\n".format(*tuple(mesh.vnC))
        # Have to it in the same operation or use mesh.origin.copy(),
        # otherwise the mesh.origin is updated.
        origin = mesh.origin + np.array([0, 0, mesh.h[2].sum()])

        nCx, nCy, nCz = mesh.shape_cells
        s += "{0:.6f} {1:.6f} {2:.6f}\n".format(*tuple(origin))
        s += ("%.6f " * nCx + "\n") % tuple(mesh.h[0])
        s += ("%.6f " * nCy + "\n") % tuple(mesh.h[1])
        s += ("%.6f " * nCz + "\n") % tuple(mesh.h[2][::-1])
        f = open(file_name, "w")
        f.write(s)
        f.close()

    def _writeUBC_2DMesh(mesh, file_name, comment_lines=""):
        """Write 2D tensor mesh to UBC-GIF formatted file.

        Parameters
        ----------
        file_name : str or file name
            full path for the output mesh file
        comment_lines : str, optional
            comment lines preceded are preceeded with '!'
        """
        if not mesh.dim == 2:
            raise Exception("Mesh must be 2D")

        def writeF(fx, outStr=""):
            # Init
            i = 0
            origin = True
            x0 = fx[i]
            f = fx[i]
            number_segment = 0
            auxStr = ""

            while True:
                i = i + 1
                if i >= fx.size:
                    break
                dx = -f + fx[i]
                f = fx[i]
                n = 1

                for j in range(i + 1, fx.size):
                    if -f + fx[j] == dx:
                        n += 1
                        i += 1
                        f = fx[j]
                    else:
                        break

                number_segment += 1
                if origin:
                    auxStr += "{:.10f} {:.10f} {:d} \n".format(x0, f, n)
                    origin = False
                else:
                    auxStr += "{:.10f} {:d} \n".format(f, n)

            auxStr = "{:d}\n".format(number_segment) + auxStr
            outStr += auxStr

            return outStr

        # Grab face coordinates
        fx = mesh.nodes_x
        fz = -mesh.nodes_y[::-1]

        # Create the string
        outStr = comment_lines
        outStr = writeF(fx, outStr=outStr)
        outStr += "\n"
        outStr = writeF(fz, outStr=outStr)

        # Write file
        f = open(file_name, "w")
        f.write(outStr)
        f.close()

    def write_UBC(mesh, file_name, models=None, directory="", comment_lines=""):
        """Write 2D or 3D tensor mesh (and models) to UBC-GIF formatted file(s).

        Parameters
        ----------
        file_name : str or file name
            full path for the output mesh file or just its name if directory is specified
        models : dict of [str, (n_cells) numpy.ndarray], optional
            The dictionary key is a string representing the model's name. Each model
            is an (n_cells) array.
        directory : str, optional
            output directory
        comment_lines : str, optional
            comment lines preceded are preceeded with '!'
        """
        fname = os.path.join(directory, file_name)
        if mesh.dim == 3:
            mesh._writeUBC_3DMesh(fname, comment_lines=comment_lines)
        elif mesh.dim == 2:
            mesh._writeUBC_2DMesh(fname, comment_lines=comment_lines)
        else:
            raise Exception("mesh must be a Tensor Mesh 2D or 3D")

        if models is None:
            return
        if not isinstance(models, dict):
            raise TypeError("models must be a dict")
        for key in models:
            if not isinstance(key, str):
                raise TypeError(
                    "The dict key must be a string representing the file name"
                )
            mesh.write_model_UBC(key, models[key], directory=directory)

    # DEPRECATED
    @classmethod
    def readUBC(TensorMesh, file_name, directory=""):
        """Read 2D or 3D tensor mesh from UBC-GIF formatted file.

        *readUBC* has been deprecated and replaced by *read_UBC*
        See Also
        --------
        read_UBC
        """
        warnings.warn(
            "TensorMesh.readUBC has been deprecated and will be removed in"
            "discretize 1.0.0. please use TensorMesh.read_UBC",
            FutureWarning,
        )
        return TensorMesh.read_UBC(file_name, directory)

    readModelUBC = deprecate_method(
        "read_model_UBC", "readModelUBC", removal_version="1.0.0", future_warn=True
    )
    writeUBC = deprecate_method(
        "write_UBC", "writeUBC", removal_version="1.0.0", future_warn=True
    )
    writeModelUBC = deprecate_method(
        "write_model_UBC", "writeModelUBC", removal_version="1.0.0", future_warn=True
    )


class TreeMeshIO(object):
    """Class for managing the input/output of tree meshes and models.

    The ``TreeMeshIO`` class contains a set of class methods specifically
    for the :class:`~discretize.TreeMesh` class. These include:

        - Read/write tree meshes to file
        - Read/write models defined on tree meshes

    """

    @classmethod
    def read_UBC(TreeMesh, file_name, directory=""):
        """Read 3D tree mesh (OcTree mesh) from UBC-GIF formatted file.

        Parameters
        ----------
        file_name : str or file name
            full path to the UBC-GIF formatted mesh file or just its name if directory is specified
        directory : str, optional
            directory where the UBC-GIF file lives

        Returns
        -------
        discretize.TreeMesh
            The tree mesh
        """
        fname = os.path.join(directory, file_name)
        fileLines = np.genfromtxt(fname, dtype=str, delimiter="\n", comments="!")
        nCunderMesh = np.array(fileLines[0].split("!")[0].split(), dtype=int)
        tswCorn = np.array(fileLines[1].split("!")[0].split(), dtype=float)
        smallCell = np.array(fileLines[2].split("!")[0].split(), dtype=float)
        # Read the index array
        indArr = np.genfromtxt(
            (line.encode("utf8") for line in fileLines[4::]), dtype=np.int64
        )
        nCunderMesh = nCunderMesh[: len(tswCorn)]  # remove information related to core

        hs = [np.ones(nr) * sz for nr, sz in zip(nCunderMesh, smallCell)]
        origin = tswCorn
        origin[-1] -= np.sum(hs[-1])

        ls = np.log2(nCunderMesh).astype(int)
        # if all ls are equal
        if min(ls) == max(ls):
            max_level = ls[0]
        else:
            max_level = min(ls) + 1

        mesh = TreeMesh(hs, origin=origin)
        levels = indArr[:, -1]
        indArr = indArr[:, :-1]

        indArr -= 1  # shift by 1....
        indArr = 2 * indArr + levels[:, None]  # get cell center index
        indArr[:, -1] = 2 * nCunderMesh[-1] - indArr[:, -1]  # switch direction of iz
        levels = max_level - np.log2(levels)  # calculate level

        mesh.__setstate__((indArr, levels))
        return mesh

    def read_model_UBC(mesh, file_name):
        """Read UBC-GIF formatted file model file for 3D tree mesh (OcTree).

        Parameters
        ----------
        file_name : str or list of str
            full path to the UBC-GIF formatted model file or
            just its name if directory is specified. It can also be a list of file_names.
        directory : str
            directory where the UBC-GIF file lives (optional)

        Returns
        -------
        (n_cells) numpy.ndarray or dict of [str, (n_cells) numpy.ndarray]
            The model defined on the mesh. If **file_name** is a ``dict``, it is a
            dictionary of models indexed by the file names.
        """
        if type(file_name) is list:
            out = {}
            for f in file_name:
                out[f] = mesh.read_model_UBC(f)
            return out

        modArr = np.loadtxt(file_name)

        ubc_order = mesh._ubc_order
        # order_ubc will re-order from treemesh ordering to UBC ordering
        # need the opposite operation
        un_order = np.empty_like(ubc_order)
        un_order[ubc_order] = np.arange(len(ubc_order))

        model = modArr[un_order].copy()  # ensure a contiguous array
        return model

    def write_UBC(mesh, file_name, models=None, directory=""):
        """Write OcTree mesh (and models) to UBC-GIF formatted files.

        Parameters
        ----------
        file_name : str
            full path for the output mesh file or just its name if directory is specified
        models : dict of [str, (n_cells) numpy.ndarray], optional
            The dictionary key is a string representing the model's name.
            Each model is a 1D numpy array of size (n_cells).
        directory : str, optional
            output directory (optional)
        """
        uniform_hs = np.array([np.allclose(h, h[0]) for h in mesh.h])
        if np.any(~uniform_hs):
            raise Exception("UBC form does not support variable cell widths")
        nCunderMesh = np.array([h.size for h in mesh.h], dtype=np.int64)

        tswCorn = mesh.origin.copy()
        tswCorn[-1] += np.sum(mesh.h[-1])

        smallCell = np.array([h[0] for h in mesh.h])
        nrCells = mesh.nC

        indArr, levels = mesh._ubc_indArr
        ubc_order = mesh._ubc_order

        indArr = indArr[ubc_order]
        levels = levels[ubc_order]

        # Write the UBC octree mesh file
        head = " ".join([f"{int(n)}" for n in nCunderMesh]) + " \n"
        head += " ".join([f"{v:.4f}" for v in tswCorn]) + " \n"
        head += " ".join([f"{v:.3f}" for v in smallCell]) + " \n"
        head += f"{int(nrCells)}"
        np.savetxt(file_name, np.c_[indArr, levels], fmt="%i", header=head, comments="")

        # Print the models
        if models is None:
            return
        if not isinstance(models, dict):
            raise TypeError("models must be a dict")
        for key in models:
            if not isinstance(key, str):
                raise TypeError(
                    "The dict key must be a string representing the file name"
                )
            mesh.write_model_UBC(key, models[key], directory=directory)

    def write_model_UBC(mesh, file_name, model, directory=""):
        """Write 3D tree model (OcTree) to UBC-GIF formatted file.

        Parameters
        ----------
        file_name : str
            full path for the output mesh file or just its name if directory is specified
        model : (n_cells) numpy.ndarray
            model values defined for each cell
        directory : str
            output directory (optional)
        """
        if type(file_name) is list:
            for f, m in zip(file_name, model):
                mesh.write_model_UBC(f, m)
        else:
            ubc_order = mesh._ubc_order
            fname = os.path.join(directory, file_name)
            m = model[ubc_order]
            np.savetxt(fname, m)

    # DEPRECATED
    @classmethod
    def readUBC(TreeMesh, file_name, directory=""):
        """Read 3D Tree mesh from UBC-GIF formatted file.

        *readUBC* has been deprecated and replaced by *read_UBC*

        See Also
        --------
        read_UBC
        """
        warnings.warn(
            "TensorMesh.readUBC has been deprecated and will be removed in"
            "discretize 1.0.0. please use TensorMesh.read_UBC",
            FutureWarning,
        )
        return TreeMesh.read_UBC(file_name, directory)

    readModelUBC = deprecate_method(
        "read_model_UBC", "readModelUBC", removal_version="1.0.0", future_warn=True
    )
    writeUBC = deprecate_method(
        "write_UBC", "writeUBC", removal_version="1.0.0", future_warn=True
    )
    writeModelUBC = deprecate_method(
        "write_model_UBC", "writeModelUBC", removal_version="1.0.0", future_warn=True
    )


class SimplexMeshIO(InterfaceSimplexReadVTK):
    """Empty class for future text based IO of a SimplexMesh."""

    pass
