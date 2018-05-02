from .TensorMesh import BaseTensorMesh
from .InnerProducts import InnerProducts
from .tree_ext import _TreeMesh
import numpy as np
from scipy.spatial import Delaunay
import scipy.sparse as sp
from . import utils
import six

class TreeMesh(_TreeMesh, BaseTensorMesh, InnerProducts):
    _meshType = 'TREE'

    #inheriting stuff from BaseTensorMesh that isn't defined in _QuadTree
    def __init__(self, h, x0=None, levels=None, **kwargs):
        BaseTensorMesh.__init__(self, h, x0, **kwargs)

        if levels is None:
            levels = int(np.log2(len(self.h[0])))

        if(self.dim == 2):
            xF = np.array([self.vectorNx[-1], self.vectorNy[-1]])
        else:
            xF = np.array([self.vectorNx[-1], self.vectorNy[-1], self.vectorNz[-1]])
        ws = xF-self.x0

        # Now can initialize cpp tree parent
        _TreeMesh.__init__(self, levels, self.x0, ws)

    def __str__(self):
        outStr = '  ---- {0!s}TreeMesh ----  '.format(
            ('Oc' if self.dim == 3 else 'Quad')
        )

        def printH(hx, outStr=''):
            i = -1
            while True:
                i = i + 1
                if i > hx.size:
                    break
                elif i == hx.size:
                    break
                h = hx[i]
                n = 1
                for j in range(i+1, hx.size):
                    if hx[j] == h:
                        n = n + 1
                        i = i + 1
                    else:
                        break
                if n == 1:
                    outStr += ' {0:.2f}, '.format(h)
                else:
                    outStr += ' {0:d}*{1:.2f}, '.format(n, h)
            return outStr[:-1]

        if self.dim == 2:
            outStr += '\n   x0: {0:.2f}'.format(self.x0[0])
            outStr += '\n   y0: {0:.2f}'.format(self.x0[1])
            outStr += printH(self.hx, outStr='\n   hx:')
            outStr += printH(self.hy, outStr='\n   hy:')
        elif self.dim == 3:
            outStr += '\n   x0: {0:.2f}'.format(self.x0[0])
            outStr += '\n   y0: {0:.2f}'.format(self.x0[1])
            outStr += '\n   z0: {0:.2f}'.format(self.x0[2])
            outStr += printH(self.hx, outStr='\n   hx:')
            outStr += printH(self.hy, outStr='\n   hy:')
            outStr += printH(self.hz, outStr='\n   hz:')
        outStr += '\n  nC: {0:d}'.format(self.nC)
        outStr += '\n  Fill: {0:2.2f}%'.format((self.fill*100))
        return outStr

    @property
    def vntF(self):
        return [self.ntFx, self.ntFy] + ([] if self.dim == 2 else [self.ntFz])

    @property
    def vntE(self):
        return [self.ntEx, self.ntEy] + ([] if self.dim == 2 else [self.ntEz])

    @property
    def cellGradStencil(self):
        if getattr(self, '_cellGradStencil', None) is None:

            self._cellGradStencil = sp.vstack([
                self._cellGradxStencil(), self._cellGradyStencil()
            ])
            if self.dim == 3:
                self._cellGradStencil = sp.vstack([
                    self._cellGradStencil, self._cellGradzStencil()
                ])

        return self._cellGradStencil

    @property
    def cellGrad(self):
        """
        Cell centered Gradient operator built off of the faceDiv operator.
        Grad =  - (Mf)^{-1} * Div * diag (volume)
        """
        if getattr(self, '_cellGrad', None) is None:

            indBoundary = np.ones(self.nC, dtype=float)

            indBoundary_Fx = (self.aveFx2CC.T * indBoundary) >= 1
            ix = np.zeros(self.nFx)
            ix[indBoundary_Fx] = 1.
            Pafx = sp.diags(ix)

            indBoundary_Fy = (self.aveFy2CC.T * indBoundary) >= 1
            iy = np.zeros(self.nFy)
            iy[indBoundary_Fy] = 1.
            Pafy = sp.diags(iy)

            MfI = self.getFaceInnerProduct(invMat=True)

            if self.dim == 2:
                Pi = sp.block_diag([Pafx, Pafy])

            elif self.dim == 3:
                indBoundary_Fz = (self.aveFz2CC.T * indBoundary) >= 1
                iz = np.zeros(self.nFz)
                iz[indBoundary_Fz] = 1.
                Pafz = sp.diags(iz)
                Pi = sp.block_diag([Pafx, Pafy, Pafz])

            self._cellGrad = -Pi * MfI * self.faceDiv.T * sp.diags(self.vol)

        return self._cellGrad

    @property
    def cellGradx(self):
        """
        Cell centered Gradient operator in x-direction (Gradx)
        Grad = sp.vstack((Gradx, Grady, Gradz))
        """
        if getattr(self, '_cellGradx', None) is None:

            nFx = self.nFx
            indBoundary = np.ones(self.nC, dtype=float)

            indBoundary_Fx = (self.aveFx2CC.T * indBoundary) >= 1
            ix = np.zeros(self.nFx)
            ix[indBoundary_Fx] = 1.
            Pafx = sp.diags(ix)

            MfI = self.getFaceInnerProduct(invMat=True)
            MfIx = sp.diags(MfI.diagonal()[:nFx])

            self._cellGradx = (
                -Pafx * MfIx * self.faceDivx.T * sp.diags(self.vol)
            )

        return self._cellGradx

    @property
    def cellGrady(self):
        """
        Cell centered Gradient operator in y-direction (Gradx)
        Grad = sp.vstack((Gradx, Grady, Gradz))
        """
        if getattr(self, '_cellGrady', None) is None:

            nFx = self.nFx
            nFy = self.nFy
            indBoundary = np.ones(self.nC, dtype=float)

            indBoundary_Fy = (self.aveFy2CC.T * indBoundary) >= 1
            iy = np.zeros(self.nFy)
            iy[indBoundary_Fy] = 1.
            Pafy = sp.diags(iy)

            MfI = self.getFaceInnerProduct(invMat=True)
            MfIy = sp.diags(MfI.diagonal()[nFx:nFx+nFy])

            self._cellGrady = (
                -Pafy * MfIy * self.faceDivy.T * sp.diags(self.vol)
            )

        return self._cellGrady

    @property
    def cellGradz(self):
        """
        Cell centered Gradient operator in y-direction (Gradz)
        Grad = sp.vstack((Gradx, Grady, Gradz))
        """
        if getattr(self, '_cellGradz', None) is None:

            nFx = self.nFx
            nFy = self.nFy
            indBoundary = np.ones(self.nC, dtype=float)

            indBoundary_Fz = (self.aveFz2CC.T * indBoundary) >= 1
            iz = np.zeros(self.nFz)
            iz[indBoundary_Fz] = 1.
            Pafz = sp.diags(iz)

            MfI = self.getFaceInnerProduct(invMat=True)
            MfIz = sp.diags(MfI.diagonal()[nFx+nFy:])

            self._cellGradz = (
                -Pafz * MfIz * self.faceDivz.T * sp.diags(self.vol)
            )

        return self._cellGradz

    @property
    def faceDivx(self):
        if getattr(self, '_faceDivx', None) is None:
            print(self.faceDiv.shape, self.nFx)
            self._faceDivx = self.faceDiv[:, :self.nFx]
        return self._faceDivx

    @property
    def faceDivy(self):
        if getattr(self, '_faceDivy', None) is None:
            self._faceDivy = self.faceDiv[:, self.nFx:self.nFx+self.nFy]
        return self._faceDivy

    @property
    def faceDivz(self):
        if getattr(self, '_faceDivz', None) is None:
            self._faceDivz = self.faceDiv[:, self.nFx+self.nFy:]
        return self._faceDivz

    @property
    def aveCC2Fx(self):
        "Construct the averaging operator on cell centers to cell x-faces."
        if getattr(self, '_aveCC2Fx', None) is None:
            tri = Delaunay(self.gridCC)
            gridF = self.gridFx
            simplexes = tri.find_simplex(gridF)
            nd = self.dim
            ns = nd+1
            nf = self.nFx
            I = np.zeros(nf*ns, dtype=np.int64)
            J = np.zeros(nf*ns, dtype=np.int64)
            V = np.zeros(nf*ns, dtype=np.float64)

            simp = tri.simplices[simplexes]
            trans = tri.transform[simplexes]
            shift = gridF-trans[:, nd]
            bs = np.einsum('ikj,ij->ik', trans[:, :nd], shift)
            bs = np.c_[bs, 1-bs.sum(axis=1)]

            I = np.zeros(nf*ns, dtype=np.int64)
            J = np.zeros(nf*ns, dtype=np.int64)
            V = np.zeros(nf*ns, dtype=np.float64)
            for i in range(nf):
                I[ns*i: ns*i+ns] = i
                if simplexes[i] == -1:
                    # Extrapolating.... from nearest cell
                    ic = self._get_containing_cell_index(gridF[i])
                    J[ns*i] = ic
                    V[ns*i] = 1.0
                    # rest are zeros from definition of J and V
                else:
                    J[ns*i:ns*i+ns] = simp[i]
                    V[ns*i:ns*i+ns] = bs[i]
            self._aveCC2Fx = sp.csr_matrix((V, (I, J)))
        return self._aveCC2Fx

    @property
    def aveCC2Fy(self):
        "Construct the averaging operator on cell centers to cell y-faces."
        if getattr(self, '_aveCC2Fy', None) is None:
            tri = Delaunay(self.gridCC)
            gridF = self.gridFy
            simplexes = tri.find_simplex(gridF)
            nd = self.dim
            ns = nd+1
            nf = self.nFy
            I = np.zeros(nf*ns, dtype=np.int64)
            J = np.zeros(nf*ns, dtype=np.int64)
            V = np.zeros(nf*ns, dtype=np.float64)

            simp = tri.simplices[simplexes]
            trans = tri.transform[simplexes]
            shift = gridF-trans[:, nd]
            bs = np.einsum('ikj,ij->ik', trans[:, :nd], shift)
            bs = np.c_[bs, 1-bs.sum(axis=1)]

            I = np.zeros(nf*ns, dtype=np.int64)
            J = np.zeros(nf*ns, dtype=np.int64)
            V = np.zeros(nf*ns, dtype=np.float64)
            for i in range(nf):
                I[ns*i: ns*i+ns] = i
                if simplexes[i] == -1:
                    # Extrapolating.... from nearest cell
                    ic = self._get_containing_cell_index(gridF[i])
                    J[ns*i] = ic
                    V[ns*i] = 1.0
                    # rest are zeros from definition of J and V
                else:
                    J[ns*i:ns*i+ns] = simp[i]
                    V[ns*i:ns*i+ns] = bs[i]
            self._aveCC2Fy = sp.csr_matrix((V, (I, J)))
        return self._aveCC2Fy

    @property
    def aveCC2Fz(self):
        "Construct the averaging operator on cell centers to cell z-faces."
        if self.dim == 2:
            raise Exception('TreeMesh has no z-faces in 2D')
        if getattr(self, '_aveCC2Fz', None) is None:
            tri = Delaunay(self.gridCC)
            gridF = self.gridFz
            simplexes = tri.find_simplex(gridF)
            nd = self.dim
            ns = nd+1
            nf = self.nFz
            I = np.zeros(nf*ns, dtype=np.int64)
            J = np.zeros(nf*ns, dtype=np.int64)
            V = np.zeros(nf*ns, dtype=np.float64)

            simp = tri.simplices[simplexes]

            trans = tri.transform[simplexes]
            shift = gridF-trans[:, nd]
            bs = np.einsum('ikj,ij->ik', trans[:, :nd], shift)
            bs = np.c_[bs, 1-bs.sum(axis=1)]

            I = np.zeros(nf*ns, dtype=np.int64)
            J = np.zeros(nf*ns, dtype=np.int64)
            V = np.zeros(nf*ns, dtype=np.float64)
            for i in range(nf):
                I[ns*i: ns*i+ns] = i
                if simplexes[i] == -1:
                    # Extrapolating.... from nearest cell
                    ic = self._get_containing_cell_index(gridF[i])
                    J[ns*i] = ic
                    V[ns*i] = 1.0
                    # rest are zeros from definition of J and V
                else:
                    J[ns*i:ns*i+ns] = simp[i]
                    V[ns*i:ns*i+ns] = bs[i]
            self._aveCC2Fz = sp.csr_matrix((V, (I, J)))
        return self._aveCC2Fz

    def point2index(self, locs):
        locs = utils.asArray_N_x_Dim(locs, self.dim)

        inds = np.empty(locs.shape[0], dtype=np.int64)
        for ind, loc in enumerate(locs):
            inds[ind] = self._get_containing_cell_index(loc)
        return inds

    @classmethod
    def readUBC(self, meshFile):
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

        max_level = int(np.log2(nCunderMesh[0]))
        mesh = TreeMesh([h1, h2, h3], x0=x0, levels=max_level)

        # Convert indArr to points in local coordinates of underlying cpp tree
        # indArr is ix, iy, iz(top-down) need it in ix, iy, iz (bottom-up)
        indArr[:, :-1] -= 1 #shift by 1....
        indArr[:, :-1] = 2*indArr[:, :-1] + indArr[:, -1, None]
        indArr[:, 2] = (2<<max_level) - indArr[:, 2]

        indArr[:, -1] = max_level-np.log2(indArr[:, -1])

        mesh._insert_cells(indArr)
        mesh.number()
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

        modList.append(modArr[un_order])
        return modList

    def writeUBC(mesh, fileName, models=None):
        """Write UBC ocTree mesh and model files from a
        octree mesh and model.
        :param string fileName: File to write to
        :param dict models: Models in a dict, where each key is the filename
        """
        nCunderMesh = np.array([h.size for h in mesh.h], dtype=np.int64)
        tswCorn = mesh.x0 + np.array([0, 0, np.sum(mesh.h[2])])
        smallCell = np.array([h.min() for h in mesh.h])
        nrCells = mesh.nC

        indArr = mesh._ubc_indArr
        ubc_order = mesh._ubc_order

        indArr = indArr[ubc_order]

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
                np.savetxt(item[0], item[1][ubc_order], fmt='%3.5e')


    def writeVTK(self, fileName, models=None):
        """Function to write a VTU file from a TreeMesh and model."""
        import vtk
        from vtk import vtkXMLUnstructuredGridWriter as Writer, VTK_VERSION
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        # Make the data parts for the vtu object
        # Points
        ptsMat = np.vstack((self.gridN, self.gridhN))

        vtkPts = vtk.vtkPoints()
        vtkPts.SetData(numpy_to_vtk(ptsMat, deep=True))
        # Cells
        cellConn = np.array([c.nodes for c in self], dtype=np.int64)

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
