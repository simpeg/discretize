from __future__ import print_function
import numpy as np
from scipy import sparse as sp
from six import string_types
import warnings
from discretize.utils import sdiag, speye, kron3, spzeros, ddx, av, av_extrap


def checkBC(bc):
    """Checks if boundary condition 'bc' is valid.

    Each bc must be either 'dirichlet' or 'neumann'

    """
    if isinstance(bc, string_types):
        bc = [bc, bc]
    assert isinstance(bc, list), 'bc must be a list'
    assert len(bc) == 2, 'bc must have two elements'

    for bc_i in bc:
        assert isinstance(bc_i, string_types), "each bc must be a string"
        if bc_i not in ['dirichlet', 'neumann']:
            raise AssertionError(
                "each bc must be either, 'dirichlet' or 'neumann'"
            )
    return bc


def ddxCellGrad(n, bc):
    """
    Create 1D derivative operator from cell-centers to nodes this means we
    go from n to n+1

    For Cell-Centered **Dirichlet**, use a ghost point::

        (u_1 - u_g)/hf = grad

            u_g       u_1      u_2
             *    |    *   |    *     ...
                  ^
                  0

        u_g = - u_1
        grad = 2*u1/dx
        negitive on the other side.

    For Cell-Centered **Neumann**, use a ghost point::

        (u_1 - u_g)/hf = 0

            u_g       u_1      u_2
             *    |    *   |    *     ...

        u_g = u_1
        grad = 0;  put a zero in.

    """
    bc = checkBC(bc)

    D = sp.spdiags((np.ones((n+1, 1))*[-1, 1]).T, [-1, 0], n+1, n,
                   format="csr")
    # Set the first side
    if(bc[0] == 'dirichlet'):
        D[0, 0] = 2
    elif(bc[0] == 'neumann'):
        D[0, 0] = 0
    # Set the second side
    if(bc[1] == 'dirichlet'):
        D[-1, -1] = -2
    elif(bc[1] == 'neumann'):
        D[-1, -1] = 0
    return D


def ddxCellGradBC(n, bc):
    """
    Create 1D derivative operator from cell-centers to nodes this means we
    go from n to n+1

    For Cell-Centered **Dirichlet**, use a ghost point::

        (u_1 - u_g)/hf = grad

         u_g       u_1      u_2
          *    |    *   |    *     ...
               ^
              u_b

    We know the value at the boundary (u_b)::

        (u_g+u_1)/2 = u_b               (the average)
        u_g = 2*u_b - u_1

        So plug in to gradient:

        (u_1 - (2*u_b - u_1))/hf = grad
        2*(u_1-u_b)/hf = grad

    Separate, because BC are known (and can move to RHS later)::

        ( 2/hf )*u_1 + ( -2/hf )*u_b = grad

                       (   ^   ) JUST RETURN THIS


    """
    bc = checkBC(bc)

    ij = (np.array([0, n]), np.array([0, 1]))
    vals = np.zeros(2)

    # Set the first side
    if(bc[0] == 'dirichlet'):
        vals[0] = -2
    elif(bc[0] == 'neumann'):
        vals[0] = 0
    # Set the second side
    if(bc[1] == 'dirichlet'):
        vals[1] = 2
    elif(bc[1] == 'neumann'):
        vals[1] = 0
    D = sp.csr_matrix((vals, ij), shape=(n+1, 2))
    return D


class DiffOperators(object):
    """
    Class creates the differential operators that you need!
    """
    def __init__(self):
        raise Exception(
            'DiffOperators is a base class providing differential'
            'operators on meshes and cannot run on its own.'
            'Inherit to your favorite Mesh class.'
        )

    ###########################################################################
    #                                                                         #
    #                             Face Divergence                             #
    #                                                                         #
    ###########################################################################
    @property
    def _faceDivStencilx(self):
        """
        Face divergence operator in the x-direction (x-faces to cell centers)
        """
        if self.dim == 1:
            Dx = ddx(self.nCx)
        elif self.dim == 2:
            Dx = sp.kron(speye(self.nCy), ddx(self.nCx))
        elif self.dim == 3:
            Dx = kron3(speye(self.nCz), speye(self.nCy), ddx(self.nCx))
        return Dx

    @property
    def _faceDivStencily(self):
        """
        Face divergence operator in the y-direction (y-faces to cell centers)
        """
        if self.dim == 1:
            return None
        elif self.dim == 2:
            Dy = sp.kron(ddx(self.nCy), speye(self.nCx))
        elif self.dim == 3:
            Dy = kron3(speye(self.nCz), ddx(self.nCy), speye(self.nCx))
        return Dy

    @property
    def _faceDivStencilz(self):
        """
        Face divergence operator in the z-direction (z-faces to cell centers)
        """
        if self.dim == 1 or self.dim == 2:
            return None
        elif self.dim == 3:
            Dz = kron3(ddx(self.nCz), speye(self.nCy), speye(self.nCx))
        return Dz

    @property
    def _faceDivStencil(self):
        # Compute faceDivergence stencil on faces
        if self.dim == 1:
            D = self._faceDivStencilx
        elif self.dim == 2:
            D = sp.hstack((
                self._faceDivStencilx,
                self._faceDivStencily
            ), format="csr")
        elif self.dim == 3:
            D = sp.hstack((
                self._faceDivStencilx,
                self._faceDivStencily,
                self._faceDivStencilz
            ), format="csr")
        return D

    @property
    def faceDiv(self):
        """
        Construct divergence operator (face-stg to cell-centres).
        """
        if getattr(self, '_faceDiv', None) is None:
            # Get the stencil of +1, -1's
            D = self._faceDivStencil
            # Compute areas of cell faces & volumes
            S = self.area
            V = self.vol
            self._faceDiv = sdiag(1/V)*D*sdiag(S)
        return self._faceDiv

    @property
    def faceDivx(self):
        """
        Construct divergence operator in the x component (face-stg to
        cell-centres).
        """
        # Compute areas of cell faces & volumes
        S = self.r(self.area, 'F', 'Fx', 'V')
        V = self.vol
        return sdiag(1/V)*self._faceDivStencilx*sdiag(S)

    @property
    def faceDivy(self):
        if(self.dim < 2):
            return None
        # Compute areas of cell faces & volumes
        S = self.r(self.area, 'F', 'Fy', 'V')
        V = self.vol
        return sdiag(1/V)*self._faceDivStencily*sdiag(S)

    @property
    def faceDivz(self):
        """
        Construct divergence operator in the z component (face-stg to
        cell-centers).
        """
        if(self.dim < 3):
            return None
        # Compute areas of cell faces & volumes
        S = self.r(self.area, 'F', 'Fz', 'V')
        V = self.vol
        return sdiag(1/V)*self._faceDivStencilz*sdiag(S)

    ###########################################################################
    #                                                                         #
    #                          Nodal Diff Operators                           #
    #                                                                         #
    ###########################################################################

    @property
    def _nodalGradStencilx(self):
        """
        Stencil for the nodal grad in the x-direction (nodes to x-edges)
        """
        if self.dim == 1:
            Gx = ddx(self.nCx)
        elif self.dim == 2:
            Gx = sp.kron(speye(self.nNy), ddx(self.nCx))
        elif self.dim == 3:
            Gx = kron3(speye(self.nNz), speye(self.nNy), ddx(self.nCx))
        return Gx

    @property
    def _nodalGradStencily(self):
        """
        Stencil for the nodal grad in the y-direction (nodes to y-edges)
        """
        if self.dim == 1:
            return None
        elif self.dim == 2:
            Gy = sp.kron(ddx(self.nCy), speye(self.nNx))
        elif self.dim == 3:
            Gy = kron3(speye(self.nNz), ddx(self.nCy), speye(self.nNx))
        return Gy

    @property
    def _nodalGradStencilz(self):
        """
        Stencil for the nodal grad in the z-direction (nodes to z- edges)
        """
        if self.dim == 1 or self.dim == 2:
            return None
        else:
            Gz = kron3(ddx(self.nCz), speye(self.nNy), speye(self.nNx))
        return Gz

    @property
    def _nodalGradStencil(self):
        """
        Stencil for the nodal grad
        """
        # Compute divergence operator on faces
        if self.dim == 1:
            G = self._nodalGradStencilx
        elif self.dim == 2:
            G = sp.vstack((
                self._nodalGradStencilx,
                self._nodalGradStencily
            ), format="csr")
        elif self.dim == 3:
            G = sp.vstack((
                self._nodalGradStencilx,
                self._nodalGradStencily,
                self._nodalGradStencilz
            ), format="csr")
        return G

    @property
    def nodalGrad(self):
        """
        Construct gradient operator (nodes to edges).
        """
        if getattr(self, '_nodalGrad', None) is None:
            G = self._nodalGradStencil
            L = self.edge
            self._nodalGrad = sdiag(1/L)*G
        return self._nodalGrad

    @property
    def _nodalLaplacianStencilx(self):
        warnings.warn('Laplacian has not been tested rigorously.')

        Dx = ddx(self.nCx)
        Lx = - Dx.T * Dx

        if self.dim == 2:
            Lx = sp.kron(speye(self.nNy), Lx)
        elif self.dim == 3:
            Lx = kron3(speye(self.nNz), speye(self.nNy), Lx)
        return Lx

    @property
    def _nodalLaplacianStencily(self):
        warnings.warn('Laplacian has not been tested rigorously.')

        if self.dim == 1:
            return None

        Dy = ddx(self.nCy)
        Ly = - Dy.T * Dy

        if self.dim == 2:
            Ly = sp.kron(Ly, speye(self.nNx))
        elif self.dim == 3:
            Ly = kron3(speye(self.nNz), Ly, speye(self.nNx))
        return Ly

    @property
    def _nodalLaplacianStencilz(self):
        warnings.warn('Laplacian has not been tested rigorously.')

        if self.dim == 1 or self.dim == 2:
            return None

        Dz = ddx(self.nCz)
        Lz = - Dz.T * Dz
        return kron3(Lz, speye(self.nNy), speye(self.nNx))

    @property
    def _nodalLaplacianx(self):
        Hx = sdiag(1./self.hx)
        if self.dim == 2:
            Hx = sp.kron(speye(self.nNy), Hx)
        elif self.dim == 3:
            Hx = kron3(speye(self.nNz), speye(self.nNy), Hx)
        return Hx.T * self._nodalGradStencilx * Hx

    @property
    def _nodalLaplaciany(self):
        Hy = sdiag(1./self.hy)
        if self.dim == 1:
            return None
        elif self.dim == 2:
            Hy = sp.kron(Hy, speye(self.nNx))
        elif self.dim == 3:
            Hy = kron3(speye(self.nNz), Hy, speye(self.nNx))
        return Hy.T * self._nodalGradStencily * Hy

    @property
    def _nodalLaplacianz(self):
        if self.dim == 1 or self.dim == 2:
            return None
        Hz = sdiag(1./self.hz)
        Hz = kron3(Hz, speye(self.nNy), speye(self.nNx))
        return Hz.T * self._nodalLaplacianStencilz * Hz

    @property
    def nodalLaplacian(self):
        """
        Construct laplacian operator (nodes to edges).
        """
        if getattr(self, '_nodalLaplacian', None) is None:
            warnings.warn('Laplacian has not been tested rigorously.')
            # Compute divergence operator on faces
            if self.dim == 1:
                self._nodalLaplacian = self._nodalLaplacianx
            elif self.dim == 2:
                self._nodalLaplacian = (
                    self._nodalLaplacianx +
                    self._nodalLaplaciany
                )
            elif self.dim == 3:
                self._nodalLaplacian = (
                    self._nodalLaplacianx +
                    self._nodalLaplaciany +
                    self._nodalLaplacianz
                )
        return self._nodalLaplacian

    ###########################################################################
    #                                                                         #
    #                                Cell Grad                                #
    #                                                                         #
    ###########################################################################

    _cellGradBC_list = 'neumann'

    def setCellGradBC(self, BC):
        """
        Function that sets the boundary conditions for cell-centred derivative
        operators.

        Example
        -------
        ..code:: python

            # Neumann in all directions
            BC = 'neumann'

            # 3D, Dirichlet in y Neumann else
            BC = ['neumann', 'dirichlet', 'neumann']

            # 3D, Neumann in x on bottom of domain,  Dirichlet else
            BC = [['neumann', 'dirichlet'], 'dirichlet', 'dirichlet']
        """

        if isinstance(BC, string_types):
            BC = [BC]*self.dim
        if isinstance(BC, list):
            assert len(BC) == self.dim, 'BC list must be the size of your mesh'
        else:
            raise Exception("BC must be a str or a list.")

        for i, bc_i in enumerate(BC):
            BC[i] = checkBC(bc_i)

        # ensure we create a new gradient next time we call it
        self._cellGrad = None
        self._cellGradBC = None
        self._cellGradBC_list = BC
        return BC

    @property
    def _cellGradxStencil(self):
        # TODO: remove this hard-coding
        BC = ['neumann', 'neumann']
        if self.dim == 1:
            G1 = ddxCellGrad(self.nCx, BC)
        elif self.dim == 2:
            G1 = sp.kron(speye(self.nCy), ddxCellGrad(self.nCx, BC))
        elif self.dim == 3:
            G1 = kron3(
                speye(self.nCz),
                speye(self.nCy),
                ddxCellGrad(self.nCx, BC)
            )
        return G1

    @property
    def _cellGradyStencil(self):
        if self.dim < 2:
            return None
        BC = ['neumann', 'neumann'] # TODO: remove this hard-coding
        n = self.vnC
        if(self.dim == 2):
            G2 = sp.kron(ddxCellGrad(n[1], BC), speye(n[0]))
        elif self.dim == 3:
            G2 = kron3(speye(n[2]), ddxCellGrad(n[1], BC), speye(n[0]))
        return G2

    @property
    def _cellGradzStencil(self):
        if self.dim < 3:
            return None
        BC = ['neumann', 'neumann'] # TODO: remove this hard-coding
        n = self.vnC
        G3 = kron3(ddxCellGrad(n[2], BC), speye(n[1]), speye(n[0]))
        return G3

    @property
    def _cellGradStencil(self):
        BC = self.setCellGradBC(self._cellGradBC_list)
        if self.dim == 1:
            G = ddxCellGrad(self.nCx, BC[0])
        elif self.dim == 2:
            G1 = sp.kron(speye(self.nCy), ddxCellGrad(self.nCx, BC[0]))
            G2 = sp.kron(ddxCellGrad(self.nCy, BC[1]), speye(self.nCx))
            G = sp.vstack((G1, G2), format="csr")
        elif self.dim == 3:
            G1 = kron3(speye(self.nCz), speye(self.nCy), ddxCellGrad(self.nCx, BC[0]))
            G2 = kron3(speye(self.nCz), ddxCellGrad(self.nCy, BC[1]), speye(self.nCx))
            G3 = kron3(ddxCellGrad(self.nCz, BC[2]), speye(self.nCy), speye(self.nCx))
            G = sp.vstack((G1, G2, G3), format="csr")
        return G

    @property
    def cellGrad(self):
        """
        The cell centered Gradient, takes you to cell faces.
        """
        if getattr(self, '_cellGrad', None) is None:
            G = self._cellGradStencil
            S = self.area  # Compute areas of cell faces & volumes
            V = self.aveCC2F*self.vol  # Average volume between adjacent cells
            self._cellGrad = sdiag(S/V)*G
        return self._cellGrad

    @property
    def cellGradBC(self):
        """
        The cell centered Gradient boundary condition matrix
        """
        if getattr(self, '_cellGradBC', None) is None:
            BC = self.setCellGradBC(self._cellGradBC_list)
            n = self.vnC
            if self.dim == 1:
                G = ddxCellGradBC(n[0], BC[0])
            elif self.dim == 2:
                G1 = sp.kron(speye(n[1]), ddxCellGradBC(n[0], BC[0]))
                G2 = sp.kron(ddxCellGradBC(n[1], BC[1]), speye(n[0]))
                G = sp.block_diag((G1, G2), format="csr")
            elif self.dim == 3:
                G1 = kron3(speye(n[2]), speye(n[1]), ddxCellGradBC(n[0], BC[0]))
                G2 = kron3(speye(n[2]), ddxCellGradBC(n[1], BC[1]), speye(n[0]))
                G3 = kron3(ddxCellGradBC(n[2], BC[2]), speye(n[1]), speye(n[0]))
                G = sp.block_diag((G1, G2, G3), format="csr")
            # Compute areas of cell faces & volumes
            S = self.area
            V = self.aveCC2F*self.vol  # Average volume between adjacent cells
            self._cellGradBC = sdiag(S/V)*G
        return self._cellGradBC

    @property
    def cellGradx(self):
        """
        Cell centered Gradient in the x dimension. Has neumann boundary
        conditions.
        """
        if getattr(self, '_cellGradx', None) is None:
            G1 = self._cellGradxStencil
            # Compute areas of cell faces & volumes
            V = self.aveCC2F*self.vol
            L = self.r(self.area/V, 'F', 'Fx', 'V')
            self._cellGradx = sdiag(L)*G1
        return self._cellGradx

    @property
    def cellGrady(self):
        if self.dim < 2:
            return None
        if getattr(self, '_cellGrady', None) is None:
            G2 = self._cellGradyStencil
            # Compute areas of cell faces & volumes
            V = self.aveCC2F*self.vol
            L = self.r(self.area/V, 'F', 'Fy', 'V')
            self._cellGrady = sdiag(L)*G2
        return self._cellGrady

    @property
    def cellGradz(self):
        """
        Cell centered Gradient in the x dimension. Has neumann boundary
        conditions.
        """
        if self.dim < 3:
            return None
        if getattr(self, '_cellGradz', None) is None:
            G3 = self._cellGradzStencil
            # Compute areas of cell faces & volumes
            V = self.aveCC2F*self.vol
            L = self.r(self.area/V, 'F', 'Fz', 'V')
            self._cellGradz = sdiag(L)*G3
        return self._cellGradz

    ###########################################################################
    #                                                                         #
    #                                Edge Curl                                #
    #                                                                         #
    ###########################################################################

    @property
    def _edgeCurlStencilx(self):
        n = self.vnC  # The number of cell centers in each direction

        D32 = kron3(ddx(n[2]), speye(n[1]), speye(n[0]+1))
        D23 = kron3(speye(n[2]), ddx(n[1]), speye(n[0]+1))
        # O1 = spzeros(np.shape(D32)[0], np.shape(D31)[1])
        O1 = spzeros((n[0]+1)*n[1]*n[2], n[0]*(n[1]+1)*(n[2]+1))

        return sp.hstack((O1, -D32, D23))

    @property
    def _edgeCurlStencily(self):
        n = self.vnC  # The number of cell centers in each direction

        D31 = kron3(ddx(n[2]), speye(n[1]+1), speye(n[0]))
        D13 = kron3(speye(n[2]), speye(n[1]+1), ddx(n[0]))
        # O2 = spzeros(np.shape(D31)[0], np.shape(D32)[1])
        O2 = spzeros(n[0]*(n[1]+1)*n[2], (n[0]+1)*n[1]*(n[2]+1))

        return sp.hstack((D31, O2, -D13))

    @property
    def _edgeCurlStencilz(self):
        n = self.vnC  # The number of cell centers in each direction

        D21 = kron3(speye(n[2]+1), ddx(n[1]), speye(n[0]))
        D12 = kron3(speye(n[2]+1), speye(n[1]), ddx(n[0]))
        # O3 = spzeros(np.shape(D21)[0], np.shape(D13)[1])
        O3 = spzeros(n[0]*n[1]*(n[2]+1), (n[0]+1)*(n[1]+1)*n[2])

        return sp.hstack((-D21, D12, O3))

    @property
    def _edgeCurlStencil(self):
        assert self.dim > 1, "Edge Curl only programed for 2 or 3D."

        # Compute divergence operator on faces
        if self.dim == 2:
            n = self.vnC  # The number of cell centers in each direction

            D21 = sp.kron(ddx(n[1]), speye(n[0]))
            D12 = sp.kron(speye(n[1]), ddx(n[0]))
            C = sp.hstack((-D21, D12), format="csr")
            return C

        elif self.dim == 3:

            # D32 = kron3(ddx(n[2]), speye(n[1]), speye(n[0]+1))
            # D23 = kron3(speye(n[2]), ddx(n[1]), speye(n[0]+1))
            # D31 = kron3(ddx(n[2]), speye(n[1]+1), speye(n[0]))
            # D13 = kron3(speye(n[2]), speye(n[1]+1), ddx(n[0]))
            # D21 = kron3(speye(n[2]+1), ddx(n[1]), speye(n[0]))
            # D12 = kron3(speye(n[2]+1), speye(n[1]), ddx(n[0]))

            # O1 = spzeros(np.shape(D32)[0], np.shape(D31)[1])
            # O2 = spzeros(np.shape(D31)[0], np.shape(D32)[1])
            # O3 = spzeros(np.shape(D21)[0], np.shape(D13)[1])

            # C = sp.vstack((sp.hstack((O1, -D32, D23)),
            #                sp.hstack((D31, O2, -D13)),
            #                sp.hstack((-D21, D12, O3))), format="csr")

            C = sp.vstack((
                self._edgeCurlStencilx,
                self._edgeCurlStencily,
                self._edgeCurlStencilz
            ), format="csr")

            return C

    @property
    def edgeCurl(self):
        """
        Construct the 3D curl operator.
        """
        L = self.edge  # Compute lengths of cell edges
        S = self.area # Compute areas of cell faces

        if getattr(self, '_edgeCurl', None) is None:

            assert self.dim > 1, "Edge Curl only programed for 2 or 3D."

            if self.dim == 2:
                self._edgeCurl = self._edgeCurlStencil*sdiag(1/S)
            elif self.dim == 3:
                self._edgeCurl = sdiag(1/S)*(self._edgeCurlStencil*sdiag(L))

        return self._edgeCurl

    def getBCProjWF(self, BC, discretization='CC'):
        """
        The weak form boundary condition projection matrices.

        Example
        -------

        .. code:: python

            # Neumann in all directions
            BC = 'neumann'

            # 3D, Dirichlet in y Neumann else
            BC = ['neumann', 'dirichlet', 'neumann']

            # 3D, Neumann in x on bottom of domain, Dirichlet else
            BC = [['neumann', 'dirichlet'], 'dirichlet', 'dirichlet']
        """

        if discretization is not 'CC':
            raise NotImplementedError(
                'Boundary conditions only implemented'
                'for CC discretization.'
            )

        if isinstance(BC, string_types):
            BC = [BC for _ in self.vnC]  # Repeat the str self.dim times
        elif isinstance(BC, list):
            assert len(BC) == self.dim, 'BC list must be the size of your mesh'
        else:
            raise Exception("BC must be a str or a list.")

        for i, bc_i in enumerate(BC):
            BC[i] = checkBC(bc_i)

        def projDirichlet(n, bc):
            bc = checkBC(bc)
            ij = ([0, n], [0, 1])
            vals = [0, 0]
            if(bc[0] == 'dirichlet'):
                vals[0] = -1
            if(bc[1] == 'dirichlet'):
                vals[1] = 1
            return sp.csr_matrix((vals, ij), shape=(n+1, 2))

        def projNeumannIn(n, bc):
            bc = checkBC(bc)
            P = sp.identity(n+1).tocsr()
            if(bc[0] == 'neumann'):
                P = P[1:, :]
            if(bc[1] == 'neumann'):
                P = P[:-1, :]
            return P

        def projNeumannOut(n, bc):
            bc = checkBC(bc)
            ij = ([0, 1], [0, n])
            vals = [0, 0]
            if(bc[0] == 'neumann'):
                vals[0] = 1
            if(bc[1] == 'neumann'):
                vals[1] = 1
            return sp.csr_matrix((vals, ij), shape=(2, n+1))

        n = self.vnC
        indF = self.faceBoundaryInd
        if self.dim == 1:
            Pbc = projDirichlet(n[0], BC[0])
            indF = indF[0] | indF[1]
            Pbc = Pbc*sdiag(self.area[indF])

            Pin = projNeumannIn(n[0], BC[0])

            Pout = projNeumannOut(n[0], BC[0])

        elif self.dim == 2:
            Pbc1 = sp.kron(speye(n[1]), projDirichlet(n[0], BC[0]))
            Pbc2 = sp.kron(projDirichlet(n[1], BC[1]), speye(n[0]))
            Pbc = sp.block_diag((Pbc1, Pbc2), format="csr")
            indF = np.r_[(indF[0] | indF[1]), (indF[2] | indF[3])]
            Pbc = Pbc*sdiag(self.area[indF])

            P1 = sp.kron(speye(n[1]), projNeumannIn(n[0], BC[0]))
            P2 = sp.kron(projNeumannIn(n[1], BC[1]), speye(n[0]))
            Pin = sp.block_diag((P1, P2), format="csr")

            P1 = sp.kron(speye(n[1]), projNeumannOut(n[0], BC[0]))
            P2 = sp.kron(projNeumannOut(n[1], BC[1]), speye(n[0]))
            Pout = sp.block_diag((P1, P2), format="csr")

        elif self.dim == 3:
            Pbc1 = kron3(speye(n[2]), speye(n[1]), projDirichlet(n[0], BC[0]))
            Pbc2 = kron3(speye(n[2]), projDirichlet(n[1], BC[1]), speye(n[0]))
            Pbc3 = kron3(projDirichlet(n[2], BC[2]), speye(n[1]), speye(n[0]))
            Pbc = sp.block_diag((Pbc1, Pbc2, Pbc3), format="csr")
            indF = np.r_[
                (indF[0] | indF[1]),
                (indF[2] | indF[3]),
                (indF[4] | indF[5])
            ]
            Pbc = Pbc*sdiag(self.area[indF])

            P1 = kron3(speye(n[2]), speye(n[1]), projNeumannIn(n[0], BC[0]))
            P2 = kron3(speye(n[2]), projNeumannIn(n[1], BC[1]), speye(n[0]))
            P3 = kron3(projNeumannIn(n[2], BC[2]), speye(n[1]), speye(n[0]))
            Pin = sp.block_diag((P1, P2, P3), format="csr")

            P1 = kron3(speye(n[2]), speye(n[1]), projNeumannOut(n[0], BC[0]))
            P2 = kron3(speye(n[2]), projNeumannOut(n[1], BC[1]), speye(n[0]))
            P3 = kron3(projNeumannOut(n[2], BC[2]), speye(n[1]), speye(n[0]))
            Pout = sp.block_diag((P1, P2, P3), format="csr")

        return Pbc, Pin, Pout

    def getBCProjWF_simple(self, discretization='CC'):
        """The weak form boundary condition projection matrices
        when mixed boundary condition is used
        """

        if discretization is not 'CC':
            raise NotImplementedError('Boundary conditions only implemented'
                                      'for CC discretization.')

        def projBC(n):
            ij = ([0, n], [0, 1])
            vals = [0, 0]
            vals[0] = 1
            vals[1] = 1
            return sp.csr_matrix((vals, ij), shape=(n+1, 2))

        def projDirichlet(n, bc):
            bc = checkBC(bc)
            ij = ([0, n], [0, 1])
            vals = [0, 0]
            if(bc[0] == 'dirichlet'):
                vals[0] = -1
            if(bc[1] == 'dirichlet'):
                vals[1] = 1
            return sp.csr_matrix((vals, ij), shape=(n+1, 2))

        BC = [['dirichlet', 'dirichlet'], ['dirichlet', 'dirichlet'],
              ['dirichlet', 'dirichlet']]
        n = self.vnC
        indF = self.faceBoundaryInd

        if self.dim == 1:
            Pbc = projDirichlet(n[0], BC[0])
            B = projBC(n[0])
            indF = indF[0] | indF[1]
            Pbc = Pbc*sdiag(self.area[indF])

        elif self.dim == 2:
            Pbc1 = sp.kron(speye(n[1]), projDirichlet(n[0], BC[0]))
            Pbc2 = sp.kron(projDirichlet(n[1], BC[1]), speye(n[0]))
            Pbc = sp.block_diag((Pbc1, Pbc2), format="csr")
            B1 = sp.kron(speye(n[1]), projBC(n[0]))
            B2 = sp.kron(projBC(n[1]), speye(n[0]))
            B = sp.block_diag((B1, B2), format="csr")
            indF = np.r_[(indF[0] | indF[1]), (indF[2] | indF[3])]
            Pbc = Pbc*sdiag(self.area[indF])

        elif self.dim == 3:
            Pbc1 = kron3(speye(n[2]), speye(n[1]), projDirichlet(n[0], BC[0]))
            Pbc2 = kron3(speye(n[2]), projDirichlet(n[1], BC[1]), speye(n[0]))
            Pbc3 = kron3(projDirichlet(n[2], BC[2]), speye(n[1]), speye(n[0]))
            Pbc = sp.block_diag((Pbc1, Pbc2, Pbc3), format="csr")
            B1 = kron3(speye(n[2]), speye(n[1]), projBC(n[0]))
            B2 = kron3(speye(n[2]), projBC(n[1]), speye(n[0]))
            B3 = kron3(projBC(n[2]), speye(n[1]), speye(n[0]))
            B = sp.block_diag((B1, B2, B3), format="csr")
            indF = np.r_[
                (indF[0] | indF[1]),
                (indF[2] | indF[3]),
                (indF[4] | indF[5])
            ]
            Pbc = Pbc*sdiag(self.area[indF])

        return Pbc, B.T

    ###########################################################################
    #                                                                         #
    #                                Averaging                                #
    #                                                                         #
    ###########################################################################

    @property
    def aveF2CC(self):
        "Construct the averaging operator on cell faces to cell centers."
        if getattr(self, '_aveF2CC', None) is None:
            if self.dim == 1:
                self._aveF2CC = self.aveFx2CC
            elif self.dim == 2:
                self._aveF2CC = (0.5)*sp.hstack((
                    self.aveFx2CC, self.aveFy2CC
                ), format="csr")
            elif self.dim == 3:
                self._aveF2CC = (1./3.)*sp.hstack((
                    self.aveFx2CC, self.aveFy2CC, self.aveFz2CC
                ), format="csr")
        return self._aveF2CC

    @property
    def aveF2CCV(self):
        "Construct the averaging operator on cell faces to cell centers."
        if getattr(self, '_aveF2CCV', None) is None:
            if self.dim == 1:
                self._aveF2CCV = self.aveFx2CC
            elif self.dim == 2:
                self._aveF2CCV = sp.block_diag((
                    self.aveFx2CC, self.aveFy2CC
                ), format="csr")
            elif self.dim == 3:
                self._aveF2CCV = sp.block_diag((
                    self.aveFx2CC, self.aveFy2CC, self.aveFz2CC
                ), format="csr")
        return self._aveF2CCV

    @property
    def aveFx2CC(self):
        """
        Construct the averaging operator on cell faces in the x direction to
        cell centers.
        """

        if getattr(self, '_aveFx2CC', None) is None:
            n = self.vnC
            if self.dim == 1:
                self._aveFx2CC = av(n[0])
            elif self.dim == 2:
                self._aveFx2CC = sp.kron(speye(n[1]), av(n[0]))
            elif self.dim == 3:
                self._aveFx2CC = kron3(speye(n[2]), speye(n[1]), av(n[0]))
        return self._aveFx2CC

    @property
    def aveFy2CC(self):
        """
        Construct the averaging operator on cell faces in the y direction to
        cell centers.
        """
        if self.dim < 2:
            return None
        if getattr(self, '_aveFy2CC', None) is None:
            n = self.vnC
            if(self.dim == 2):
                self._aveFy2CC = sp.kron(av(n[1]), speye(n[0]))
            elif self.dim == 3:
                self._aveFy2CC = kron3(speye(n[2]), av(n[1]), speye(n[0]))
        return self._aveFy2CC

    @property
    def aveFz2CC(self):
        """
        Construct the averaging operator on cell faces in the z direction to
        cell centers.
        """
        if self.dim < 3:
            return None
        if getattr(self, '_aveFz2CC', None) is None:
            n = self.vnC
            if(self.dim == 3):
                self._aveFz2CC = kron3(av(n[2]), speye(n[1]), speye(n[0]))
        return self._aveFz2CC

    @property
    def aveCC2F(self):
        "Construct the averaging operator on cell centers to faces."
        if getattr(self, '_aveCC2F', None) is None:
            if self.dim == 1:
                self._aveCC2F = av_extrap(self.nCx)
            elif self.dim == 2:
                self._aveCC2F = sp.vstack((
                    sp.kron(speye(self.nCy), av_extrap(self.nCx)),
                    sp.kron(av_extrap(self.nCy), speye(self.nCx))
                ), format="csr")
            elif self.dim == 3:
                self._aveCC2F = sp.vstack((
                    kron3(
                        speye(self.nCz), speye(self.nCy), av_extrap(self.nCx)
                    ),
                    kron3(
                        speye(self.nCz), av_extrap(self.nCy), speye(self.nCx)
                    ),
                    kron3(
                        av_extrap(self.nCz), speye(self.nCy), speye(self.nCx)
                    )
                ), format="csr")
        return self._aveCC2F

    @property
    def aveCCV2F(self):
        """
        Construct the averaging operator on cell centers to
        faces as a vector.
        """
        if getattr(self, '_aveCCV2F', None) is None:
            if self.dim == 1:
                self._aveCCV2F = self.aveCC2F
            elif self.dim == 2:
                aveCCV2Fx = sp.kron(speye(self.nCy), av_extrap(self.nCx))
                aveCC2VFy = sp.kron(av_extrap(self.nCy), speye(self.nCx))
                self._aveCCV2F = sp.block_diag((
                    aveCCV2Fx, aveCC2VFy
                ), format="csr")
            elif self.dim == 3:
                aveCCV2Fx = kron3(
                    speye(self.nCz), speye(self.nCy), av_extrap(self.nCx)
                )
                aveCC2VFy = kron3(
                    speye(self.nCz), av_extrap(self.nCy), speye(self.nCx)
                )
                aveCC2BFz = kron3(
                    av_extrap(self.nCz), speye(self.nCy), speye(self.nCx)
                )
                self._aveCCV2F = sp.block_diag((
                        aveCCV2Fx, aveCC2VFy, aveCC2BFz
                ), format="csr")
        return self._aveCCV2F

    @property
    def aveE2CC(self):
        "Construct the averaging operator on cell edges to cell centers."
        if getattr(self, '_aveE2CC', None) is None:
            if self.dim == 1:
                self._avE2CC = self.aveEx2CC
            elif self.dim == 2:
                self._avE2CC = 0.5*sp.hstack(
                    (self.aveEx2CC, self.aveEy2CC), format="csr"
                )
            elif self.dim == 3:
                self._avE2CC = (1./3)*sp.hstack((
                    self.aveEx2CC, self.aveEy2CC, self.aveEz2CC
                ), format="csr")
        return self._avE2CC

    @property
    def aveE2CCV(self):
        "Construct the averaging operator on cell edges to cell centers."
        if getattr(self, '_aveE2CCV', None) is None:
            if self.dim == 1:
                self._aveE2CCV = self.aveEx2CC
            elif self.dim == 2:
                self._aveE2CCV = sp.block_diag(
                    (self.aveEx2CC, self.aveEy2CC), format="csr"
                )
            elif self.dim == 3:
                self._aveE2CCV = sp.block_diag(
                    (self.aveEx2CC, self.aveEy2CC, self.aveEz2CC), format="csr"
                )
        return self._aveE2CCV

    @property
    def aveEx2CC(self):
        """
        Construct the averaging operator on cell edges in the x direction to
        cell centers.
        """
        if getattr(self, '_aveEx2CC', None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            if self.dim == 1:
                self._aveEx2CC = speye(n[0])
            elif self.dim == 2:
                self._aveEx2CC = sp.kron(av(n[1]), speye(n[0]))
            elif self.dim == 3:
                self._aveEx2CC = kron3(av(n[2]), av(n[1]), speye(n[0]))
        return self._aveEx2CC

    @property
    def aveEy2CC(self):
        """
        Construct the averaging operator on cell edges in the y direction to
        cell centers.
        """
        if self.dim < 2:
            return None
        if getattr(self, '_aveEy2CC', None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            if(self.dim == 2):
                self._aveEy2CC = sp.kron(speye(n[1]), av(n[0]))
            elif self.dim == 3:
                self._aveEy2CC = kron3(av(n[2]), speye(n[1]), av(n[0]))
        return self._aveEy2CC

    @property
    def aveEz2CC(self):
        """
        Construct the averaging operator on cell edges in the z direction to
        cell centers.
        """
        if self.dim < 3:
            return None
        if getattr(self, '_aveEz2CC', None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            if(self.dim == 3):
                self._aveEz2CC = kron3(speye(n[2]), av(n[1]), av(n[0]))
        return self._aveEz2CC

    @property
    def aveN2CC(self):
        "Construct the averaging operator on cell nodes to cell centers."
        if getattr(self, '_aveN2CC', None) is None:
            # The number of cell centers in each direction
            if self.dim == 1:
                self._aveN2CC = av(self.nCx)
            elif self.dim == 2:
                self._aveN2CC = sp.kron(av(self.nCy), av(self.nCx)).tocsr()
            elif self.dim == 3:
                self._aveN2CC = kron3(av(self.nCz), av(self.nCy), av(self.nCx)).tocsr()
        return self._aveN2CC

    @property
    def _aveN2Ex(self):
        """
        Averaging operator on cell nodes to x-edges
        """
        if self.dim == 1:
            aveN2Ex = av(self.nCx)
        elif self.dim == 2:
            aveN2Ex = sp.kron(speye(self.nNy), av(self.nCx))
        elif self.dim == 3:
            aveN2Ex = kron3(speye(self.nNz), speye(self.nNy), av(self.nCx))
        return aveN2Ex

    @property
    def _aveN2Ey(self):
        """
        Averaging operator on cell nodes to y-edges
        """
        if self.dim == 1:
            return None
        elif self.dim == 2:
            aveN2Ey = sp.kron(av(self.nCy), speye(self.nNx))
        elif self.dim == 3:
            aveN2Ey = kron3(speye(self.nNz), av(self.nCy), speye(self.nNx))
        return aveN2Ey

    @property
    def _aveN2Ez(self):
        if self.dim == 1 or self.dim == 2:
            return None
        elif self.dim == 3:
            aveN2Ez = kron3(av(self.nCz), speye(self.nNy), speye(self.nNx))
        return aveN2Ez

    @property
    def aveN2E(self):
        """
        Construct the averaging operator on cell nodes to cell edges, keeping
        each dimension separate.
        """
        if getattr(self, '_aveN2E', None) is None:
            # The number of cell centers in each direction
            if self.dim == 1:
                self._aveN2E = self._aveN2Ex
            elif self.dim == 2:
                self._aveN2E = sp.vstack((
                    self._aveN2Ex, self._aveN2Ey
                ), format="csr")
            elif self.dim == 3:
                self._aveN2E = sp.vstack((
                    self._aveN2Ex, self._aveN2Ey, self._aveN2Ez
                ), format="csr")
        return self._aveN2E

    @property
    def _aveN2Fx(self):
        if self.dim == 1:
            aveN2Fx = av(self.nCx)
        elif self.dim == 2:
            aveN2Fx = sp.kron(av(self.nCy), speye(self.nNx))
        elif self.dim == 3:
            aveN2Fx = kron3(av(self.nCz), av(self.nCy), speye(self.nNx))
        return aveN2Fx

    @property
    def _aveN2Fy(self):
        if self.dim == 1:
            return None
        elif self.dim == 2:
            aveN2Fy = sp.kron(speye(self.nNy), av(self.nCx))
        elif self.dim == 3:
            aveN2Fy = kron3(av(self.nCz), speye(self.nNy), av(self.nCx))
        return aveN2Fy

    @property
    def _aveN2Fz(self):
        if self.dim == 1 or self.dim == 2:
            return None
        else:
            aveN2Fz = kron3(speye(self.nNz), av(self.nCy), av(self.nCx))
        return aveN2Fz

    @property
    def aveN2F(self):
        """
        Construct the averaging operator on cell nodes to cell faces, keeping
        each dimension separate.
        """
        if getattr(self, '_aveN2F', None) is None:
            # The number of cell centers in each direction
            if self.dim == 1:
                self._aveN2F = self._aveN2Fx
            elif self.dim == 2:
                self._aveN2F = sp.vstack((
                    self._aveN2Fx, self._aveN2Fy
                ), format="csr")
            elif self.dim == 3:
                self._aveN2F = sp.vstack((
                    self._aveN2Fx, self._aveN2Fy, self._aveN2Fz
                ), format="csr")
        return self._aveN2F
