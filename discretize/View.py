from __future__ import print_function
import numpy as np
import warnings
from discretize.utils import mkvc, ndgrid
from six import integer_types
try:
    import matplotlib.pyplot as plt
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print('Trouble importing matplotlib.')


class TensorView(object):
    """Provides viewing functions for TensorMesh

    This class is inherited by TensorMesh
    """
    def __init__(self):
        pass

    # def components(self):

    #     plotAll = len(imageType) == 1
    #     options = {"direction":direction, "numbering":numbering, "annotationColor":annotationColor, "showIt":False}
    #     fig = plt.figure(figNum)
    #     # Determine the subplot number: 131, 121
    #     numPlots = 130 if plotAll else len(imageType)//2*10+100
    #     pltNum = 1
    #     fxyz = self.r(I, 'F', 'F', 'M')
    #     if plotAll or 'Fx' in imageType:
    #         ax_x = plt.subplot(numPlots+pltNum)
    #         self.plotImage(fxyz[0], imageType='Fx', ax=ax_x, **options)
    #         pltNum +=1
    #     if plotAll or 'Fy' in imageType:
    #         ax_y = plt.subplot(numPlots+pltNum)
    #         self.plotImage(fxyz[1], imageType='Fy', ax=ax_y, **options)
    #         pltNum +=1
    #     if plotAll or 'Fz' in imageType:
    #         ax_z = plt.subplot(numPlots+pltNum)
    #         self.plotImage(fxyz[2], imageType='Fz', ax=ax_z, **options)
    #         pltNum +=1
    #     if showIt: plt.show()

    def plotImage(
        self, v, vType='CC', grid=False, view='real',
        ax=None, clim=None, showIt=False,
        pcolorOpts=None,
        streamOpts=None,
        gridOpts=None,
        numbering=True, annotationColor='w',
        range_x=None, range_y=None, sample_grid=None,
        stream_threshold=None
    ):
        """
        Mesh.plotImage(v)

        Plots scalar fields on the given mesh.

        Input:

        :param numpy.array v: vector

        Optional Inputs:

        :param str vType: type of vector ('CC', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez')
        :param matplotlib.axes.Axes ax: axis to plot to
        :param bool showIt: call plt.show()

        3D Inputs:

        :param bool numbering: show numbering of slices, 3D only
        :param str annotationColor: color of annotation, e.g. 'w', 'k', 'b'

        .. plot::
            :include-source:

            import discretize
            import numpy as np
            M = discretize.TensorMesh([20, 20])
            v = np.sin(M.gridCC[:, 0]*2*np.pi)*np.sin(M.gridCC[:, 1]*2*np.pi)
            M.plotImage(v, showIt=True)

        .. plot::
            :include-source:

            import discretize
            import numpy as np
            M = discretize.TensorMesh([20, 20, 20])
            v = np.sin(M.gridCC[:, 0]*2*np.pi)*np.sin(M.gridCC[:, 1]*2*np.pi)*np.sin(M.gridCC[:, 2]*2*np.pi)
            M.plotImage(v, annotationColor='k', showIt=True)

        """
        if pcolorOpts is None:
            pcolorOpts = {}
        if streamOpts is None:
            streamOpts = {'color': 'k'}
        if gridOpts is None:
            gridOpts = {'color': 'k'}

        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(111)
        else:
            if not isinstance(ax, matplotlib.axes.Axes):
                raise AssertionError("ax must be an Axes!")
            fig = ax.figure

        if self.dim == 1:
            if vType == 'CC':
                ph = ax.plot(
                    self.vectorCCx, v, linestyle="-", color="C1", marker="o"
                )
            elif vType == 'N':
                ph = ax.plot(
                    self.vectorNx, v, linestyle="-", color="C0", marker="s"
                )
            ax.set_xlabel("x")
            ax.axis('tight')
        elif self.dim == 2:
            return self._plotImage2D(
                v, vType=vType, grid=grid, view=view,
                ax=ax, clim=clim, showIt=showIt,
                pcolorOpts=pcolorOpts, streamOpts=streamOpts,
                gridOpts=gridOpts, range_x=range_x, range_y=range_y,
                sample_grid=sample_grid, stream_threshold=stream_threshold
            )
        elif self.dim == 3:
            # get copy of image and average to cell-centers is necessary
            if vType == 'CC':
                vc = v.reshape(self.vnC, order='F')
            elif vType == 'N':
                vc = (self.aveN2CC*v).reshape(self.vnC, order='F')
            elif vType in ['Fx', 'Fy', 'Fz', 'Ex', 'Ey', 'Ez']:
                aveOp = 'ave' + vType[0] + '2CCV'
                # n = getattr(self, 'vn'+vType[0])
                # if 'x' in vType: v = np.r_[v, np.zeros(n[1]), np.zeros(n[2])]
                # if 'y' in vType: v = np.r_[np.zeros(n[0]), v, np.zeros(n[2])]
                # if 'z' in vType: v = np.r_[np.zeros(n[0]), np.zeros(n[1]), v]
                v = getattr(self, aveOp)*v # average to cell centers
                ind_xyz = {'x': 0, 'y': 1, 'z': 2}[vType[1]]
                vc = self.r(
                    v.reshape((self.nC, -1), order='F'), 'CC', 'CC', 'M'
                )[ind_xyz]

            # determine number oE slices in x and y dimension
            nX = int(np.ceil(np.sqrt(self.nCz)))
            nY = int(np.ceil(self.nCz/nX))

            #  allocate space for montage
            nCx = self.nCx
            nCy = self.nCy

            C = np.zeros((nX*nCx, nY*nCy))

            for iy in range(int(nY)):
                for ix in range(int(nX)):
                    iz = ix + iy*nX
                    if iz < self.nCz:
                        C[ix*nCx:(ix+1)*nCx, iy*nCy:(iy+1)*nCy] = vc[:, :, iz]
                    else:
                        C[ix*nCx:(ix+1)*nCx, iy*nCy:(iy+1)*nCy] = np.nan

            C = np.ma.masked_where(np.isnan(C), C)
            xx = np.r_[0, np.cumsum(np.kron(np.ones((nX, 1)), self.hx).ravel())]
            yy = np.r_[0, np.cumsum(np.kron(np.ones((nY, 1)), self.hy).ravel())]
            # Plot the mesh

            if clim is None:
                clim = [C.min(), C.max()]
            ph = ax.pcolormesh(xx, yy, C.T, vmin=clim[0], vmax=clim[1])
            # Plot the lines
            gx =  np.arange(nX+1)*(self.vectorNx[-1]-self.x0[0])
            gy =  np.arange(nY+1)*(self.vectorNy[-1]-self.x0[1])
            # Repeat and seperate with NaN
            gxX = np.c_[gx, gx, gx+np.nan].ravel()
            gxY = np.kron(np.ones((nX+1, 1)), np.array([0, sum(self.hy)*nY, np.nan])).ravel()
            gyX = np.kron(np.ones((nY+1, 1)), np.array([0, sum(self.hx)*nX, np.nan])).ravel()
            gyY = np.c_[gy, gy, gy+np.nan].ravel()
            ax.plot(gxX, gxY, annotationColor+'-', linewidth=2)
            ax.plot(gyX, gyY, annotationColor+'-', linewidth=2)
            ax.axis('tight')

            if numbering:
                pad = np.sum(self.hx)*0.04
                for iy in range(int(nY)):
                    for ix in range(int(nX)):
                        iz = ix + iy*nX
                        if iz < self.nCz:
                            ax.text((ix+1)*(self.vectorNx[-1]-self.x0[0])-pad, (iy)*(self.vectorNy[-1]-self.x0[1])+pad,
                                     '#{0:.0f}'.format(iz), color=annotationColor, verticalalignment='bottom', horizontalalignment='right', size='x-large')

        ax.set_title(vType)
        if showIt:
            plt.show()
        return ph

    def plotSlice(
        self, v, vType='CC',
        normal='Z', ind=None, grid=False, view='real',
        ax=None, clim=None, showIt=False,
        pcolorOpts=None,
        streamOpts=None,
        gridOpts=None,
        range_x=None,
        range_y=None,
        sample_grid=None,
        stream_threshold=None
    ):

        """
        Plots a slice of a 3D mesh.

        .. plot::

            import discretize
            from pymatsolver import Solver
            import numpy as np
            hx = [(5, 2, -1.3), (2, 4), (5, 2, 1.3)]
            hy = [(2, 2, -1.3), (2, 6), (2, 2, 1.3)]
            hz = [(2, 2, -1.3), (2, 6), (2, 2, 1.3)]
            M = discretize.TensorMesh([hx, hy, hz])
            q = np.zeros(M.vnC)
            q[[4, 4], [4, 4], [2, 6]]=[-1, 1]
            q = discretize.utils.mkvc(q)
            A = M.faceDiv * M.cellGrad
            b = Solver(A) * (q)
            M.plotSlice(M.cellGrad*b, 'F', view='vec', grid=True, showIt=True, pcolorOpts={'alpha':0.8})

        """
        normal = normal.upper()
        if pcolorOpts is None:
            pcolorOpts = {}
        if streamOpts is None:
            streamOpts = {'color':'k'}
        if gridOpts is None:
            gridOpts = {'color':'k', 'alpha':0.5}
        if type(vType) in [list, tuple]:
            if ax is not None:
                raise AssertionError(
                    "cannot specify an axis to plot on with this function."
                )
            fig, axs = plt.subplots(1, len(vType))
            out = []
            for vTypeI, ax in zip(vType, axs):
                out += [
                    self.plotSlice(
                        v, vType=vTypeI, normal=normal, ind=ind, grid=grid,
                        view=view, ax=ax, clim=clim, showIt=False,
                        pcolorOpts=pcolorOpts, streamOpts=streamOpts,
                        gridOpts=gridOpts, stream_threshold=stream_threshold
                    )
                ]
            return out
        viewOpts = ['real', 'imag', 'abs', 'vec']
        normalOpts = ['X', 'Y', 'Z']
        vTypeOpts = ['CC', 'CCv', 'N', 'F', 'E', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez']

        # Some user error checking
        if vType not in vTypeOpts:
            raise AssertionError(
                "vType must be in ['{0!s}']".format("', '".join(vTypeOpts))
            )
        if not self.dim == 3:
            raise AssertionError(
                'Must be a 3D mesh. Use plotImage.'
            )
        if view not in viewOpts:
            raise AssertionError(
                "view must be in ['{0!s}']".format("', '".join(viewOpts))
            )
        if normal not in normalOpts:
            raise AssertionError(
                "normal must be in ['{0!s}']".format("', '".join(normalOpts))
            )
        if not isinstance(grid, bool):
            raise AssertionError('grid must be a boolean')

        szSliceDim = getattr(self, 'nC'+normal.lower()) #: Size of the sliced dimension
        if ind is None: ind = int(szSliceDim/2)
        if type(ind) not in integer_types:
            raise AssertionError('ind must be an integer')

        if (v.dtype == complex and view == 'vec'):
            raise AssertionError('Can not plot a complex vector.')
        # The slicing and plotting code!!

        def getIndSlice(v):
            if normal == 'X':
                v = v[ind, :, :]
            elif normal == 'Y':
                v = v[:, ind, :]
            elif normal == 'Z':
                v = v[:, :, ind]
            return v

        def doSlice(v):
            if vType == 'CC':
                return getIndSlice(self.r(v, 'CC', 'CC', 'M'))
            elif vType == 'CCv':
                if view != 'vec':
                    raise AssertionError('Other types for CCv not supported')
            else:
                # Now just deal with 'F' and 'E' (x, y, z, maybe...)
                aveOp = 'ave' + vType + ('2CCV' if view == 'vec' else '2CC')
                Av = getattr(self, aveOp)
                if v.size == Av.shape[1]:
                    v = Av * v
                else:
                    v = self.r(v, vType[0], vType) # get specific component
                    v = Av * v
                # we should now be averaged to cell centers (might be a vector)
            v = self.r(v.reshape((self.nC, -1), order='F'), 'CC', 'CC', 'M')
            if view == 'vec':
                outSlice = []
                if 'X' not in normal: outSlice.append(getIndSlice(v[0]))
                if 'Y' not in normal: outSlice.append(getIndSlice(v[1]))
                if 'Z' not in normal: outSlice.append(getIndSlice(v[2]))
                return np.r_[mkvc(outSlice[0]), mkvc(outSlice[1])]
            else:
                return getIndSlice(self.r(v, 'CC', 'CC', 'M'))

        h2d = []
        x2d = []
        if 'X' not in normal:
            h2d.append(self.hx)
            x2d.append(self.x0[0])
        if 'Y' not in normal:
            h2d.append(self.hy)
            x2d.append(self.x0[1])
        if 'Z' not in normal:
            h2d.append(self.hz)
            x2d.append(self.x0[2])
        tM = self.__class__(h=h2d, x0=x2d)  #: Temp Mesh
        v2d = doSlice(v)

        if ax is None:
            plt.figure()
            ax = plt.subplot(111)
        else:
            if not isinstance(ax, matplotlib.axes.Axes):
                raise AssertionError("ax must be an matplotlib.axes.Axes")

        out = tM._plotImage2D(
            v2d, vType=('CCv' if view == 'vec' else 'CC'),
            grid=grid, view=view,
            ax=ax, clim=clim, showIt=showIt,
            pcolorOpts=pcolorOpts, streamOpts=streamOpts,
            gridOpts=gridOpts,
            range_x=range_x,
            range_y=range_y,
            sample_grid=sample_grid,
            stream_threshold=stream_threshold
        )

        ax.set_xlabel('y' if normal == 'X' else 'x')
        ax.set_ylabel('y' if normal == 'Z' else 'z')
        ax.set_title('Slice {0:.0f}'.format(ind))
        return out

    def _plotImage2D(
        self, v, vType='CC', grid=False, view='real',
        ax=None, clim=None, showIt=False,
        pcolorOpts=None,
        streamOpts=None,
        gridOpts=None,
        range_x=None,
        range_y=None,
        sample_grid=None,
        stream_threshold=None
    ):

        if pcolorOpts is None:
            pcolorOpts = {}
        if streamOpts is None:
            streamOpts = {'color': 'k'}
        if gridOpts is None:
            gridOpts = {'color': 'k'}
        vTypeOptsCC = ['N', 'CC', 'Fx', 'Fy', 'Ex', 'Ey']
        vTypeOptsV = ['CCv', 'F', 'E']
        vTypeOpts = vTypeOptsCC + vTypeOptsV
        if view == 'vec':
            if vType not in vTypeOptsV:
                raise AssertionError(
                    "vType must be in ['{0!s}'] when view='vec'".format(
                        "', '".join(vTypeOptsV)
                    )
                )
        if vType not in vTypeOpts:
            raise AssertionError(
                "vType must be in ['{0!s}']".format("', '".join(vTypeOpts))
            )

        viewOpts = ['real', 'imag', 'abs', 'vec']
        if view not in viewOpts:
            raise AssertionError(
                "view must be in ['{0!s}']".format("', '".join(viewOpts))
            )

        if ax is None:
            plt.figure()
            ax = plt.subplot(111)
        else:
            if not isinstance(ax, matplotlib.axes.Axes):
                raise AssertionError(
                    "ax must be an matplotlib.axes.Axes"
                )

        # Reshape to a cell centered variable
        if vType == 'CC':
            pass
        elif vType == 'CCv':
            if view != 'vec':
                raise AssertionError('Other types for CCv not supported')
        elif vType in ['F', 'E', 'N']:
            aveOp = 'ave' + vType + ('2CCV' if view == 'vec' else '2CC')
            v = getattr(self, aveOp)*v  # average to cell centers (might be a vector)
        elif vType in ['Fx', 'Fy', 'Ex', 'Ey']:
            aveOp = 'ave' + vType[0] + '2CCV'
            v = getattr(self, aveOp)*v  # average to cell centers (might be a vector)
            xORy = {'x': 0, 'y':1 }[vType[1]]
            v = v.reshape((self.nC, -1), order='F')[:, xORy]

        out = ()
        if view in ['real', 'imag', 'abs']:
            v = self.r(v, 'CC', 'CC', 'M')
            v = getattr(np, view)(v) # e.g. np.real(v)
            if clim is None:
                clim = [v.min(), v.max()]
            v = np.ma.masked_where(np.isnan(v), v)
            out += (ax.pcolormesh(self.vectorNx, self.vectorNy, v.T, vmin=clim[0], vmax=clim[1], **pcolorOpts), )
        elif view in ['vec']:
            # Matplotlib seems to not support irregular
            # spaced vectors at the moment. So we will
            # Interpolate down to a regular mesh at the
            # smallest mesh size in this 2D slice.
            if sample_grid is not None:
                hxmin = sample_grid[0]
                hymin = sample_grid[1]
            else:
                hxmin = self.hx.min()
                hymin = self.hy.min()

            if range_x is not None:
                dx = (range_x[1] - range_x[0])
                nxi = int(dx/hxmin)
                hx = np.ones(nxi)*dx/nxi
                x0_x = range_x[0]
            else:
                nxi = int(self.hx.sum()/hxmin)
                hx = np.ones(nxi)*self.hx.sum()/nxi
                x0_x = self.x0[0]

            if range_y is not None:
                dy = (range_y[1] - range_y[0])
                nyi = int(dy/hymin)
                hy = np.ones(nyi)*dy/nyi
                x0_y = range_y[0]
            else:
                nyi = int(self.hy.sum()/hymin)
                hy = np.ones(nyi)*self.hy.sum()/nyi
                x0_y = self.x0[1]

            U, V = self.r(v.reshape((self.nC, -1), order='F'), 'CC', 'CC', 'M')
            if clim is None:
                uv = np.sqrt(U**2 + V**2)
                clim = [uv.min(), uv.max()]

            tMi = self.__class__(h=[hx, hy], x0=np.r_[x0_x, x0_y])
            P = self.getInterpolationMat(tMi.gridCC, 'CC', zerosOutside=True)

            Ui = tMi.r(P*mkvc(U), 'CC', 'CC', 'M')
            Vi = tMi.r(P*mkvc(V), 'CC', 'CC', 'M')
            # End Interpolation

            x = self.vectorNx
            y = self.vectorNy

            ind_CCx = np.ones(self.vnC, dtype=bool)
            ind_CCy = np.ones(self.vnC, dtype=bool)
            if range_x is not None:
                x = tMi.vectorNx

            if range_y is not None:
                y = tMi.vectorNy

            if range_x is not None or range_y is not None:  # use interpolated values
                U = Ui
                V = Vi

            if stream_threshold is not None:
                mask_me = np.sqrt(Ui**2 + Vi**2) <= stream_threshold
                Ui = np.ma.masked_where(mask_me, Ui)
                Vi = np.ma.masked_where(mask_me, Vi)

            out += (
                ax.pcolormesh(
                    x, y, np.sqrt(U**2+V**2).T, vmin=clim[0], vmax=clim[1],
                    **pcolorOpts),
            )
            out += (
                ax.streamplot(
                    tMi.vectorCCx, tMi.vectorCCy, Ui.T, Vi.T, **streamOpts
                ),
            )

        if grid:
            xXGrid = np.c_[self.vectorNx, self.vectorNx, np.nan*np.ones(self.nNx)].flatten()
            xYGrid = np.c_[self.vectorNy[0]*np.ones(self.nNx), self.vectorNy[-1]*np.ones(self.nNx), np.nan*np.ones(self.nNx)].flatten()
            yXGrid = np.c_[self.vectorNx[0]*np.ones(self.nNy), self.vectorNx[-1]*np.ones(self.nNy), np.nan*np.ones(self.nNy)].flatten()
            yYGrid = np.c_[self.vectorNy, self.vectorNy, np.nan*np.ones(self.nNy)].flatten()
            out += (ax.plot(np.r_[xXGrid, yXGrid], np.r_[xYGrid, yYGrid], **gridOpts)[0], )

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        if range_x is not None:
            ax.set_xlim(*range_x)
        else:
            ax.set_xlim(*self.vectorNx[[0, -1]])

        if range_y is not None:
            ax.set_ylim(*range_y)
        else:
            ax.set_ylim(*self.vectorNy[[0, -1]])

        if showIt:
            plt.show()
        return out

    def plotGrid(
        self, ax=None, nodes=False, faces=False, centers=False, edges=False,
        lines=True, showIt=False
    ):
        """Plot the nodal, cell-centered and staggered grids for 1,2 and 3 dimensions.

        :param bool nodes: plot nodes
        :param bool faces: plot faces
        :param bool centers: plot centers
        :param bool edges: plot edges
        :param bool lines: plot lines connecting nodes
        :param bool showIt: call plt.show()

        .. plot::
           :include-source:

           import discretize
           import numpy as np
           h1 = np.linspace(.1, .5, 3)
           h2 = np.linspace(.1, .5, 5)
           mesh = discretize.TensorMesh([h1, h2])
           mesh.plotGrid(nodes=True, faces=True, centers=True, lines=True, showIt=True)

        .. plot::
           :include-source:

           import discretize
           import numpy as np
           h1 = np.linspace(.1, .5, 3)
           h2 = np.linspace(.1, .5, 5)
           h3 = np.linspace(.1, .5, 3)
           mesh = discretize.TensorMesh([h1, h2, h3])
           mesh.plotGrid(nodes=True, faces=True, centers=True, lines=True, showIt=True)

        """

        axOpts = {'projection': '3d'} if self.dim == 3 else {}
        if ax is None:
            plt.figure()
            ax = plt.subplot(111, **axOpts)
        else:
            if not isinstance(ax, matplotlib.axes.Axes):
                raise AssertionError("ax must be an matplotlib.axes.Axes")

        if self.dim == 1:
            if nodes:
                ax.plot(
                    self.gridN, np.ones(self.nN), color="C0", marker="s",
                    linestyle=""
                )
            if centers:
                ax.plot(
                    self.gridCC, np.ones(self.nC), color="C1", marker="o",
                    linestyle=""
                )
            if lines:
                ax.plot(
                    self.gridN, np.ones(self.nN), color="C0", linestyle=".-"
                )
            ax.set_xlabel('x1')
        elif self.dim == 2:
            if nodes:
                ax.plot(
                    self.gridN[:, 0], self.gridN[:, 1], color="C0", marker="s",
                    linestyle=""
                )
            if centers:
                ax.plot(
                    self.gridCC[:, 0], self.gridCC[:, 1], color="C1",
                    marker="o", linestyle=""
                )
            if faces:
                ax.plot(
                    self.gridFx[:, 0], self.gridFx[:, 1], color="C2",
                    marker=">", linestyle=""
                )
                ax.plot(
                    self.gridFy[:, 0], self.gridFy[:, 1], color="C2",
                    marker="^", linestyle=""
                )
            if edges:
                ax.plot(
                    self.gridEx[:, 0], self.gridEx[:, 1], color="C3",
                    marker=">", linestyle=""
                )
                ax.plot(
                    self.gridEy[:, 0], self.gridEy[:, 1], color="C3",
                    marker="^", linestyle=""
                )

            # Plot the grid lines
            if lines:
                NN = self.r(self.gridN, 'N', 'N', 'M')
                X1 = np.c_[mkvc(NN[0][0, :]), mkvc(NN[0][self.nCx, :]), mkvc(NN[0][0, :])*np.nan].flatten()
                Y1 = np.c_[mkvc(NN[1][0, :]), mkvc(NN[1][self.nCx, :]), mkvc(NN[1][0, :])*np.nan].flatten()
                X2 = np.c_[mkvc(NN[0][:, 0]), mkvc(NN[0][:, self.nCy]), mkvc(NN[0][:, 0])*np.nan].flatten()
                Y2 = np.c_[mkvc(NN[1][:, 0]), mkvc(NN[1][:, self.nCy]), mkvc(NN[1][:, 0])*np.nan].flatten()
                X = np.r_[X1, X2]
                Y = np.r_[Y1, Y2]
                ax.plot(X, Y, color="C0", linestyle="-")

            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
        elif self.dim == 3:
            if nodes:
                ax.plot(
                    self.gridN[:, 0], self.gridN[:, 1], color="C0", marker="s",
                    linestyle="", zs=self.gridN[:, 2]
                )
            if centers:
                ax.plot(
                    self.gridCC[:, 0], self.gridCC[:, 1], color="C1",
                    marker="o", linestyle="", zs=self.gridCC[:, 2]
                )
            if faces:
                ax.plot(
                    self.gridFx[:, 0], self.gridFx[:, 1], color="C2",
                    marker=">", linestyle="", zs=self.gridFx[:, 2]
                )
                ax.plot(
                    self.gridFy[:, 0], self.gridFy[:, 1], color="C2",
                    marker="<", linestyle="", zs=self.gridFy[:, 2]
                )
                ax.plot(
                    self.gridFz[:, 0], self.gridFz[:, 1], color="C2",
                    marker="^", linestyle="", zs=self.gridFz[:, 2]
                )
            if edges:
                ax.plot(
                    self.gridEx[:, 0], self.gridEx[:, 1], color="C3",
                    marker=">", linestyle="", zs=self.gridEx[:, 2]
                )
                ax.plot(
                    self.gridEy[:, 0], self.gridEy[:, 1], color="C3",
                    marker="<", linestyle="", zs=self.gridEy[:, 2]
                )
                ax.plot(
                    self.gridEz[:, 0], self.gridEz[:, 1], color="C3",
                    marker="^", linestyle="", zs=self.gridEz[:, 2]
                )

            # Plot the grid lines
            if lines:
                NN = self.r(self.gridN, 'N', 'N', 'M')
                X1 = np.c_[mkvc(NN[0][0, :, :]), mkvc(NN[0][self.nCx, :, :]), mkvc(NN[0][0, :, :])*np.nan].flatten()
                Y1 = np.c_[mkvc(NN[1][0, :, :]), mkvc(NN[1][self.nCx, :, :]), mkvc(NN[1][0, :, :])*np.nan].flatten()
                Z1 = np.c_[mkvc(NN[2][0, :, :]), mkvc(NN[2][self.nCx, :, :]), mkvc(NN[2][0, :, :])*np.nan].flatten()
                X2 = np.c_[mkvc(NN[0][:, 0, :]), mkvc(NN[0][:, self.nCy, :]), mkvc(NN[0][:, 0, :])*np.nan].flatten()
                Y2 = np.c_[mkvc(NN[1][:, 0, :]), mkvc(NN[1][:, self.nCy, :]), mkvc(NN[1][:, 0, :])*np.nan].flatten()
                Z2 = np.c_[mkvc(NN[2][:, 0, :]), mkvc(NN[2][:, self.nCy, :]), mkvc(NN[2][:, 0, :])*np.nan].flatten()
                X3 = np.c_[mkvc(NN[0][:, :, 0]), mkvc(NN[0][:, :, self.nCz]), mkvc(NN[0][:, :, 0])*np.nan].flatten()
                Y3 = np.c_[mkvc(NN[1][:, :, 0]), mkvc(NN[1][:, :, self.nCz]), mkvc(NN[1][:, :, 0])*np.nan].flatten()
                Z3 = np.c_[mkvc(NN[2][:, :, 0]), mkvc(NN[2][:, :, self.nCz]), mkvc(NN[2][:, :, 0])*np.nan].flatten()
                X = np.r_[X1, X2, X3]
                Y = np.r_[Y1, Y2, Y3]
                Z = np.r_[Z1, Z2, Z3]
                ax.plot(X, Y, color="C0", linestyle="-", zs=Z)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('x3')

        ax.grid(True)
        if showIt:
            plt.show()

        return ax


class CylView(object):

    def _plotCylTensorMesh(self, plotType, *args, **kwargs):

        if not self.isSymmetric:
            raise Exception('We have not yet implemented this type of view.')
        assert plotType in ['plotImage', 'plotGrid']
        # Hackity Hack:
        # Just create a TM and use its view.
        from discretize import TensorMesh

        if len(args) > 0:
            val = args[0]

        vType = kwargs.get('vType', None)
        mirror = kwargs.pop('mirror', None)
        mirror_data = kwargs.pop('mirror_data', None)

        if mirror_data is not None and mirror is None:
            mirror = True

        if vType is not None:
            if vType.upper() != 'CCV':
                if vType.upper() == 'F':
                    val = mkvc(self.aveF2CCV * val)
                    if mirror_data is not None:
                        mirror_data = mkvc(self.aveF2CCV * mirror_data)
                    kwargs['vType'] = 'CCv'  # now the vector is cell centered
                if vType.upper() == 'E':
                    val = mkvc(self.aveE2CCV * val)
                    if mirror_data is not None:
                        mirror_data = mkvc(self.aveE2CCV * mirror_data)
                args = (val,) + args[1:]

        if mirror is True:
            # create a mirrored mesh
            hx = np.hstack([np.flipud(self.hx), self.hx])
            x00 = self.x0[0] - self.hx.sum()
            M = TensorMesh([hx, self.hz], x0=[x00, self.x0[2]])

            if mirror_data is None:
                mirror_data = val

            if len(val) == self.nC:  # only a single value at cell centers
                val = val.reshape(self.vnC[0], self.vnC[2], order='F')
                mirror_val = mirror_data.reshape(
                    self.vnC[0], self.vnC[2], order='F'
                )
                val = mkvc(np.vstack([np.flipud(mirror_val), val]))

            elif len(val) == 2*self.nC:
                val_x = val[:self.nC].reshape(
                    self.vnC[0], self.vnC[2], order='F'
                )
                val_z = val[self.nC:].reshape(
                    self.vnC[0], self.vnC[2], order='F'
                )

                mirror_x = mirror_data[:self.nC].reshape(
                    self.vnC[0], self.vnC[2], order='F'
                )
                mirror_z = mirror_data[self.nC:].reshape(
                    self.vnC[0], self.vnC[2], order='F'
                )

                val_x = mkvc(np.vstack([-1.*np.flipud(mirror_x), val_x])) # by symmetry
                val_z = mkvc(np.vstack([np.flipud(mirror_z), val_z]))

                val = np.hstack([val_x, val_z])

            args = (val,) + args[1:]
        else:
            M = TensorMesh([self.hx, self.hz], x0=[self.x0[0], self.x0[2]])

        ax = kwargs.get('ax', None)
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(111)
            kwargs['ax'] = ax
        else:
            if not isinstance(ax, matplotlib.axes.Axes):
                raise AssertionError("ax must be an matplotlib.axes.Axes")
            fig = ax.figure

        # Don't show things in the TM.plotImage
        showIt = kwargs.get('showIt', False)
        kwargs['showIt'] = False

        out = getattr(M, plotType)(*args, **kwargs)

        ax.set_xlabel('x')
        ax.set_ylabel('z')

        if showIt:
            plt.show()

        return out

    def plotGrid(self, *args, **kwargs):
        if self.isSymmetric:
            return self._plotCylTensorMesh('plotGrid', *args, **kwargs)

        # allow a slice to be provided for the mesh
        slc = kwargs.pop('slice', None)
        if isinstance(slc, str):
            slc = slc.lower()
        if slc not in ['theta', 'z', 'both', None]:
            raise AssertionError(
                "slice must be either 'theta','z', or 'both' not {}".format(
                    slc
                )
            )

        # if slc is None, provide slices in both the theta and z directions
        if slc == 'theta':
            return self._plotGridThetaSlice(*args, **kwargs)
        elif slc == 'z':
            return self._plotGridZSlice(*args, **kwargs)
        else:
            ax = kwargs.pop('ax', None)
            if ax is not None:
                if not isinstance(ax, list) or len(ax) != 2:
                    warnings.warn(
                        "two axes handles must be provided to plot both theta "
                        "and z slices through the mesh. Over-writing the axes."
                    )
                    ax = None
                else:
                    # find the one with a polar projection and pass it to the
                    # theta slice, other one to the z-slice
                    polarax = [a for a in ax if a.__class__.__name__ == 'PolarAxesSubplot']
                    if len(polarax) != 1:
                        warnings.warn("""
No polar axes provided. Over-writing the axes. If you prefer to create your
own, please use

    `ax = plt.subplot(121, projection='polar')`

for reference, see: http://matplotlib.org/examples/pylab_examples/polar_demo.html
                    https://github.com/matplotlib/matplotlib/issues/312
                    """)
                        ax = None

                    else:
                        polarax = polarax[0]
                        cartax = [a for a in ax if a != polarax][0]

            # ax may have been None to start with or set to None
            if ax is None:
                fig = plt.figure(figsize=(12, 5))
                polarax = plt.subplot(121, projection='polar')
                cartax = plt.subplot(122)

            # update kwargs with respective axes handles
            kwargspolar = kwargs.copy()
            kwargspolar['ax'] = polarax

            kwargscart = kwargs.copy()
            kwargscart['ax'] = cartax

            ax = []
            ax.append(self._plotGridZSlice(*args, **kwargspolar))
            ax.append(self._plotGridThetaSlice(*args, **kwargscart))
            plt.tight_layout()

        return ax

    def _plotGridThetaSlice(self, *args, **kwargs):
        if self.isSymmetric:
            return self.plotGrid(*args, **kwargs)

        # make a cyl symmetric mesh
        h2d = [self.hx, 1, self.hz]
        mesh2D = self.__class__(h=h2d, x0=self.x0)
        return mesh2D.plotGrid(*args, **kwargs)

    def _plotGridZSlice(self, *args, **kwargs):
        # https://github.com/matplotlib/matplotlib/issues/312
        ax = kwargs.get('ax', None)
        if ax is not None:
            if ax.__class__.__name__ != "PolarAxesSubplot":
                warnings.warn(
                    """
Creating new axes with Polar projection. If you prefer to create your own, please use

    `ax = plt.subplot(121, projection='polar')`

for reference, see: http://matplotlib.org/examples/pylab_examples/polar_demo.html
                    https://github.com/matplotlib/matplotlib/issues/312
                    """
                )
                ax = plt.subplot(111, projection='polar')
        else:
            ax = plt.subplot(111, projection='polar')

        # radial lines
        NN = ndgrid(self.vectorNx, self.vectorNy, np.r_[0])[:, :2]
        NN = NN.reshape((self.vnN[0], self.vnN[1], 2), order='F')
        NN = [NN[:, :, 0], NN[:, :, 1]]
        X1 = np.c_[
            mkvc(NN[0][0, :]),
            mkvc(NN[0][self.nCx, :]),
            mkvc(NN[0][0, :])*np.nan
        ].flatten()
        Y1 = np.c_[
            mkvc(NN[1][0, :]),
            mkvc(NN[1][self.nCx, :]),
            mkvc(NN[1][0, :])*np.nan
        ].flatten()

        ax.plot(Y1, X1, linestyle="-", color="C0")

        # circles
        n = 100
        XY2 = [
            ax.plot(
                np.linspace(0., np.pi*2, n), r*np.ones(n), linestyle="-",
                color="C0"
            )
            for r in self.vectorNx
        ]

        return ax

    def plotImage(self, *args, **kwargs):
        return self._plotCylTensorMesh('plotImage', *args, **kwargs)


class CurviView(object):
    """
    Provides viewing functions for CurvilinearMesh

    This class is inherited by CurvilinearMesh

    """
    def __init__(self):
        pass

    def plotGrid(
        self, ax=None, nodes=False, faces=False, centers=False, edges=False,
        lines=True, showIt=False
    ):
        """Plot the nodal, cell-centered and staggered grids for 1, 2 and 3 dimensions.


        .. plot::
            :include-source:

            import discretize
            X, Y = discretize.utils.exampleLrmGrid([3, 3], 'rotate')
            M = discretize.CurvilinearMesh([X, Y])
            M.plotGrid(showIt=True)

        """
        import matplotlib.pyplot as plt
        import matplotlib
        from mpl_toolkits.mplot3d import Axes3D

        axOpts = {'projection': '3d'} if self.dim == 3 else {}
        if ax is None:
            ax = plt.subplot(111, **axOpts)

        NN = self.r(self.gridN, 'N', 'N', 'M')
        if self.dim == 2:

            if lines:
                X1 = np.c_[mkvc(NN[0][:-1, :]), mkvc(NN[0][1:, :]), mkvc(NN[0][:-1, :])*np.nan].flatten()
                Y1 = np.c_[mkvc(NN[1][:-1, :]), mkvc(NN[1][1:, :]), mkvc(NN[1][:-1, :])*np.nan].flatten()

                X2 = np.c_[mkvc(NN[0][:, :-1]), mkvc(NN[0][:, 1:]), mkvc(NN[0][:, :-1])*np.nan].flatten()
                Y2 = np.c_[mkvc(NN[1][:, :-1]), mkvc(NN[1][:, 1:]), mkvc(NN[1][:, :-1])*np.nan].flatten()

                X = np.r_[X1, X2]
                Y = np.r_[Y1, Y2]

                ax.plot(X, Y, color="C0", linestyle="-")
            if centers:
                ax.plot(
                    self.gridCC[:, 0], self.gridCC[:, 1], color="C1",
                    linestyle="", marker="o"
                )

            # Nx = self.r(self.normals, 'F', 'Fx', 'V')
            # Ny = self.r(self.normals, 'F', 'Fy', 'V')
            # Tx = self.r(self.tangents, 'E', 'Ex', 'V')
            # Ty = self.r(self.tangents, 'E', 'Ey', 'V')

            # ax.plot(self.gridN[:, 0], self.gridN[:, 1], 'bo')

            # nX = np.c_[self.gridFx[:, 0], self.gridFx[:, 0] + Nx[0]*length, self.gridFx[:, 0]*np.nan].flatten()
            # nY = np.c_[self.gridFx[:, 1], self.gridFx[:, 1] + Nx[1]*length, self.gridFx[:, 1]*np.nan].flatten()
            # ax.plot(self.gridFx[:, 0], self.gridFx[:, 1], 'rs')
            # ax.plot(nX, nY, 'r-')

            # nX = np.c_[self.gridFy[:, 0], self.gridFy[:, 0] + Ny[0]*length, self.gridFy[:, 0]*np.nan].flatten()
            # nY = np.c_[self.gridFy[:, 1], self.gridFy[:, 1] + Ny[1]*length, self.gridFy[:, 1]*np.nan].flatten()
            # #ax.plot(self.gridFy[:, 0], self.gridFy[:, 1], 'gs')
            # ax.plot(nX, nY, 'g-')

            # tX = np.c_[self.gridEx[:, 0], self.gridEx[:, 0] + Tx[0]*length, self.gridEx[:, 0]*np.nan].flatten()
            # tY = np.c_[self.gridEx[:, 1], self.gridEx[:, 1] + Tx[1]*length, self.gridEx[:, 1]*np.nan].flatten()
            # ax.plot(self.gridEx[:, 0], self.gridEx[:, 1], 'r^')
            # ax.plot(tX, tY, 'r-')

            # nX = np.c_[self.gridEy[:, 0], self.gridEy[:, 0] + Ty[0]*length, self.gridEy[:, 0]*np.nan].flatten()
            # nY = np.c_[self.gridEy[:, 1], self.gridEy[:, 1] + Ty[1]*length, self.gridEy[:, 1]*np.nan].flatten()
            # #ax.plot(self.gridEy[:, 0], self.gridEy[:, 1], 'g^')
            # ax.plot(nX, nY, 'g-')

        elif self.dim == 3:
            X1 = np.c_[mkvc(NN[0][:-1, :, :]), mkvc(NN[0][1:, :, :]), mkvc(NN[0][:-1, :, :])*np.nan].flatten()
            Y1 = np.c_[mkvc(NN[1][:-1, :, :]), mkvc(NN[1][1:, :, :]), mkvc(NN[1][:-1, :, :])*np.nan].flatten()
            Z1 = np.c_[mkvc(NN[2][:-1, :, :]), mkvc(NN[2][1:, :, :]), mkvc(NN[2][:-1, :, :])*np.nan].flatten()

            X2 = np.c_[mkvc(NN[0][:, :-1, :]), mkvc(NN[0][:, 1:, :]), mkvc(NN[0][:, :-1, :])*np.nan].flatten()
            Y2 = np.c_[mkvc(NN[1][:, :-1, :]), mkvc(NN[1][:, 1:, :]), mkvc(NN[1][:, :-1, :])*np.nan].flatten()
            Z2 = np.c_[mkvc(NN[2][:, :-1, :]), mkvc(NN[2][:, 1:, :]), mkvc(NN[2][:, :-1, :])*np.nan].flatten()

            X3 = np.c_[mkvc(NN[0][:, :, :-1]), mkvc(NN[0][:, :, 1:]), mkvc(NN[0][:, :, :-1])*np.nan].flatten()
            Y3 = np.c_[mkvc(NN[1][:, :, :-1]), mkvc(NN[1][:, :, 1:]), mkvc(NN[1][:, :, :-1])*np.nan].flatten()
            Z3 = np.c_[mkvc(NN[2][:, :, :-1]), mkvc(NN[2][:, :, 1:]), mkvc(NN[2][:, :, :-1])*np.nan].flatten()

            X = np.r_[X1, X2, X3]
            Y = np.r_[Y1, Y2, Y3]
            Z = np.r_[Z1, Z2, Z3]

            ax.plot(X, Y, 'C0', zs=Z)
            ax.set_zlabel('x3')

        ax.grid(True)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

        if showIt:
            plt.show()

        return ax

    def plotImage(
        self, I, ax=None, showIt=False, grid=False, clim=None
    ):
        if self.dim == 3:
            raise NotImplementedError('This is not yet done!')

        import matplotlib.pyplot as plt
        import matplotlib
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.colors as colors
        import matplotlib.cm as cmx

        if ax is None:
            ax = plt.subplot(111)

        jet = cm = plt.get_cmap('jet')
        cNorm  = colors.Normalize(
            vmin=I.min() if clim is None else clim[0],
            vmax=I.max() if clim is None else clim[1])

        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        # ax.set_xlim((self.x0[0], self.h[0].sum()))
        # ax.set_ylim((self.x0[1], self.h[1].sum()))

        Nx = self.r(self.gridN[:, 0], 'N', 'N', 'M')
        Ny = self.r(self.gridN[:, 1], 'N', 'N', 'M')
        cell = self.r(I, 'CC', 'CC', 'M')

        for ii in range(self.nCx):
            for jj in range(self.nCy):
                I = [ii, ii+1, ii+1, ii]
                J = [jj, jj, jj+1, jj+1]
                ax.add_patch(plt.Polygon(np.c_[Nx[I, J], Ny[I, J]], facecolor=scalarMap.to_rgba(cell[ii, jj]), edgecolor='k' if grid else 'none'))

        scalarMap._A = []  # http://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if showIt:
            plt.show()
        return [scalarMap]
