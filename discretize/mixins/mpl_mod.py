import numpy as np
import warnings
from discretize.utils import mkvc, ndgrid
from discretize.utils.code_utils import deprecate_method

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib import rc_params
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import discretize


class InterfaceMPL(object):
    """This class is used for simple ``discretize`` mesh plotting using matplotlib.

    This interface adds three methods to the meshes. ``plot_grid`` will plot the
    grid points of each mesh in 2D and 3D. ``plot_image`` for 2D image ploting of
    models. and ``plot_slice`` for plotting a 2D slice through a 3D mesh.
    """

    def plot_grid(
        self,
        ax=None,
        nodes=False,
        faces=False,
        centers=False,
        edges=False,
        lines=True,
        show_it=False,
        **kwargs,
    ):
        """Plot the nodal, cell-centered and staggered grids.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            The axes to draw on. None produces a new Axes.
        nodes, faces, centers, edges, lines : bool, optional
            Whether to plot the corresponding item
        show_it : bool, optional
            whether to call plt.show()
        color : Color or str, optional
            If lines=True, the color of the lines, defaults to first color.
        linewidth : float, optional
            If lines=True, the linewidth for the lines.

        Returns
        -------
        matplotlib.axes.Axes
            Axes handle for the plot

        Other Parameters
        ----------------
        edges_x, edges_y, edges_z, faces_x, faces_y, faces_z : bool, optional
            When plotting a ``TreeMesh``, these are also options to plot the
            individual component items.
        cell_line : bool, optional
            When plotting a ``TreeMesh``, you can also plot a line through the
            cell centers in order.
        slice : {'both', 'theta', 'z'}
            When plotting a ``CylindricalMesh``, which dimension to slice over.

        Notes
        -----
        Excess arguments are passed on to `plot`

        Examples
        --------

        Plotting a 2D TensorMesh grid

        >>> from matplotlib import pyplot as plt
        >>> import discretize
        >>> import numpy as np
        >>> h1 = np.linspace(.1, .5, 3)
        >>> h2 = np.linspace(.1, .5, 5)
        >>> mesh = discretize.TensorMesh([h1, h2])
        >>> mesh.plot_grid(nodes=True, faces=True, centers=True, lines=True)
        >>> plt.show()

        Plotting a 3D TensorMesh grid

        >>> from matplotlib import pyplot as plt
        >>> import discretize
        >>> import numpy as np
        >>> h1 = np.linspace(.1, .5, 3)
        >>> h2 = np.linspace(.1, .5, 5)
        >>> h3 = np.linspace(.1, .5, 3)
        >>> mesh = discretize.TensorMesh([h1, h2, h3])
        >>> mesh.plot_grid(nodes=True, faces=True, centers=True, lines=True)
        >>> plt.show()

        Plotting a 2D CurvilinearMesh

        >>> from matplotlib import pyplot as plt
        >>> import discretize
        >>> X, Y = discretize.utils.exampleLrmGrid([10, 10], 'rotate')
        >>> M = discretize.CurvilinearMesh([X, Y])
        >>> M.plot_grid()
        >>> plt.show()

        Plotting a 3D CurvilinearMesh

        >>> from matplotlib import pyplot as plt
        >>> import discretize
        >>> X, Y, Z = discretize.utils.exampleLrmGrid([5, 5, 5], 'rotate')
        >>> M = discretize.CurvilinearMesh([X, Y, Z])
        >>> M.plot_grid()
        >>> plt.show()

        Plotting a 2D TreeMesh

        >>> from matplotlib import pyplot as plt
        >>> import discretize
        >>> M = discretize.TreeMesh([32, 32])
        >>> M.insert_cells([[0.25, 0.25]], [4])
        >>> M.plot_grid()
        >>> plt.show()

        Plotting a 3D TreeMesh

        >>> from matplotlib import pyplot as plt
        >>> import discretize
        >>> M = discretize.TreeMesh([32, 32, 32])
        >>> M.insert_cells([[0.3, 0.75, 0.22]], [4])
        >>> M.plot_grid()
        >>> plt.show()
        """
        mesh_type = self._meshType.lower()
        plotters = {
            "tree": self.__plot_grid_tree,
            "tensor": self.__plot_grid_tensor,
            "curv": self.__plot_grid_curv,
            "cyl": self.__plot_grid_cyl,
        }
        try:
            plotter = plotters[mesh_type]
        except KeyError:
            raise NotImplementedError(
                "Mesh type `{}` does not have a plot_grid implementation.".format(
                    type(self).__name__
                )
            )

        if "showIt" in kwargs:
            show_it = kwargs.pop("showIt")
            warnings.warn(
                "showIt has been deprecated, please use show_it", FutureWarning
            )

        if ax is not None:
            ax_test = ax
            if not isinstance(ax, (list, tuple, np.ndarray)):
                ax_test = (ax,)
            for a in ax_test:
                if not isinstance(a, matplotlib.axes.Axes):
                    raise TypeError("ax must be an matplotlib.axes.Axes")
        elif mesh_type != "cyl":
            axOpts = {"projection": "3d"} if self.dim == 3 else {}
            plt.figure()
            ax = plt.subplot(111, **axOpts)

        rcParams = rc_params()
        if lines:
            kwargs["color"] = kwargs.get("color", rcParams["lines.color"])
            kwargs["linewidth"] = kwargs.get("linewidth", rcParams["lines.linewidth"])

        out = plotter(
            ax=ax,
            nodes=nodes,
            faces=faces,
            centers=centers,
            edges=edges,
            lines=lines,
            **kwargs,
        )
        if show_it:
            plt.show()
        return out

    def plot_image(
        self,
        v,
        v_type="CC",
        grid=False,
        view="real",
        ax=None,
        clim=None,
        show_it=False,
        pcolor_opts=None,
        stream_opts=None,
        grid_opts=None,
        range_x=None,
        range_y=None,
        sample_grid=None,
        stream_thickness=None,
        stream_threshold=None,
        **kwargs,
    ):
        """Plots fields on the given mesh.

        Parameters
        ----------
        v : numpy.ndarray
            values to plot
        v_type : {'CC','CCV', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez'}
            Where the values of v are defined.
        view : {'real', 'imag', 'abs', 'vec'}
            How to view the array.
        ax : matplotlib.axes.Axes, optional
            The axes to draw on. None produces a new Axes.
        clim : tuple of float, optional
            length 2 tuple of (vmin, vmax) for the color limits
        range_x, range_y : tuple of float, optional
            length 2 tuple of (min, max) for the bounds of the plot axes.
        pcolor_opts : dict, optional
            Arguments passed on to ``pcolormesh``
        grid : bool, optional
            Whether to plot the edges of the mesh cells.
        grid_opts : dict, optional
            If ``grid`` is true, arguments passed on to ``plot`` for grid
        sample_grid : tuple of numpy.ndarray, optional
            If ``view`` == 'vec', mesh cell widths (hx, hy) to interpolate onto for vector plotting
        stream_opts : dict, optional
            If ``view`` == 'vec', arguments passed on to ``streamplot``
        stream_thickness : float, optional
            If ``view`` == 'vec', linewidth for ``streamplot``
        stream_threshold : float, optional
            If ``view`` == 'vec', only plots vectors with magnitude above this threshold
        show_it : bool, optional
            Whether to call plt.show()
        numbering : bool, optional
            For 3D TensorMesh only, show the numbering of the slices
        annotation_color : Color or str, optional
            For 3D TensorMesh only, color of the annotation

        Examples
        --------
        2D ``TensorMesh`` plotting

        >>> from matplotlib import pyplot as plt
        >>> import discretize
        >>> import numpy as np
        >>> M = discretize.TensorMesh([20, 20])
        >>> v = np.sin(M.gridCC[:, 0]*2*np.pi)*np.sin(M.gridCC[:, 1]*2*np.pi)
        >>> M.plot_image(v)
        >>> plt.show()

        3D ``TensorMesh`` plotting

        >>> import discretize
        >>> import numpy as np
        >>> M = discretize.TensorMesh([20, 20, 20])
        >>> v = np.sin(M.gridCC[:, 0]*2*np.pi)*np.sin(M.gridCC[:, 1]*2*np.pi)*np.sin(M.gridCC[:, 2]*2*np.pi)
        >>> M.plot_image(v, annotation_color='k')
        >>> plt.show()
        """
        mesh_type = self._meshType.lower()
        plotters = {
            "tree": self.__plot_image_tree,
            "tensor": self.__plot_image_tensor,
            "curv": self.__plot_image_curv,
            "cyl": self.__plot_image_cyl,
        }
        try:
            plotter = plotters[mesh_type]
        except KeyError:
            raise NotImplementedError(
                "Mesh type `{}` does not have a plot_image implementation.".format(
                    type(self).__name__
                )
            )

        if "pcolorOpts" in kwargs:
            pcolor_opts = kwargs.pop("pcolorOpts")
            warnings.warn(
                "pcolorOpts has been deprecated, please use pcolor_opts",
                DeprecationWarning,
            )
        if "streamOpts" in kwargs:
            stream_opts = kwargs.pop("streamOpts")
            warnings.warn(
                "streamOpts has been deprecated, please use stream_opts",
                DeprecationWarning,
            )
        if "gridOpts" in kwargs:
            grid_opts = kwargs.pop("gridOpts")
            warnings.warn(
                "gridOpts has been deprecated, please use grid_opts", DeprecationWarning
            )
        if "showIt" in kwargs:
            show_it = kwargs.pop("showIt")
            warnings.warn(
                "showIt has been deprecated, please use show_it", DeprecationWarning
            )
        if "vType" in kwargs:
            v_type = kwargs.pop("vType")
            warnings.warn(
                "vType has been deprecated, please use v_type", DeprecationWarning
            )

        # Some Error checking and common defaults
        if pcolor_opts is None:
            pcolor_opts = {}
        if stream_opts is None:
            stream_opts = {"color": "k"}
        if grid_opts is None:
            if grid:
                grid_opts = {"color": "k"}
            else:
                grid_opts = {}
        v_typeOptsCC = ["N", "CC", "Fx", "Fy", "Ex", "Ey"]
        v_typeOptsV = ["CCv", "F", "E"]
        v_typeOpts = v_typeOptsCC + v_typeOptsV
        if view == "vec":
            if v_type not in v_typeOptsV:
                raise ValueError(
                    "v_type must be in ['{0!s}'] when view='vec'".format(
                        "', '".join(v_typeOptsV)
                    )
                )
        if v_type not in v_typeOpts:
            raise ValueError(
                "v_type must be in ['{0!s}']".format("', '".join(v_typeOpts))
            )

        viewOpts = ["real", "imag", "abs", "vec"]
        if view not in viewOpts:
            raise ValueError("view must be in ['{0!s}']".format("', '".join(viewOpts)))

        if v.dtype == complex and view == "vec":
            raise NotImplementedError("Can not plot a complex vector.")

        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(111)
        else:
            if not isinstance(ax, matplotlib.axes.Axes):
                raise TypeError("ax must be an Axes!")
            fig = ax.figure
        if clim is not None:
            pcolor_opts["vmin"] = clim[0]
            pcolor_opts["vmax"] = clim[1]

        out = plotter(
            v,
            v_type=v_type,
            view=view,
            ax=ax,
            range_x=range_x,
            range_y=range_y,
            pcolor_opts=pcolor_opts,
            grid=grid,
            grid_opts=grid_opts,
            sample_grid=sample_grid,
            stream_opts=stream_opts,
            stream_threshold=stream_threshold,
            stream_thickness=stream_thickness,
            **kwargs,
        )
        if show_it:
            plt.show()
        return out

    def plot_slice(
        self,
        v,
        v_type="CC",
        normal="Z",
        ind=None,
        grid=False,
        view="real",
        ax=None,
        clim=None,
        show_it=False,
        pcolor_opts=None,
        stream_opts=None,
        grid_opts=None,
        range_x=None,
        range_y=None,
        sample_grid=None,
        stream_threshold=None,
        stream_thickness=None,
        **kwargs,
    ):
        """Plots slice of fields on the given 3D mesh.

        Parameters
        ----------
        v : numpy.ndarray
            values to plot
        v_type : {'CC','CCV', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez'}, or tuple of these options
            Where the values of v are defined.
        normal : {'Z', 'X', 'Y'}
            Normal direction of slicing plane.
        ind : None, optional
            index along dimension of slice. Defaults to the center index.
        view : {'real', 'imag', 'abs', 'vec'}
            How to view the array.
        ax : matplotlib.axes.Axes, optional
            The axes to draw on. None produces a new Axes. Must be None if ``v_type`` is a tuple.
        clim : tuple of float, optional
            length 2 tuple of (vmin, vmax) for the color limits
        range_x, range_y : tuple of float, optional
            length 2 tuple of (min, max) for the bounds of the plot axes.
        pcolor_opts : dict, optional
            Arguments passed on to ``pcolormesh``
        grid : bool, optional
            Whether to plot the edges of the mesh cells.
        grid_opts : dict, optional
            If ``grid`` is true, arguments passed on to ``plot`` for the edges
        sample_grid : tuple of numpy.ndarray, optional
            If ``view`` == 'vec', mesh cell widths (hx, hy) to interpolate onto for vector plotting
        stream_opts : dict, optional
            If ``view`` == 'vec', arguments passed on to ``streamplot``
        stream_thickness : float, optional
            If ``view`` == 'vec', linewidth for ``streamplot``
        stream_threshold : float, optional
            If ``view`` == 'vec', only plots vectors with magnitude above this threshold
        show_it : bool, optional
            Whether to call plt.show()

        Examples
        --------
        Plot a slice of a 3D ``TensorMesh`` solution to a Laplace's equaiton.

        First build the mesh:

        >>> from matplotlib import pyplot as plt
        >>> import discretize
        >>> from pymatsolver import Solver
        >>> import numpy as np
        >>> hx = [(5, 2, -1.3), (2, 4), (5, 2, 1.3)]
        >>> hy = [(2, 2, -1.3), (2, 6), (2, 2, 1.3)]
        >>> hz = [(2, 2, -1.3), (2, 6), (2, 2, 1.3)]
        >>> M = discretize.TensorMesh([hx, hy, hz])

        then build the necessary parts of the PDE:

        >>> q = np.zeros(M.vnC)
        >>> q[[4, 4], [4, 4], [2, 6]]=[-1, 1]
        >>> q = discretize.utils.mkvc(q)
        >>> A = M.face_divergence * M.cell_gradient
        >>> b = Solver(A) * (q)

        and finaly, plot the vector values of the result, which are defined on faces

        >>> M.plot_slice(M.cell_gradient*b, 'F', view='vec', grid=True, pcolor_opts={'alpha':0.8})
        >>> plt.show()
        """
        mesh_type = self._meshType.lower()
        plotters = {
            "tree": self.__plot_slice_tree,
            "tensor": self.__plot_slice_tensor,
            # 'curv': self.__plot_slice_curv,
            # 'cyl': self.__plot_slice_cyl,
        }
        try:
            plotter = plotters[mesh_type]
        except KeyError:
            raise NotImplementedError(
                "Mesh type `{}` does not have a plot_slice implementation.".format(
                    type(self).__name__
                )
            )

        normal = normal.upper()
        if "pcolorOpts" in kwargs:
            pcolor_opts = kwargs["pcolorOpts"]
            warnings.warn(
                "pcolorOpts has been deprecated, please use pcolor_opts",
                DeprecationWarning,
            )
        if "streamOpts" in kwargs:
            stream_opts = kwargs["streamOpts"]
            warnings.warn(
                "streamOpts has been deprecated, please use stream_opts",
                DeprecationWarning,
            )
        if "gridOpts" in kwargs:
            grid_opts = kwargs["gridOpts"]
            warnings.warn(
                "gridOpts has been deprecated, please use grid_opts", DeprecationWarning
            )
        if "showIt" in kwargs:
            show_it = kwargs["showIt"]
            warnings.warn(
                "showIt has been deprecated, please use show_it", DeprecationWarning
            )
        if "vType" in kwargs:
            v_type = kwargs["vType"]
            warnings.warn(
                "vType has been deprecated, please use v_type", DeprecationWarning
            )
        if pcolor_opts is None:
            pcolor_opts = {}
        if stream_opts is None:
            stream_opts = {"color": "k"}
        if grid_opts is None:
            if grid:
                grid_opts = {"color": "k"}
            else:
                grid_opts = {}
        if type(v_type) in [list, tuple]:
            if ax is not None:
                raise TypeError("cannot specify an axis to plot on with this function.")
            fig, axs = plt.subplots(1, len(v_type))
            out = []
            for v_typeI, ax in zip(v_type, axs):
                out += [
                    self.plot_slice(
                        v,
                        v_type=v_typeI,
                        normal=normal,
                        ind=ind,
                        grid=grid,
                        view=view,
                        ax=ax,
                        clim=clim,
                        show_it=False,
                        pcolor_opts=pcolor_opts,
                        stream_opts=stream_opts,
                        grid_opts=grid_opts,
                        stream_threshold=stream_threshold,
                        stream_thickness=stream_thickness,
                    )
                ]
            return out
        viewOpts = ["real", "imag", "abs", "vec"]
        normalOpts = ["X", "Y", "Z"]
        v_typeOpts = [
            "CC",
            "CCv",
            "N",
            "F",
            "E",
            "Fx",
            "Fy",
            "Fz",
            "E",
            "Ex",
            "Ey",
            "Ez",
        ]

        # Some user error checking
        if v_type not in v_typeOpts:
            raise ValueError(
                "v_type must be in ['{0!s}']".format("', '".join(v_typeOpts))
            )
        if not self.dim == 3:
            raise TypeError("Must be a 3D mesh. Use plotImage.")
        if view not in viewOpts:
            raise ValueError("view must be in ['{0!s}']".format("', '".join(viewOpts)))
        if normal not in normalOpts:
            raise ValueError(
                "normal must be in ['{0!s}']".format("', '".join(normalOpts))
            )
        if not isinstance(grid, bool):
            raise TypeError("grid must be a boolean")

        if v.dtype == complex and view == "vec":
            raise NotImplementedError("Can not plot a complex vector.")

        if self.dim == 2:
            raise NotImplementedError("Must be a 3D mesh. Use plotImage.")

        if ax is None:
            plt.figure()
            ax = plt.subplot(111)
        else:
            if not isinstance(ax, matplotlib.axes.Axes):
                raise TypeError("ax must be an matplotlib.axes.Axes")

        if clim is not None:
            pcolor_opts["vmin"] = clim[0]
            pcolor_opts["vmax"] = clim[1]

        out = plotter(
            v,
            v_type=v_type,
            normal=normal,
            ind=ind,
            grid=grid,
            view=view,
            ax=ax,
            pcolor_opts=pcolor_opts,
            stream_opts=stream_opts,
            grid_opts=grid_opts,
            range_x=range_x,
            range_y=range_y,
            sample_grid=sample_grid,
            stream_threshold=stream_threshold,
            stream_thickness=stream_thickness,
            **kwargs,
        )
        if show_it:
            plt.show()
        return out

    def plot_3d_slicer(
        self,
        v,
        xslice=None,
        yslice=None,
        zslice=None,
        v_type="CC",
        view="real",
        axis="xy",
        transparent=None,
        clim=None,
        xlim=None,
        ylim=None,
        zlim=None,
        aspect="auto",
        grid=[2, 2, 1],
        pcolor_opts=None,
        fig=None,
        **kwargs,
    ):
        """Plot slices of a 3D volume, interactively (scroll wheel).

        If called from a notebook, make sure to set

            %matplotlib notebook

        See the class `discretize.View.Slicer` for more information.

        It returns nothing. However, if you need the different figure handles
        you can get it via

          `fig = plt.gcf()`

        and subsequently its children via

          `fig.get_children()`

        and recursively deeper, e.g.,

          `fig.get_children()[0].get_children()`.

        One can also provide an existing figure instance, which can be useful
        for interactive widgets in Notebooks. The provided figure is cleared
        first.

        """
        mesh_type = self._meshType.lower()
        if mesh_type != "tensor":
            raise NotImplementedError(
                "plot_3d_slicer has only been implemented for a TensorMesh"
            )
        # Initiate figure
        if fig is None:
            fig = plt.figure()
        else:
            fig.clf()

        if "pcolorOpts" in kwargs:
            pcolor_opts = kwargs["pcolorOpts"]
            warnings.warn(
                "pcolorOpts has been deprecated, please use pcolor_opts",
                DeprecationWarning,
            )

        # Populate figure
        tracker = Slicer(
            self,
            v,
            xslice,
            yslice,
            zslice,
            v_type,
            view,
            axis,
            transparent,
            clim,
            xlim,
            ylim,
            zlim,
            aspect,
            grid,
            pcolor_opts,
        )

        # Connect figure to scrolling
        fig.canvas.mpl_connect("scroll_event", tracker.onscroll)

        # Show figure
        plt.show()

    # TensorMesh plotting
    def __plot_grid_tensor(
        self,
        ax=None,
        nodes=False,
        faces=False,
        centers=False,
        edges=False,
        lines=True,
        color="C0",
        linewidth=1.0,
        **kwargs,
    ):

        if self.dim == 1:
            if nodes:
                ax.plot(
                    self.gridN, np.ones(self.nN), color="C0", marker="s", linestyle=""
                )
            if centers:
                ax.plot(
                    self.gridCC, np.ones(self.nC), color="C1", marker="o", linestyle=""
                )
            if lines:
                ax.plot(self.gridN, np.ones(self.nN), color="C0", linestyle="-")
            ax.set_xlabel("x1")
        elif self.dim == 2:
            if nodes:
                ax.plot(
                    self.gridN[:, 0],
                    self.gridN[:, 1],
                    color="C0",
                    marker="s",
                    linestyle="",
                )
            if centers:
                ax.plot(
                    self.gridCC[:, 0],
                    self.gridCC[:, 1],
                    color="C1",
                    marker="o",
                    linestyle="",
                )
            if faces:
                ax.plot(
                    self.gridFx[:, 0],
                    self.gridFx[:, 1],
                    color="C2",
                    marker=">",
                    linestyle="",
                )
                ax.plot(
                    self.gridFy[:, 0],
                    self.gridFy[:, 1],
                    color="C2",
                    marker="^",
                    linestyle="",
                )
            if edges:
                ax.plot(
                    self.gridEx[:, 0],
                    self.gridEx[:, 1],
                    color="C3",
                    marker=">",
                    linestyle="",
                )
                ax.plot(
                    self.gridEy[:, 0],
                    self.gridEy[:, 1],
                    color="C3",
                    marker="^",
                    linestyle="",
                )

            # Plot the grid lines
            if lines:
                NN = self.reshape(self.gridN, "N", "N", "M")
                nCx, nCy = self.shape_cells
                X1 = np.c_[
                    mkvc(NN[0][0, :]), mkvc(NN[0][nCx, :]), mkvc(NN[0][0, :]) * np.nan
                ].flatten()
                Y1 = np.c_[
                    mkvc(NN[1][0, :]), mkvc(NN[1][nCx, :]), mkvc(NN[1][0, :]) * np.nan
                ].flatten()
                X2 = np.c_[
                    mkvc(NN[0][:, 0]), mkvc(NN[0][:, nCy]), mkvc(NN[0][:, 0]) * np.nan
                ].flatten()
                Y2 = np.c_[
                    mkvc(NN[1][:, 0]), mkvc(NN[1][:, nCy]), mkvc(NN[1][:, 0]) * np.nan
                ].flatten()
                X = np.r_[X1, X2]
                Y = np.r_[Y1, Y2]
                ax.plot(X, Y, color=color, linestyle="-", lw=linewidth)

            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
        elif self.dim == 3:
            if nodes:
                ax.plot(
                    self.gridN[:, 0],
                    self.gridN[:, 1],
                    color="C0",
                    marker="s",
                    linestyle="",
                    zs=self.gridN[:, 2],
                )
            if centers:
                ax.plot(
                    self.gridCC[:, 0],
                    self.gridCC[:, 1],
                    color="C1",
                    marker="o",
                    linestyle="",
                    zs=self.gridCC[:, 2],
                )
            if faces:
                ax.plot(
                    self.gridFx[:, 0],
                    self.gridFx[:, 1],
                    color="C2",
                    marker=">",
                    linestyle="",
                    zs=self.gridFx[:, 2],
                )
                ax.plot(
                    self.gridFy[:, 0],
                    self.gridFy[:, 1],
                    color="C2",
                    marker="<",
                    linestyle="",
                    zs=self.gridFy[:, 2],
                )
                ax.plot(
                    self.gridFz[:, 0],
                    self.gridFz[:, 1],
                    color="C2",
                    marker="^",
                    linestyle="",
                    zs=self.gridFz[:, 2],
                )
            if edges:
                ax.plot(
                    self.gridEx[:, 0],
                    self.gridEx[:, 1],
                    color="C3",
                    marker=">",
                    linestyle="",
                    zs=self.gridEx[:, 2],
                )
                ax.plot(
                    self.gridEy[:, 0],
                    self.gridEy[:, 1],
                    color="C3",
                    marker="<",
                    linestyle="",
                    zs=self.gridEy[:, 2],
                )
                ax.plot(
                    self.gridEz[:, 0],
                    self.gridEz[:, 1],
                    color="C3",
                    marker="^",
                    linestyle="",
                    zs=self.gridEz[:, 2],
                )

            # Plot the grid lines
            if lines:
                nCx, nCy, nCz = self.shape_cells
                NN = self.reshape(self.gridN, "N", "N", "M")
                X1 = np.c_[
                    mkvc(NN[0][0, :, :]),
                    mkvc(NN[0][nCx, :, :]),
                    mkvc(NN[0][0, :, :]) * np.nan,
                ].flatten()
                Y1 = np.c_[
                    mkvc(NN[1][0, :, :]),
                    mkvc(NN[1][nCx, :, :]),
                    mkvc(NN[1][0, :, :]) * np.nan,
                ].flatten()
                Z1 = np.c_[
                    mkvc(NN[2][0, :, :]),
                    mkvc(NN[2][nCx, :, :]),
                    mkvc(NN[2][0, :, :]) * np.nan,
                ].flatten()
                X2 = np.c_[
                    mkvc(NN[0][:, 0, :]),
                    mkvc(NN[0][:, nCy, :]),
                    mkvc(NN[0][:, 0, :]) * np.nan,
                ].flatten()
                Y2 = np.c_[
                    mkvc(NN[1][:, 0, :]),
                    mkvc(NN[1][:, nCy, :]),
                    mkvc(NN[1][:, 0, :]) * np.nan,
                ].flatten()
                Z2 = np.c_[
                    mkvc(NN[2][:, 0, :]),
                    mkvc(NN[2][:, nCy, :]),
                    mkvc(NN[2][:, 0, :]) * np.nan,
                ].flatten()
                X3 = np.c_[
                    mkvc(NN[0][:, :, 0]),
                    mkvc(NN[0][:, :, nCz]),
                    mkvc(NN[0][:, :, 0]) * np.nan,
                ].flatten()
                Y3 = np.c_[
                    mkvc(NN[1][:, :, 0]),
                    mkvc(NN[1][:, :, nCz]),
                    mkvc(NN[1][:, :, 0]) * np.nan,
                ].flatten()
                Z3 = np.c_[
                    mkvc(NN[2][:, :, 0]),
                    mkvc(NN[2][:, :, nCz]),
                    mkvc(NN[2][:, :, 0]) * np.nan,
                ].flatten()
                X = np.r_[X1, X2, X3]
                Y = np.r_[Y1, Y2, Y3]
                Z = np.r_[Z1, Z2, Z3]
                ax.plot(X, Y, color=color, linestyle="-", lw=linewidth, zs=Z)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("x3")

        ax.grid(True)

        return ax

    def __plot_image_tensor(
        self,
        v,
        v_type="CC",
        grid=False,
        view="real",
        ax=None,
        pcolor_opts=None,
        stream_opts=None,
        grid_opts=None,
        numbering=True,
        annotation_color="w",
        range_x=None,
        range_y=None,
        sample_grid=None,
        stream_threshold=None,
        **kwargs,
    ):

        if "annotationColor" in kwargs:
            annotation_color = kwargs.pop("annotationColor")
            warnings.warn(
                "annotationColor has been deprecated, please use annotation_color",
                DeprecationWarning,
            )

        if self.dim == 1:
            if v_type == "CC":
                ph = ax.plot(
                    self.cell_centers_x, v, linestyle="-", color="C1", marker="o"
                )
            elif v_type == "N":
                ph = ax.plot(
                    self.nodes_x, v, linestyle="-", color="C0", marker="s"
                )
            ax.set_xlabel("x")
            ax.axis("tight")
        elif self.dim == 2:
            return self.__plot_image_tensor2D(
                v,
                v_type=v_type,
                grid=grid,
                view=view,
                ax=ax,
                pcolor_opts=pcolor_opts,
                stream_opts=stream_opts,
                grid_opts=grid_opts,
                range_x=range_x,
                range_y=range_y,
                sample_grid=sample_grid,
                stream_threshold=stream_threshold,
            )
        elif self.dim == 3:
            # get copy of image and average to cell-centers is necessary
            if v_type == "CC":
                vc = v.reshape(self.vnC, order="F")
            elif v_type == "N":
                vc = (self.aveN2CC * v).reshape(self.vnC, order="F")
            elif v_type in ["Fx", "Fy", "Fz", "Ex", "Ey", "Ez"]:
                aveOp = "ave" + v_type[0] + "2CCV"
                # n = getattr(self, 'vn'+v_type[0])
                # if 'x' in v_type: v = np.r_[v, np.zeros(n[1]), np.zeros(n[2])]
                # if 'y' in v_type: v = np.r_[np.zeros(n[0]), v, np.zeros(n[2])]
                # if 'z' in v_type: v = np.r_[np.zeros(n[0]), np.zeros(n[1]), v]
                v = getattr(self, aveOp) * v  # average to cell centers
                ind_xyz = {"x": 0, "y": 1, "z": 2}[v_type[1]]
                vc = self.reshape(v.reshape((self.nC, -1), order="F"), "CC", "CC", "M")[
                    ind_xyz
                ]

            nCx, nCy, nCz = self.shape_cells
            # determine number oE slices in x and y dimension
            nX = int(np.ceil(np.sqrt(nCz)))
            nY = int(np.ceil(nCz / nX))

            #  allocate space for montage
            C = np.zeros((nX * nCx, nY * nCy))

            for iy in range(int(nY)):
                for ix in range(int(nX)):
                    iz = ix + iy * nX
                    if iz < nCz:
                        C[ix * nCx : (ix + 1) * nCx, iy * nCy : (iy + 1) * nCy] = vc[
                            :, :, iz
                        ]
                    else:
                        C[ix * nCx : (ix + 1) * nCx, iy * nCy : (iy + 1) * nCy] = np.nan

            C = np.ma.masked_where(np.isnan(C), C)
            xx = np.r_[0, np.cumsum(np.kron(np.ones((nX, 1)), self.h[0]).ravel())]
            yy = np.r_[0, np.cumsum(np.kron(np.ones((nY, 1)), self.h[1]).ravel())]
            # Plot the mesh

            ph = ax.pcolormesh(xx, yy, C.T, **pcolor_opts)
            # Plot the lines
            gx = np.arange(nX + 1) * (self.nodes_x[-1] - self.origin[0])
            gy = np.arange(nY + 1) * (self.nodes_y[-1] - self.origin[1])
            # Repeat and seperate with NaN
            gxX = np.c_[gx, gx, gx + np.nan].ravel()
            gxY = np.kron(
                np.ones((nX + 1, 1)), np.array([0, sum(self.h[1]) * nY, np.nan])
            ).ravel()
            gyX = np.kron(
                np.ones((nY + 1, 1)), np.array([0, sum(self.h[0]) * nX, np.nan])
            ).ravel()
            gyY = np.c_[gy, gy, gy + np.nan].ravel()
            ax.plot(gxX, gxY, annotation_color + "-", linewidth=2)
            ax.plot(gyX, gyY, annotation_color + "-", linewidth=2)
            ax.axis("tight")

            if numbering:
                pad = np.sum(self.h[0]) * 0.04
                for iy in range(int(nY)):
                    for ix in range(int(nX)):
                        iz = ix + iy * nX
                        if iz < nCz:
                            ax.text(
                                (ix + 1) * (self.nodes_x[-1] - self.origin[0]) - pad,
                                (iy) * (self.nodes_y[-1] - self.origin[1]) + pad,
                                "#{0:.0f}".format(iz),
                                color=annotation_color,
                                verticalalignment="bottom",
                                horizontalalignment="right",
                                size="x-large",
                            )

        ax.set_title(v_type)
        return (ph,)

    def __plot_image_tensor2D(
        self,
        v,
        v_type="CC",
        grid=False,
        view="real",
        ax=None,
        pcolor_opts=None,
        stream_opts=None,
        grid_opts=None,
        range_x=None,
        range_y=None,
        sample_grid=None,
        stream_threshold=None,
        stream_thickness=None,
    ):
        """Common function for plotting an image of a TensorMesh"""

        if ax is None:
            plt.figure()
            ax = plt.subplot(111)
        else:
            if not isinstance(ax, matplotlib.axes.Axes):
                raise AssertionError("ax must be an matplotlib.axes.Axes")

        # Reshape to a cell centered variable
        if v_type == "CC":
            pass
        elif v_type == "CCv":
            if view != "vec":
                raise AssertionError("Other types for CCv not supported")
        elif v_type in ["F", "E", "N"]:
            aveOp = "ave" + v_type + ("2CCV" if view == "vec" else "2CC")
            v = getattr(self, aveOp) * v  # average to cell centers (might be a vector)
        elif v_type in ["Fx", "Fy", "Ex", "Ey"]:
            aveOp = "ave" + v_type[0] + "2CCV"
            v = getattr(self, aveOp) * v  # average to cell centers (might be a vector)
            xORy = {"x": 0, "y": 1}[v_type[1]]
            v = v.reshape((self.nC, -1), order="F")[:, xORy]

        out = ()
        if view in ["real", "imag", "abs"]:
            v = self.reshape(v, "CC", "CC", "M")
            v = getattr(np, view)(v)  # e.g. np.real(v)
            v = np.ma.masked_where(np.isnan(v), v)
            out += (
                ax.pcolormesh(
                    self.nodes_x,
                    self.nodes_y,
                    v.T,
                    **{**pcolor_opts, **grid_opts},
                ),
            )
        elif view in ["vec"]:
            # Matplotlib seems to not support irregular
            # spaced vectors at the moment. So we will
            # Interpolate down to a regular mesh at the
            # smallest mesh size in this 2D slice.
            if sample_grid is not None:
                hxmin = sample_grid[0]
                hymin = sample_grid[1]
            else:
                hxmin = self.h[0].min()
                hymin = self.h[1].min()

            if range_x is not None:
                dx = range_x[1] - range_x[0]
                nxi = int(dx / hxmin)
                hx = np.ones(nxi) * dx / nxi
                origin_x = range_x[0]
            else:
                nxi = int(self.h[0].sum() / hxmin)
                hx = np.ones(nxi) * self.h[0].sum() / nxi
                origin_x = self.origin[0]

            if range_y is not None:
                dy = range_y[1] - range_y[0]
                nyi = int(dy / hymin)
                hy = np.ones(nyi) * dy / nyi
                origin_y = range_y[0]
            else:
                nyi = int(self.h[1].sum() / hymin)
                hy = np.ones(nyi) * self.h[1].sum() / nyi
                origin_y = self.origin[1]

            U, V = self.reshape(v.reshape((self.nC, -1), order="F"), "CC", "CC", "M")

            tMi = self.__class__(h=[hx, hy], origin=np.r_[origin_x, origin_y])
            P = self.get_interpolation_matrix(tMi.gridCC, "CC", zerosOutside=True)

            Ui = tMi.reshape(P * mkvc(U), "CC", "CC", "M")
            Vi = tMi.reshape(P * mkvc(V), "CC", "CC", "M")
            # End Interpolation

            x = self.nodes_x
            y = self.nodes_y

            if range_x is not None:
                x = tMi.nodes_x

            if range_y is not None:
                y = tMi.nodes_y

            if range_x is not None or range_y is not None:  # use interpolated values
                U = Ui
                V = Vi

            if stream_threshold is not None:
                mask_me = np.sqrt(Ui ** 2 + Vi ** 2) <= stream_threshold
                Ui = np.ma.masked_where(mask_me, Ui)
                Vi = np.ma.masked_where(mask_me, Vi)

            if stream_thickness is not None:
                scaleFact = np.copy(stream_thickness)

                # Calculate vector amplitude
                vecAmp = np.sqrt(U ** 2 + V ** 2).T

                # Form bounds to knockout the top and bottom 10%
                vecAmp_sort = np.sort(vecAmp.ravel())
                nVecAmp = vecAmp.size
                tenPercInd = int(np.ceil(0.1 * nVecAmp))
                lowerBound = vecAmp_sort[tenPercInd]
                upperBound = vecAmp_sort[-tenPercInd]

                lowInds = np.where(vecAmp < lowerBound)
                vecAmp[lowInds] = lowerBound

                highInds = np.where(vecAmp > upperBound)
                vecAmp[highInds] = upperBound

                # Normalize amplitudes 0-1
                norm_thickness = vecAmp / vecAmp.max()

                # Scale by user defined thickness factor
                stream_thickness = scaleFact * norm_thickness

                # Add linewidth to stream_opts
                stream_opts.update({"linewidth": stream_thickness})

            out += (
                ax.pcolormesh(
                    x,
                    y,
                    np.sqrt(U ** 2 + V ** 2).T,
                    **{**pcolor_opts, **grid_opts},
                ),
            )
            out += (
                ax.streamplot(
                    tMi.cell_centers_x,
                    tMi.cell_centers_y,
                    Ui.T,
                    Vi.T,
                    **stream_opts,
                ),
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if range_x is not None:
            ax.set_xlim(*range_x)
        else:
            ax.set_xlim(*self.nodes_x[[0, -1]])

        if range_y is not None:
            ax.set_ylim(*range_y)
        else:
            ax.set_ylim(*self.nodes_y[[0, -1]])
        return out

    def __plot_slice_tensor(
        self,
        v,
        v_type="CC",
        normal="z",
        ind=None,
        grid=False,
        view="real",
        ax=None,
        pcolor_opts=None,
        stream_opts=None,
        grid_opts=None,
        range_x=None,
        range_y=None,
        sample_grid=None,
        stream_threshold=None,
        stream_thickness=None,
        **kwargs,
    ):
        dim_ind = {"X": 0, "Y": 1, "Z": 2}[normal]
        szSliceDim = self.shape_cells[dim_ind]  #: Size of the sliced dimension
        if ind is None:
            ind = szSliceDim // 2
        if not isinstance(ind, int):
            raise TypeError("ind must be an integer")

        def getIndSlice(v):
            if normal == "X":
                v = v[ind, :, :]
            elif normal == "Y":
                v = v[:, ind, :]
            elif normal == "Z":
                v = v[:, :, ind]
            return v

        def doSlice(v):
            if v_type == "CC":
                return getIndSlice(self.reshape(v, "CC", "CC", "M"))
            elif v_type == "CCv":
                if view != "vec":
                    raise AssertionError("Other types for CCv not supported")
            else:
                # Now just deal with 'F' and 'E' (x, y, z, maybe...)
                aveOp = "ave" + v_type + ("2CCV" if view == "vec" else "2CC")
                Av = getattr(self, aveOp)
                if v.size == Av.shape[1]:
                    v = Av * v
                else:
                    v = self.reshape(v, v_type[0], v_type)  # get specific component
                    v = Av * v
                # we should now be averaged to cell centers (might be a vector)
            v = self.reshape(v.reshape((self.nC, -1), order="F"), "CC", "CC", "M")
            if view == "vec":
                outSlice = []
                if "X" not in normal:
                    outSlice.append(getIndSlice(v[0]))
                if "Y" not in normal:
                    outSlice.append(getIndSlice(v[1]))
                if "Z" not in normal:
                    outSlice.append(getIndSlice(v[2]))
                return np.r_[mkvc(outSlice[0]), mkvc(outSlice[1])]
            else:
                return getIndSlice(self.reshape(v, "CC", "CC", "M"))

        h2d = []
        x2d = []
        if "X" not in normal:
            h2d.append(self.h[0])
            x2d.append(self.origin[0])
        if "Y" not in normal:
            h2d.append(self.h[1])
            x2d.append(self.origin[1])
        if "Z" not in normal:
            h2d.append(self.h[2])
            x2d.append(self.origin[2])
        tM = self.__class__(h=h2d, origin=x2d)  #: Temp Mesh
        v2d = doSlice(v)

        out = tM.__plot_image_tensor2D(
            v2d,
            v_type=("CCv" if view == "vec" else "CC"),
            grid=grid,
            view=view,
            ax=ax,
            pcolor_opts=pcolor_opts,
            stream_opts=stream_opts,
            grid_opts=grid_opts,
            range_x=range_x,
            range_y=range_y,
            sample_grid=sample_grid,
            stream_threshold=stream_threshold,
            stream_thickness=stream_thickness,
        )

        ax.set_xlabel("y" if normal == "X" else "x")
        ax.set_ylabel("y" if normal == "Z" else "z")
        ax.set_title("Slice {0:.0f}".format(ind))
        return out

    # CylindricalMesh plotting
    def __plotCylTensorMesh(self, plotType, *args, **kwargs):
        if not self.is_symmetric:
            raise NotImplementedError("We have not yet implemented this type of view.")
        if plotType not in ["plot_image", "plot_grid"]:
            raise TypeError("plotType must be either 'plot_grid' or 'plot_image'.")

        if len(args) > 0:
            val = args[0]

        v_type = kwargs.get("v_type", None)
        mirror = kwargs.pop("mirror", None)
        mirror_data = kwargs.pop("mirror_data", None)

        if mirror_data is not None and mirror is None:
            mirror = True

        if v_type is not None:
            if v_type.upper() != "CCV":
                if v_type.upper() == "F":
                    val = mkvc(self.aveF2CCV * val)
                    if mirror_data is not None:
                        mirror_data = mkvc(self.aveF2CCV * mirror_data)
                    kwargs["v_type"] = "CCv"  # now the vector is cell centered
                if v_type.upper() == "E":
                    val = mkvc(self.aveE2CCV * val)
                    if mirror_data is not None:
                        mirror_data = mkvc(self.aveE2CCV * mirror_data)
                args = (val,) + args[1:]

        if mirror:
            # create a mirrored mesh
            hx = np.hstack([np.flipud(self.h[0]), self.h[0]])
            origin0 = self.origin[0] - self.h[0].sum()
            M = discretize.TensorMesh([hx, self.h[2]], origin=[origin0, self.origin[2]])

            if mirror_data is None:
                mirror_data = val

            if len(val) == self.nC:  # only a single value at cell centers
                val = val.reshape(self.vnC[0], self.vnC[2], order="F")
                mirror_val = mirror_data.reshape(self.vnC[0], self.vnC[2], order="F")
                val = mkvc(np.vstack([np.flipud(mirror_val), val]))

            elif len(val) == 2 * self.nC:
                val_x = val[: self.nC].reshape(self.vnC[0], self.vnC[2], order="F")
                val_z = val[self.nC :].reshape(self.vnC[0], self.vnC[2], order="F")

                mirror_x = mirror_data[: self.nC].reshape(
                    self.vnC[0], self.vnC[2], order="F"
                )
                mirror_z = mirror_data[self.nC :].reshape(
                    self.vnC[0], self.vnC[2], order="F"
                )

                val_x = mkvc(
                    np.vstack([-1.0 * np.flipud(mirror_x), val_x])
                )  # by symmetry
                val_z = mkvc(np.vstack([np.flipud(mirror_z), val_z]))

                val = np.hstack([val_x, val_z])

            args = (val,) + args[1:]
        else:
            M = discretize.TensorMesh(
                [self.h[0], self.h[2]], origin=[self.origin[0], self.origin[2]]
            )

        ax = kwargs.get("ax", None)
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(111)
            kwargs["ax"] = ax
        else:
            if not isinstance(ax, matplotlib.axes.Axes):
                raise AssertionError("ax must be an matplotlib.axes.Axes")
            fig = ax.figure

        out = getattr(M, plotType)(*args, **kwargs)

        ax.set_xlabel("x")
        ax.set_ylabel("z")

        return out

    def __plot_grid_cyl(self, *args, **kwargs):
        if self.is_symmetric:
            return self.__plotCylTensorMesh("plot_grid", *args, **kwargs)

        # allow a slice to be provided for the mesh
        slc = kwargs.pop("slice", None)
        if isinstance(slc, str):
            slc = slc.lower()
        if slc not in ["theta", "z", "both", None]:
            raise ValueError(
                "slice must be either 'theta','z', or 'both' not {}".format(slc)
            )

        # if slc is None, provide slices in both the theta and z directions
        if slc == "theta":
            return self.__plotGridThetaSlice(*args, **kwargs)
        elif slc == "z":
            return self.__plotGridZSlice(*args, **kwargs)
        else:
            ax = kwargs.pop("ax", None)
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
                    polarax = [
                        a for a in ax if a.__class__.__name__ == "PolarAxesSubplot"
                    ]
                    if len(polarax) != 1:
                        warnings.warn(
                            """
                            No polar axes provided. Over-writing the axes. If you prefer to create your
                            own, please use

                                `ax = plt.subplot(121, projection='polar')`

                            for reference, see: http://matplotlib.org/examples/pylab_examples/polar_demo.html
                                                https://github.com/matplotlib/matplotlib/issues/312
                            """
                        )
                        ax = None

                    else:
                        polarax = polarax[0]
                        cartax = [a for a in ax if a != polarax][0]

            # ax may have been None to start with or set to None
            if ax is None:
                fig = plt.figure(figsize=(12, 5))
                polarax = plt.subplot(121, projection="polar")
                cartax = plt.subplot(122)

            # update kwargs with respective axes handles
            kwargspolar = kwargs.copy()
            kwargspolar["ax"] = polarax

            kwargscart = kwargs.copy()
            kwargscart["ax"] = cartax

            ax = []
            ax.append(self.__plotGridZSlice(*args, **kwargspolar))
            ax.append(self.__plotGridThetaSlice(*args, **kwargscart))
            plt.tight_layout()

        return ax

    def __plotGridThetaSlice(self, *args, **kwargs):
        # make a cyl symmetric mesh
        h2d = [self.h[0], 1, self.h[2]]
        mesh2D = self.__class__(h=h2d, origin=self.origin)
        return mesh2D.plot_grid(*args, **kwargs)

    def __plotGridZSlice(self, *args, **kwargs):
        # https://github.com/matplotlib/matplotlib/issues/312
        ax = kwargs.get("ax", None)
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
                ax = plt.subplot(111, projection="polar")
        else:
            ax = plt.subplot(111, projection="polar")

        # radial lines
        NN = ndgrid(self.nodes_x, self.nodes_y, np.r_[0])[:, :2]
        NN = NN.reshape((self.vnN[0], self.vnN[1], 2), order="F")
        NN = [NN[:, :, 0], NN[:, :, 1]]
        X1 = np.c_[
            mkvc(NN[0][0, :]),
            mkvc(NN[0][self.shape_cells[0], :]),
            mkvc(NN[0][0, :]) * np.nan,
        ].flatten()
        Y1 = np.c_[
            mkvc(NN[1][0, :]),
            mkvc(NN[1][self.shape_cells[0], :]),
            mkvc(NN[1][0, :]) * np.nan,
        ].flatten()

        color = kwargs.get("color", "C0")
        linewidth = kwargs.get("linewidth", 1.0)
        ax.plot(Y1, X1, linestyle="-", color=color, lw=linewidth)

        # circles
        n = 100
        XY2 = [
            ax.plot(
                np.linspace(0.0, np.pi * 2, n),
                r * np.ones(n),
                linestyle="-",
                color=color,
                lw=linewidth,
            )
            for r in self.nodes_x
        ]

        return ax

    def __plot_image_cyl(self, *args, **kwargs):
        return self.__plotCylTensorMesh("plot_image", *args, **kwargs)

    # CurvilinearMesh plotting:
    def __plot_grid_curv(
        self,
        ax=None,
        nodes=False,
        faces=False,
        centers=False,
        edges=False,
        lines=True,
        color="C0",
        linewidth=1.0,
        **kwargs,
    ):
        NN = self.reshape(self.gridN, "N", "N", "M")
        if self.dim == 2:
            if lines:
                X1 = np.c_[
                    mkvc(NN[0][:-1, :]),
                    mkvc(NN[0][1:, :]),
                    mkvc(NN[0][:-1, :]) * np.nan,
                ].flatten()
                Y1 = np.c_[
                    mkvc(NN[1][:-1, :]),
                    mkvc(NN[1][1:, :]),
                    mkvc(NN[1][:-1, :]) * np.nan,
                ].flatten()

                X2 = np.c_[
                    mkvc(NN[0][:, :-1]),
                    mkvc(NN[0][:, 1:]),
                    mkvc(NN[0][:, :-1]) * np.nan,
                ].flatten()
                Y2 = np.c_[
                    mkvc(NN[1][:, :-1]),
                    mkvc(NN[1][:, 1:]),
                    mkvc(NN[1][:, :-1]) * np.nan,
                ].flatten()

                X = np.r_[X1, X2]
                Y = np.r_[Y1, Y2]

                ax.plot(X, Y, color=color, linewidth=linewidth, linestyle="-", **kwargs)
        elif self.dim == 3:
            X1 = np.c_[
                mkvc(NN[0][:-1, :, :]),
                mkvc(NN[0][1:, :, :]),
                mkvc(NN[0][:-1, :, :]) * np.nan,
            ].flatten()
            Y1 = np.c_[
                mkvc(NN[1][:-1, :, :]),
                mkvc(NN[1][1:, :, :]),
                mkvc(NN[1][:-1, :, :]) * np.nan,
            ].flatten()
            Z1 = np.c_[
                mkvc(NN[2][:-1, :, :]),
                mkvc(NN[2][1:, :, :]),
                mkvc(NN[2][:-1, :, :]) * np.nan,
            ].flatten()

            X2 = np.c_[
                mkvc(NN[0][:, :-1, :]),
                mkvc(NN[0][:, 1:, :]),
                mkvc(NN[0][:, :-1, :]) * np.nan,
            ].flatten()
            Y2 = np.c_[
                mkvc(NN[1][:, :-1, :]),
                mkvc(NN[1][:, 1:, :]),
                mkvc(NN[1][:, :-1, :]) * np.nan,
            ].flatten()
            Z2 = np.c_[
                mkvc(NN[2][:, :-1, :]),
                mkvc(NN[2][:, 1:, :]),
                mkvc(NN[2][:, :-1, :]) * np.nan,
            ].flatten()

            X3 = np.c_[
                mkvc(NN[0][:, :, :-1]),
                mkvc(NN[0][:, :, 1:]),
                mkvc(NN[0][:, :, :-1]) * np.nan,
            ].flatten()
            Y3 = np.c_[
                mkvc(NN[1][:, :, :-1]),
                mkvc(NN[1][:, :, 1:]),
                mkvc(NN[1][:, :, :-1]) * np.nan,
            ].flatten()
            Z3 = np.c_[
                mkvc(NN[2][:, :, :-1]),
                mkvc(NN[2][:, :, 1:]),
                mkvc(NN[2][:, :, :-1]) * np.nan,
            ].flatten()

            X = np.r_[X1, X2, X3]
            Y = np.r_[Y1, Y2, Y3]
            Z = np.r_[Z1, Z2, Z3]

            ax.plot(X, Y, Z, color=color, linewidth=linewidth, linestyle="-", **kwargs)
            ax.set_zlabel("x3")

        if nodes:
            ax.plot(*self.gridN.T, color=color, linestyle="", marker="s")
        if centers:
            ax.plot(*self.gridCC.T, color="C1", linestyle="", marker="o")
        if faces:
            ax.plot(*self.gridFx.T, color="C2", marker=">", linestyle="")
            ax.plot(*self.gridFy.T, color="C2", marker="<", linestyle="")
            if self.dim == 3:
                ax.plot(*self.gridFz.T, color="C2", marker="^", linestyle="")
        if edges:
            ax.plot(*self.gridEx.T, color="C3", marker=">", linestyle="")
            ax.plot(*self.gridEy.T, color="C3", marker="<", linestyle="")
            if self.dim == 3:
                ax.plot(*self.gridEz.T, color="C3", marker="^", linestyle="")

        ax.grid(True)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

        return ax

    def __plot_image_curv(
        self,
        v,
        v_type="CC",
        grid=False,
        view="real",
        ax=None,
        pcolor_opts=None,
        grid_opts=None,
        range_x=None,
        range_y=None,
        **kwargs,
    ):
        if self.dim == 3:
            raise NotImplementedError("This is not yet done!")
        if view == "vec":
            raise NotImplementedError(
                "Vector ploting is not supported on CurvilinearMesh (yet)"
            )
        if view in ["real", "imag", "abs"]:
            v = getattr(np, view)(v)  # e.g. np.real(v)
        if v_type == "CC":
            I = v
        elif v_type == "N":
            I = self.aveN2CC * v
        elif v_type in ["Fx", "Fy", "Ex", "Ey"]:
            aveOp = "ave" + v_type[0] + "2CCV"
            ind_xy = {"x": 0, "y": 1}[v_type[1]]
            I = (getattr(self, aveOp) * v).reshape(2, self.nC)[
                ind_xy
            ]  # average to cell centers
        I = np.ma.masked_where(np.isnan(I), I)
        X, Y = (x.T for x in self.node_list)
        out = ax.pcolormesh(
            X,
            Y,
            I.reshape(self.vnC[::-1]),
            antialiased=True,
            **pcolor_opts,
            **grid_opts,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return (out,)

    def __plot_grid_tree(
        self,
        ax=None,
        nodes=False,
        faces=False,
        centers=False,
        edges=False,
        lines=True,
        cell_line=False,
        faces_x=False,
        faces_y=False,
        faces_z=False,
        edges_x=False,
        edges_y=False,
        edges_z=False,
        **kwargs,
    ):
        if faces:
            faces_x = faces_y = True
            if self.dim == 3:
                faces_z = True
        if edges:
            edges_x = edges_y = True
            if self.dim == 3:
                edges_z = True
        if lines or nodes:
            grid_n_full = np.r_[self.nodes, self.hanging_nodes]
        if nodes:
            ax.plot(*grid_n_full.T, color="C0", marker="s", linestyle="")
            # Hanging Nodes
            ax.plot(
                *self.gridhN.T,
                color="C0",
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor="none",
                markeredgecolor="C0",
            )
        if centers:
            ax.plot(*self.cell_centers.T, color="C1", marker="o", linestyle="")
        if cell_line:
            ax.plot(*self.cell_centers.T, color="C1", linestyle=":")
            ax.plot(
                self.cell_centers[[0, -1], 0],
                self.cell_centers[[0, -1], 1],
                color="C1",
                marker="o",
                linestyle="",
            )

        y_mark = "<" if self.dim == 3 else "^"

        if faces_x:
            ax.plot(
                *np.r_[self.faces_x, self.hanging_faces_x].T,
                color="C2",
                marker=">",
                linestyle="",
            )
            # Hanging Faces x
            ax.plot(
                *self.hanging_faces_x.T,
                color="C2",
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor="none",
                markeredgecolor="C2",
            )
        if faces_y:
            ax.plot(
                *np.r_[self.faces_y, self.hanging_faces_y].T,
                color="C2",
                marker=y_mark,
                linestyle="",
            )
            # Hanging Faces y
            ax.plot(
                *self.hanging_faces_y.T,
                color="C2",
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor="none",
                markeredgecolor="C2",
            )
        if faces_z:
            ax.plot(
                *np.r_[self.faces_z, self.hanging_faces_z].T,
                color="C2",
                marker="^",
                linestyle="",
            )
            # Hangin Faces z
            ax.plot(
                *self.hanging_faces_z.T,
                color="C2",
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor="none",
                markeredgecolor="C2",
            )
        if edges_x:
            ax.plot(
                *np.r_[self.edges_x, self.hanging_edges_x].T,
                color="C3",
                marker=">",
                linestyle="",
            )
            # Hanging Edges x
            ax.plot(
                *self.hanging_edges_x.T,
                color="C3",
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor="none",
                markeredgecolor="C3",
            )
        if edges_y:
            ax.plot(
                *np.r_[self.edges_y, self.hanging_edges_y].T,
                color="C3",
                marker=y_mark,
                linestyle="",
            )
            # Hanging Edges y
            ax.plot(
                *self.hanging_edges_y.T,
                color="C3",
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor="none",
                markeredgecolor="C3",
            )
        if edges_z:
            ax.plot(
                *np.r_[self.edges_z, self.hanging_edges_z].T,
                color="C3",
                marker="^",
                linestyle="",
            )
            # Hanging Edges z
            ax.plot(
                *self.hanging_edges_z.T,
                color="C3",
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor="none",
                markeredgecolor="C3",
            )

        if lines:
            edge_nodes = self.edge_nodes
            lines = np.r_[grid_n_full[edge_nodes[0]], grid_n_full[edge_nodes[1]]]
            if self.dim == 2:
                line_segments = LineCollection(lines, **kwargs)
            else:
                lines = np.r_[lines, grid_n_full[edge_nodes[2]]]
                line_segments = Line3DCollection(lines, **kwargs)
            ax.add_collection(line_segments)
            ax.autoscale()

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        if self.dim == 3:
            ax.set_zlabel("x3")

        ax.grid(True)

        return ax

    def __plot_image_tree(
        self,
        v,
        v_type="CC",
        grid=False,
        view="real",
        ax=None,
        pcolor_opts=None,
        grid_opts=None,
        range_x=None,
        range_y=None,
        **kwargs,
    ):
        if self.dim == 3:
            raise NotImplementedError(
                "plotImage is not implemented for 3D TreeMesh, please use plotSlice"
            )

        if view == "vec":
            raise NotImplementedError(
                "Vector ploting is not supported on TreeMesh (yet)"
            )

        if view in ["real", "imag", "abs"]:
            v = getattr(np, view)(v)  # e.g. np.real(v)
        if v_type == "CC":
            I = v
        elif v_type == "N":
            I = self.aveN2CC * v
        elif v_type in ["Fx", "Fy", "Ex", "Ey"]:
            aveOp = "ave" + v_type[0] + "2CCV"
            ind_xy = {"x": 0, "y": 1}[v_type[1]]
            I = (getattr(self, aveOp) * v).reshape(2, self.n_cells)[
                ind_xy
            ]  # average to cell centers

        # pcolormesh call signature
        # def pcolormesh(self, *args, alpha=None, norm=None, cmap=None, vmin=None,
        #            vmax=None, shading='flat', antialiased=False, **kwargs):
        alpha = pcolor_opts.pop("alpha", None)
        norm = pcolor_opts.pop("norm", None)
        cmap = pcolor_opts.pop("cmap", None)
        vmin = pcolor_opts.pop("vmin", None)
        vmax = pcolor_opts.pop("vmax", None)
        shading = pcolor_opts.pop("shading", "flat")
        antialiased = pcolor_opts.pop("antialiased", False)

        node_grid = np.r_[self.nodes, self.hanging_nodes]
        cell_nodes = self.cell_nodes[:, (0, 1, 3, 2)]
        cell_verts = node_grid[cell_nodes]

        # Below taken from pcolormesh source code with QuadMesh exchanged to PolyCollection
        collection = PolyCollection(
            cell_verts, antialiased=antialiased, **{**pcolor_opts, **grid_opts}
        )
        collection.set_alpha(alpha)
        collection.set_array(I)
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        try:
            collection._scale_norm(norm, vmin, vmax)
        except AttributeError:
            collection.set_clim(vmin, vmax)
            collection.autoscale_None()

        ax.grid(False)

        ax.add_collection(collection, autolim=False)

        if range_x is not None:
            minx, maxx = range_x
        else:
            minx, maxx = self.nodes_x[[0, -1]]

        if range_y is not None:
            miny, maxy = range_x
        else:
            miny, maxy = self.nodes_y[[0, -1]]

        collection.sticky_edges.x[:] = [minx, maxx]
        collection.sticky_edges.y[:] = [miny, maxy]
        corners = (minx, miny), (maxx, maxy)
        ax.update_datalim(corners)
        ax._request_autoscale_view()

        return (collection,)

    def __plot_slice_tree(
        self,
        v,
        v_type="CC",
        normal="Z",
        ind=None,
        grid=False,
        view="real",
        ax=None,
        pcolor_opts=None,
        stream_opts=None,
        grid_opts=None,
        range_x=None,
        range_y=None,
        **kwargs,
    ):
        if view == "vec":
            raise NotImplementedError(
                "Vector view plotting is not implemented for TreeMesh (yet)"
            )
        normalInd = {"X": 0, "Y": 1, "Z": 2}[normal]
        antiNormalInd = {"X": [1, 2], "Y": [0, 2], "Z": [0, 1]}[normal]

        h2d = (self.h[antiNormalInd[0]], self.h[antiNormalInd[1]])
        x2d = (self.origin[antiNormalInd[0]], self.origin[antiNormalInd[1]])

        #: Size of the sliced dimension
        szSliceDim = len(self.h[normalInd])
        if ind is None:
            ind = szSliceDim // 2

        cc_tensor = [None, None, None]
        for i in range(3):
            cc_tensor[i] = np.cumsum(np.r_[self.origin[i], self.h[i]])
            cc_tensor[i] = (cc_tensor[i][1:] + cc_tensor[i][:-1]) * 0.5
        slice_loc = cc_tensor[normalInd][ind]

        if not isinstance(ind, int):
            raise ValueError("ind must be an integer")

        # create a temporary TreeMesh with the slice through
        temp_mesh = discretize.TreeMesh(h2d, x2d)
        level_diff = self.max_level - temp_mesh.max_level

        XS = [None, None, None]
        XS[antiNormalInd[0]], XS[antiNormalInd[1]] = np.meshgrid(
            cc_tensor[antiNormalInd[0]], cc_tensor[antiNormalInd[1]]
        )
        XS[normalInd] = np.ones_like(XS[antiNormalInd[0]]) * slice_loc
        loc_grid = np.c_[XS[0].reshape(-1), XS[1].reshape(-1), XS[2].reshape(-1)]
        inds = np.unique(self._get_containing_cell_indexes(loc_grid))

        grid2d = self.gridCC[inds][:, antiNormalInd]
        levels = self._cell_levels_by_indexes(inds) - level_diff
        temp_mesh.insert_cells(grid2d, levels)
        tm_gridboost = np.empty((temp_mesh.nC, 3))
        tm_gridboost[:, antiNormalInd] = temp_mesh.gridCC
        tm_gridboost[:, normalInd] = slice_loc

        # interpolate values to self.gridCC if not 'CC'
        if v_type != "CC":
            aveOp = "ave" + v_type + "2CC"
            Av = getattr(self, aveOp)
            if v.size == Av.shape[1]:
                v = Av * v
            elif len(v_type) == 2:
                # was one of Fx, Fy, Fz, Ex, Ey, Ez
                # assuming v has all three components in these cases
                vec_ind = {"x": 0, "y": 1, "z": 2}[v_type[1]]
                if v_type[0] == "E":
                    i_s = np.cumsum([0, self.nEx, self.nEy, self.nEz])
                elif v_type[0] == "F":
                    i_s = np.cumsum([0, self.nFx, self.nFy, self.nFz])
                v = v[i_s[vec_ind] : i_s[vec_ind + 1]]
                v = Av * v

        # interpolate values from self.gridCC to grid2d
        ind_3d_to_2d = self._get_containing_cell_indexes(tm_gridboost)
        v2d = v[ind_3d_to_2d]

        out = temp_mesh.plot_image(
            v2d,
            v_type="CC",
            grid=grid,
            view=view,
            ax=ax,
            pcolor_opts=pcolor_opts,
            grid_opts=grid_opts,
            range_x=range_x,
            range_y=range_y,
        )

        ax.set_xlabel("y" if normal == "X" else "x")
        ax.set_ylabel("y" if normal == "Z" else "z")
        ax.set_title("Slice {0:d}, {1!s} = {2:4.2f}".format(ind, normal, slice_loc))
        return out

    plotGrid = deprecate_method("plot_grid", "plotGrid", removal_version="1.0.0")
    plotImage = deprecate_method("plot_image", "plotImage", removal_version="1.0.0")
    plotSlice = deprecate_method("plot_slice", "plotSlice", removal_version="1.0.0")


class Slicer(object):
    """Plot slices of a 3D volume, interactively (scroll wheel).

    If called from a notebook, make sure to set

        %matplotlib notebook

    The straight forward usage for the Slicer is through, e.g., a
    `TensorMesh`-mesh, by accessing its `mesh.plot_3d_slicer`.

    If you, however, call this class directly, you have first to initiate a
    figure, and afterwards connect it:

    >>> # You have to initialize a figure
    >>> fig = plt.figure()
    >>> # Then you have to get the tracker from the Slicer
    >>> tracker = discretize.View.Slicer(mesh, Lpout)
    >>> # Finally you have to connect the tracker to the figure
    >>> fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    >>> plt.show()


    **Parameters**

    v : array
        Data array of length self.nC.

    xslice, yslice, zslice : floats, optional
        Initial slice locations (in meter);
        defaults to the middle of the volume.

    v_type: str
        Type of visualization. Default is 'CC'.
        One of ['CC', 'Fx', 'Fy', 'Fz', 'Ex', 'Ey', 'Ez'].

    view : str
        Which component to show. Defaults to 'real'.
        One of  ['real', 'imag', 'abs'].

    axis : 'xy' (default) or 'yx'
        'xy': horizontal axis is x, vertical axis is y. Reversed otherwise.

    transparent : 'slider' or list of floats or pairs of floats, optional
        Values to be removed. E.g. air, water.
        If single value, only exact matches are removed. Pairs are treated as
        ranges. E.g. [0.3, [1, 4], [-np.infty, -10]] removes all values equal
        to 0.3, all values between 1 and 4, and all values smaller than -10.
        If 'slider' is provided it will plot an interactive slider to choose
        the shown range.

    clim : None or list of [min, max]
        For `pcolormesh` (`vmin`, `vmax`). Note: if you use a `norm` (e.g.,
        `LogNorm`) then `vmin`/`vmax` have to be provided in the norm.

    xlim, ylim, zlim : None or list of [min, max]
        Axis limits.

    aspect : 'auto', 'equal', or num
        Aspect ratio of subplots. Defaults to 'auto'.

        A list of two values can be provided. The first will be for the
        XY-plot, the second for the XZ- and YZ-plots, e.g. ['equal', 2] to have
        the vertical dimension exaggerated by a factor of 2.

        WARNING: For anything else than 'auto', unexpected things might happen
                 when zooming, and the subplot-arrangement won't look pretty.

    grid : list of 3 int
        Number of cells occupied by x, y, and z dimension on plt.subplot2grid.

    pcolor_opts : dictionary
        Passed to `pcolormesh`.

    """

    def __init__(
        self,
        mesh,
        v,
        xslice=None,
        yslice=None,
        zslice=None,
        v_type="CC",
        view="real",
        axis="xy",
        transparent=None,
        clim=None,
        xlim=None,
        ylim=None,
        zlim=None,
        aspect="auto",
        grid=[2, 2, 1],
        pcolor_opts=None,
        **kwargs,
    ):
        """Initialize interactive figure."""

        # 0. Some checks, not very extensive
        if "pcolorOpts" in kwargs:
            pcolor_opts = kwargs["pcolorOpts"]
            warnings.warn(
                "pcolorOpts has been deprecated, please use pcolor_opts",
                DeprecationWarning,
            )

        # Add pcolor_opts to self
        self.pc_props = pcolor_opts if pcolor_opts is not None else {}

        # (a) Mesh dimensionality
        if mesh.dim != 3:
            err = "Must be a 3D mesh. Use plotImage instead."
            err += " Mesh provided has {} dimension(s).".format(mesh.dim)
            raise ValueError(err)

        # (b) v_type  # Not yet working for ['CCv']
        v_typeOpts = ["CC", "Fx", "Fy", "Fz", "Ex", "Ey", "Ez"]
        if v_type not in v_typeOpts:
            err = "v_type must be in ['{0!s}'].".format("', '".join(v_typeOpts))
            err += " v_type provided: '{0!s}'.".format(v_type)
            raise ValueError(err)

        if v_type != "CC":
            aveOp = "ave" + v_type + "2CC"
            Av = getattr(mesh, aveOp)
            if v.size == Av.shape[1]:
                v = Av * v
            else:
                v = mesh.reshape(v, v_type[0], v_type)  # get specific component
                v = Av * v

        # (c) vOpts  # Not yet working for 'vec'

        # Backwards compatibility
        if view in ["xy", "yx"]:
            axis = view
            view = "real"

        viewOpts = ["real", "imag", "abs"]
        if view in viewOpts:
            v = getattr(np, view)(v)  # e.g. np.real(v)
        else:
            err = "view must be in ['{0!s}'].".format("', '".join(viewOpts))
            err += " view provided: '{0!s}'.".format(view)
            raise ValueError(err)

        # 1. Store relevant data

        # Store data in self as (nx, ny, nz)
        self.v = mesh.reshape(v.reshape((mesh.nC, -1), order="F"), "CC", "CC", "M")
        self.v = np.ma.masked_where(np.isnan(self.v), self.v)

        # Store relevant information from mesh in self
        self.x = mesh.nodes_x  # x-node locations
        self.y = mesh.nodes_y  # y-node locations
        self.z = mesh.nodes_z  # z-node locations
        self.xc = mesh.cell_centers_x  # x-cell center locations
        self.yc = mesh.cell_centers_y  # y-cell center locations
        self.zc = mesh.cell_centers_z  # z-cell center locations

        # Axis: Default ('xy'): horizontal axis is x, vertical axis is y.
        # Reversed otherwise.
        self.yx = axis == "yx"

        # Store initial slice indices; if not provided, takes the middle.
        if xslice is not None:
            self.xind = np.argmin(np.abs(self.xc - xslice))
        else:
            self.xind = self.xc.size // 2
        if yslice is not None:
            self.yind = np.argmin(np.abs(self.yc - yslice))
        else:
            self.yind = self.yc.size // 2
        if zslice is not None:
            self.zind = np.argmin(np.abs(self.zc - zslice))
        else:
            self.zind = self.zc.size // 2

        # Aspect ratio
        if isinstance(aspect, (list, tuple)):
            aspect1 = aspect[0]
            aspect2 = aspect[1]
        else:
            aspect1 = aspect
            aspect2 = aspect
        if aspect2 in ["auto", "equal"]:
            aspect3 = aspect2
        else:
            aspect3 = 1.0 / aspect2

        # set color limits if clim is None (and norm doesn't have vmin, vmax).
        if clim is None:
            if "norm" in self.pc_props:
                vmin = self.pc_props["norm"].vmin
                vmax = self.pc_props["norm"].vmax
            else:
                vmin = vmax = None
            clim = [
                np.nanmin(self.v) if vmin is None else vmin,
                np.nanmax(self.v) if vmax is None else vmax,
            ]
            # In the case of a homogeneous fullspace provide a small range to
            # avoid problems with colorbar and the three subplots.
            if clim[0] == clim[1]:
                clim[0] *= 0.99
                clim[1] *= 1.01
        else:
            self.pc_props["vmin"] = clim[0]
            self.pc_props["vmax"] = clim[1]

        # ensure vmin/vmax of the norm is consistent with clim
        if "norm" in self.pc_props:
            self.pc_props["norm"].vmin = clim[0]
            self.pc_props["norm"].vmax = clim[1]

        # 2. Start populating figure

        # Get plot2grid dimension
        figgrid = (grid[0] + grid[2], grid[1] + grid[2])

        # Create subplots
        self.fig = plt.gcf()
        self.fig.subplots_adjust(wspace=0.075, hspace=0.1)

        # X-Y
        self.ax1 = plt.subplot2grid(
            figgrid, (0, 0), colspan=grid[1], rowspan=grid[0], aspect=aspect1
        )
        if self.yx:
            self.ax1.set_ylabel("x")
            if ylim is not None:
                self.ax1.set_xlim([ylim[0], ylim[1]])
            if xlim is not None:
                self.ax1.set_ylim([xlim[0], xlim[1]])
        else:
            self.ax1.set_ylabel("y")
            if xlim is not None:
                self.ax1.set_xlim([xlim[0], xlim[1]])
            if ylim is not None:
                self.ax1.set_ylim([ylim[0], ylim[1]])
        self.ax1.xaxis.set_ticks_position("top")
        plt.setp(self.ax1.get_xticklabels(), visible=False)

        # X-Z
        self.ax2 = plt.subplot2grid(
            figgrid,
            (grid[0], 0),
            colspan=grid[1],
            rowspan=grid[2],
            sharex=self.ax1,
            aspect=aspect2,
        )
        self.ax2.yaxis.set_ticks_position("both")
        if self.yx:
            self.ax2.set_xlabel("y")
            if ylim is not None:
                self.ax2.set_xlim([ylim[0], ylim[1]])
        else:
            self.ax2.set_xlabel("x")
            if xlim is not None:
                self.ax2.set_xlim([xlim[0], xlim[1]])
        self.ax2.set_ylabel("z")
        if zlim is not None:
            self.ax2.set_ylim([zlim[0], zlim[1]])

        # Z-Y
        self.ax3 = plt.subplot2grid(
            figgrid,
            (0, grid[1]),
            colspan=grid[2],
            rowspan=grid[0],
            sharey=self.ax1,
            aspect=aspect3,
        )
        self.ax3.yaxis.set_ticks_position("right")
        self.ax3.xaxis.set_ticks_position("both")
        self.ax3.invert_xaxis()
        plt.setp(self.ax3.get_yticklabels(), visible=False)
        if self.yx:
            if xlim is not None:
                self.ax3.set_ylim([xlim[0], xlim[1]])
        else:
            if ylim is not None:
                self.ax3.set_ylim([ylim[0], ylim[1]])
        if zlim is not None:
            self.ax3.set_xlim([zlim[1], zlim[0]])

        # Cross-line properties
        # We have to lines, a thick white one, and in the middle a thin black
        # one, to assure that the lines can be seen on dark and on bright
        # spots.
        self.clpropsw = {"c": "w", "lw": 2, "zorder": 10}
        self.clpropsk = {"c": "k", "lw": 1, "zorder": 11}

        # Initial draw
        self.update_xy()
        self.update_xz()
        self.update_zy()

        # Create colorbar
        plt.colorbar(self.zy_pc, pad=0.15)

        # Remove transparent value
        if isinstance(transparent, str) and transparent.lower() == "slider":
            # Sliders
            self.ax_smin = plt.axes([0.7, 0.11, 0.15, 0.03])
            self.ax_smax = plt.axes([0.7, 0.15, 0.15, 0.03])

            # Limits slightly below/above actual limits, clips otherwise
            self.smin = Slider(self.ax_smin, "Min", *clim, valinit=clim[0])
            self.smax = Slider(self.ax_smax, "Max", *clim, valinit=clim[1])

            def update(val):
                self.v.mask = False  # Re-set
                self.v = np.ma.masked_outside(self.v.data, self.smin.val, self.smax.val)
                # Update plots
                self.update_xy()
                self.update_xz()
                self.update_zy()

            self.smax.on_changed(update)
            self.smin.on_changed(update)

        elif transparent is not None:

            # Loop over values
            for value in transparent:
                # If value is a list/tuple, we treat is as a range
                if isinstance(value, (list, tuple)):
                    self.v = np.ma.masked_inside(self.v, value[0], value[1])
                else:  # Exact value
                    self.v = np.ma.masked_equal(self.v, value)

            # Update plots
            self.update_xy()
            self.update_xz()
            self.update_zy()

        # 3. Keep depth in X-Z and Z-Y in sync

        def do_adjust():
            """Return True if z-axis in X-Z and Z-Y are different."""
            one = np.array(self.ax2.get_ylim())
            two = np.array(self.ax3.get_xlim())[::-1]
            return sum(abs(one - two)) > 0.001  # Difference at least 1 m.

        def on_ylims_changed(ax):
            """Adjust Z-Y if X-Z changed."""
            if do_adjust():
                self.ax3.set_xlim([self.ax2.get_ylim()[1], self.ax2.get_ylim()[0]])

        def on_xlims_changed(ax):
            """Adjust X-Z if Z-Y changed."""
            if do_adjust():
                self.ax2.set_ylim([self.ax3.get_xlim()[1], self.ax3.get_xlim()[0]])

        self.ax3.callbacks.connect("xlim_changed", on_xlims_changed)
        self.ax2.callbacks.connect("ylim_changed", on_ylims_changed)

    def onscroll(self, event):
        """Update index and data when scrolling."""

        # Get scroll direction
        if event.button == "up":
            pm = 1
        else:
            pm = -1

        # Update slice index depending on subplot over which mouse is
        if event.inaxes == self.ax1:  # X-Y
            self.zind = (self.zind + pm) % self.zc.size
            self.update_xy()
        elif event.inaxes == self.ax2:  # X-Z
            if self.yx:
                self.xind = (self.xind + pm) % self.xc.size
            else:
                self.yind = (self.yind + pm) % self.yc.size
            self.update_xz()
        elif event.inaxes == self.ax3:  # Z-Y
            if self.yx:
                self.yind = (self.yind + pm) % self.yc.size
            else:
                self.xind = (self.xind + pm) % self.xc.size
            self.update_zy()

        plt.draw()

    def update_xy(self):
        """Update plot for change in Z-index."""

        # Clean up
        self._clear_elements(["xy_pc", "xz_ahw", "xz_ahk", "zy_avw", "zy_avk"])

        # Draw X-Y slice
        if self.yx:
            zdat = np.rot90(self.v[:, :, self.zind].transpose())
            hor = self.y
            ver = self.x
        else:
            zdat = self.v[:, :, self.zind].transpose()
            hor = self.x
            ver = self.y
        self.xy_pc = self.ax1.pcolormesh(hor, ver, zdat, **self.pc_props)

        # Draw Z-slice intersection in X-Z plot
        self.xz_ahw = self.ax2.axhline(self.zc[self.zind], **self.clpropsw)
        self.xz_ahk = self.ax2.axhline(self.zc[self.zind], **self.clpropsk)

        # Draw Z-slice intersection in Z-Y plot
        self.zy_avw = self.ax3.axvline(self.zc[self.zind], **self.clpropsw)
        self.zy_avk = self.ax3.axvline(self.zc[self.zind], **self.clpropsk)

    def update_xz(self):
        """Update plot for change in Y-index."""

        # Clean up
        self._clear_elements(["xz_pc", "zy_ahk", "zy_ahw", "xy_ahk", "xy_ahw"])

        # Draw X-Z slice
        if self.yx:
            ydat = self.v[-self.xind, :, :].transpose()
            hor = self.y
            ver = self.z
            ind = self.xc[self.xind]
        else:
            ydat = self.v[:, self.yind, :].transpose()
            hor = self.x
            ver = self.z
            ind = self.yc[self.yind]
        self.xz_pc = self.ax2.pcolormesh(hor, ver, ydat, **self.pc_props)

        # Draw X-slice intersection in X-Y plot
        self.xy_ahw = self.ax1.axhline(ind, **self.clpropsw)
        self.xy_ahk = self.ax1.axhline(ind, **self.clpropsk)

        # Draw X-slice intersection in Z-Y plot
        self.zy_ahw = self.ax3.axhline(ind, **self.clpropsw)
        self.zy_ahk = self.ax3.axhline(ind, **self.clpropsk)

    def update_zy(self):
        """Update plot for change in X-index."""

        # Clean up
        self._clear_elements(["zy_pc", "xz_avw", "xz_avk", "xy_avw", "xy_avk"])

        # Draw Z-Y slice
        if self.yx:
            xdat = np.flipud(self.v[:, self.yind, :])
            hor = self.z
            ver = self.x
            ind = self.yc[self.yind]
        else:
            xdat = self.v[self.xind, :, :]
            hor = self.z
            ver = self.y
            ind = self.xc[self.xind]
        self.zy_pc = self.ax3.pcolormesh(hor, ver, xdat, **self.pc_props)

        # Draw Y-slice intersection in X-Y plot
        self.xy_avw = self.ax1.axvline(ind, **self.clpropsw)
        self.xy_avk = self.ax1.axvline(ind, **self.clpropsk)

        # Draw Y-slice intersection in X-Z plot
        self.xz_avw = self.ax2.axvline(ind, **self.clpropsw)
        self.xz_avk = self.ax2.axvline(ind, **self.clpropsk)

    def _clear_elements(self, names):
        """Remove elements from list <names> from plot if they exists."""
        for element in names:
            if hasattr(self, element):
                getattr(self, element).remove()
