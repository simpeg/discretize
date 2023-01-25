"""Module for ``vtk`` interaction with ``discretize``.

This module provides a way for ``discretize`` meshes to be
converted to VTK data objects (and back when possible) if the
`VTK Python package`_ is available.
The :class:`discretize.mixins.vtk_mod.InterfaceVTK` class becomes inherrited
by all mesh objects and allows users to directly convert any given mesh by
calling that mesh's ``to_vtk()`` method
(note that this method will not be available if VTK is not available).

.. _`VTK Python package`: https://pypi.org/project/vtk/

This functionality was originally developed so that discretize could be
interoperable with PVGeo_, providing a direct interface for discretize meshes
within ParaView and other VTK powered platforms. This interoperablity allows
users to visualize their finite volume meshes and model data from discretize
along side all their other georeferenced datasets in a common rendering
environment.

.. _PVGeo: http://pvgeo.org
.. _pyvista: http://docs.pyvista.org

Another notable VTK powered software platforms is ``pyvista`` (see pyvista_ docs)
which provides a direct interface to the VTK software library through accesible
Python data structures and NumPy arrays::

    pip install pyvista

By default, the ``to_vtk()`` method will return a ``pyvista`` data object so that
users can immediately start visualizing their data in 3D.

See :ref:`pyvista_demo_ref` for an example of the types of integrated
visualizations that are possible leveraging the link between discretize, pyvista_,
and PVGeo_:

.. image:: ../images/pyvista_laguna_del_maule.png
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

.. _`Laguna del Maule Bouguer Gravity`: http://docs.simpeg.xyz/content/examples/20-published/plot_laguna_del_maule_inversion.html

"""
import os
import numpy as np
from discretize.utils import cyl2cart

# from ..utils import cyl2cart

import warnings


def load_vtk(extra=None):
    """Lazy load principal VTK routines.

    This is not beautiful. But if VTK is installed, but never used, it reduces
    load time significantly.
    """
    import vtk as _vtk
    import vtk.util.numpy_support as _nps

    if extra:
        if isinstance(extra, str):
            return _vtk, _nps, getattr(_vtk, extra)
        else:
            return _vtk, _nps, [getattr(_vtk, e) for e in extra]
    else:
        return _vtk, _nps


def assign_cell_data(vtkDS, models=None):
    """Assign the model(s) to the VTK dataset as ``CellData``.

    Parameters
    ----------
    vtkDS : pyvista.Common
        Any given VTK data object that has cell data
    models : dict of str:numpy.ndarray
        Name('s) and array('s). Match number of cells
    """
    _, _nps = load_vtk()

    nc = vtkDS.GetNumberOfCells()
    if models is not None:
        for name, mod in models.items():
            # Convert numpy array
            if mod.shape[0] != nc:
                raise RuntimeError(
                    'Number of model cells ({}) (first axis of model array) for "{}" does not match number of mesh cells ({}).'.format(
                        mod.shape[0], name, nc
                    )
                )
            vtkDoubleArr = _nps.numpy_to_vtk(mod, deep=1)
            vtkDoubleArr.SetName(name)
            vtkDS.GetCellData().AddArray(vtkDoubleArr)
    return vtkDS


class InterfaceVTK(object):
    """VTK interface for ``discretize`` meshes.

    Class enabling straight forward conversion between ``discretize``
    meshes and their corresponding `VTK <https://vtk.org/doc/nightly/html/index.html>`__ or
    `PyVista <https://docs.pyvista.org/>`__ data objects. Since ``InterfaceVTK``
    is inherritted by the :class:`~discretize.base.BaseMesh` class, this
    functionality can be called directly from any ``discretize`` mesh!
    Currently this functionality is implemented for :class:`~discretize.CurvilinearMesh`,
    :class:`~discretize.TreeMesh` and :class:`discretize.TensorMesh` classes; not
    implemented for :class:`~discretize.CylindricalMesh`.

    It should be noted that if your mesh is defined on a reference frame that is **not** the
    traditional <X,Y,Z> system with vectors of :math:`(1,0,0)`, :math:`(0,1,0)`,
    and :math:`(0,0,1)`, then the mesh in VTK will be rotated so that it is plotted
    on the traditional reference frame; see examples below.

    Examples
    --------
    The following are examples which use the VTK interface to convert
    discretize meshes to VTK data objects and write to VTK formatted files.
    In the first example example, a tensor mesh whose axes lie on the
    traditional reference frame is converted to a
    :class:`pyvista.RectilinearGrid` object.

    >>> import discretize
    >>> import numpy as np
    >>> h1 = np.linspace(.1, .5, 3)
    >>> h2 = np.linspace(.1, .5, 5)
    >>> h3 = np.linspace(.1, .8, 3)
    >>> mesh = discretize.TensorMesh([h1, h2, h3])

    Get a VTK data object

    >>> dataset = mesh.to_vtk()

    Save this mesh to a VTK file

    >>> mesh.writeVTK('sample_mesh')

    Here, the reference frame of the mesh is rotated. In this case, conversion
    to VTK produces a :class:`pyvista.StructuredGrid` object.

    >>> axis_u = (1,-1,0)
    >>> axis_v = (-1,-1,0)
    >>> axis_w = (0,0,1)
    >>> mesh.orientation = np.array([
    ...    axis_u,
    ...    axis_v,
    ...    axis_w
    ... ])

    Yield the rotated vtkStructuredGrid

    >>> dataset_r = mesh.to_vtk()

    or write it out to a VTK format

    >>> mesh.writeVTK('sample_rotated')

    The two above code snippets produced a :class:`pyvista.RectilinearGrid` and a
    :class:`pyvista.StructuredGrid` respecitvely. To demonstarte the difference, we
    have plotted the two datasets next to each other where the first mesh is in
    green and its data axes are parrallel to the traditional cartesian reference
    frame. The second, rotated mesh is shown in red and its data axes are
    rotated from the traditional Cartesian reference frame as specified by the
    *orientation* property.

    >>> import pyvista
    >>> pyvista.set_plot_theme('document')

    >>> p = pyvista.BackgroundPlotter()
    >>> p.add_mesh(dataset, color='green', show_edges=True)
    >>> p.add_mesh(dataset_r, color='maroon', show_edges=True)
    >>> p.show_grid()
    >>> p.screenshot('vtk-rotated-example.png')

    .. image:: ../../images/vtk-rotated-example.png

    """

    def __tree_mesh_to_vtk(mesh, models=None):
        """Convert the TreeMesh to a vtk object.

        Constructs a :class:`pyvista.UnstructuredGrid` object of this tree mesh and
        the given models as ``cell_arrays`` of that ``pyvista`` dataset.

        Parameters
        ----------
        mesh : discretize.TreeMesh
            The tree mesh to convert to a :class:`pyvista.UnstructuredGrid`
        models : dict(numpy.ndarray)
            Name('s) and array('s). Match number of cells

        """
        _vtk, _nps = load_vtk()

        # Make the data parts for the vtu object
        # Points
        ptsMat = np.vstack((mesh.gridN, mesh.gridhN))

        # Adjust if result was 2D (voxels are pixels in 2D):
        VTK_CELL_TYPE = _vtk.VTK_VOXEL
        if ptsMat.shape[1] == 2:
            # Add Z values of 0.0 if 2D
            ptsMat = np.c_[ptsMat, np.zeros(ptsMat.shape[0])]
            VTK_CELL_TYPE = _vtk.VTK_PIXEL
        if ptsMat.shape[1] != 3:
            raise RuntimeError("Points of the mesh are improperly defined.")
        # Rotate the points to the cartesian system
        ptsMat = np.dot(ptsMat, mesh.rotation_matrix)
        # Grab the points
        vtkPts = _vtk.vtkPoints()
        vtkPts.SetData(_nps.numpy_to_vtk(ptsMat, deep=True))
        # Cells
        cellArray = [c for c in mesh]
        cellConn = np.array([cell.nodes for cell in cellArray])
        cellsMat = np.concatenate(
            (np.ones((cellConn.shape[0], 1), dtype=int) * cellConn.shape[1], cellConn),
            axis=1,
        ).ravel()
        cellsArr = _vtk.vtkCellArray()
        cellsArr.SetNumberOfCells(cellConn.shape[0])
        cellsArr.SetCells(
            cellConn.shape[0],
            _nps.numpy_to_vtk(cellsMat, deep=True, array_type=_vtk.VTK_ID_TYPE),
        )
        # Make the object
        output = _vtk.vtkUnstructuredGrid()
        output.SetPoints(vtkPts)
        output.SetCells(VTK_CELL_TYPE, cellsArr)
        # Add the level of refinement as a cell array
        cell_levels = np.array([cell._level for cell in cellArray])
        refineLevelArr = _nps.numpy_to_vtk(cell_levels, deep=1)
        refineLevelArr.SetName("octreeLevel")
        output.GetCellData().AddArray(refineLevelArr)
        ubc_order = mesh._ubc_order
        # order_ubc will re-order from treemesh ordering to UBC ordering
        # need the opposite operation
        un_order = np.empty_like(ubc_order)
        un_order[ubc_order] = np.arange(len(ubc_order))
        order = _nps.numpy_to_vtk(un_order)
        order.SetName("index_cell_corner")
        output.GetCellData().AddArray(order)
        # Assign the model('s) to the object
        return assign_cell_data(output, models=models)

    def __simplex_mesh_to_vtk(mesh, models=None):
        """Convert the SimplexMesh to a vtk object.

        Constructs a :class:`pyvista.UnstructuredGrid` object of this simplex mesh and
        the given models as ``cell_arrays`` of that ``pyvista`` dataset.

        Parameters
        ----------
        mesh : discretize.SimplexMesh
            The simplex mesh to convert to a :class:`pyvista.UnstructuredGrid`

        models : dict(numpy.ndarray)
            Name('s) and array('s). Match number of cells
        """
        _vtk, _nps = load_vtk()

        # Make the data parts for the vtu object
        # Points
        pts = mesh.nodes
        if mesh.dim == 2:
            cell_type = _vtk.VTK_TRIANGLE
            pts = np.c_[pts, np.zeros(mesh.n_nodes)]
        elif mesh.dim == 3:
            cell_type = _vtk.VTK_TETRA
        vtk_pts = _vtk.vtkPoints()
        vtk_pts.SetData(_nps.numpy_to_vtk(pts, deep=True))

        cell_con_array = np.c_[np.full(mesh.n_cells, mesh.dim + 1), mesh.simplices]
        cells = _vtk.vtkCellArray()
        cells.SetNumberOfCells(mesh.n_cells)
        cells.SetCells(
            mesh.n_cells,
            _nps.numpy_to_vtk(
                cell_con_array.reshape(-1), deep=True, array_type=_vtk.VTK_ID_TYPE
            ),
        )

        output = _vtk.vtkUnstructuredGrid()
        output.SetPoints(vtk_pts)
        output.SetCells(cell_type, cells)
        # Assign the model('s) to the object
        return assign_cell_data(output, models=models)

    @staticmethod
    def __create_structured_grid(ptsMat, dims, models=None):
        """Build a structured grid (an internal helper)."""
        _vtk, _nps = load_vtk()

        # Adjust if result was 2D:
        if ptsMat.shape[1] == 2:
            # Figure out which dim is null
            nullDim = dims.index(None)
            ptsMat = np.insert(ptsMat, nullDim, np.zeros(ptsMat.shape[0]), axis=1)
        if ptsMat.shape[1] != 3:
            raise RuntimeError("Points of the mesh are improperly defined.")
        # Convert the points
        vtkPts = _vtk.vtkPoints()
        vtkPts.SetData(_nps.numpy_to_vtk(ptsMat, deep=True))
        # Uncover hidden dimension
        dims = tuple(0 if dim is None else dim + 1 for dim in dims)
        output = _vtk.vtkStructuredGrid()
        output.SetDimensions(dims[0], dims[1], dims[2])  # note this subtracts 1
        output.SetPoints(vtkPts)
        # Assign the model('s) to the object
        return assign_cell_data(output, models=models)

    def __get_rotated_nodes(mesh):
        """Rotate mesh nodes (a helper routine)."""
        nodes = mesh.gridN
        if mesh.dim == 1:
            nodes = np.c_[mesh.gridN, np.zeros((mesh.nN, 2))]
        elif mesh.dim == 2:
            nodes = np.c_[mesh.gridN, np.zeros((mesh.nN, 1))]
        # Now garuntee nodes are correct
        if nodes.shape != (mesh.nN, 3):
            raise RuntimeError("Nodes of the grid are improperly defined.")
        # Rotate the points based on the axis orientations
        return np.dot(nodes, mesh.rotation_matrix)

    def __tensor_mesh_to_vtk(mesh, models=None):
        """Convert the TensorMesh to a vtk object.

        Constructs a :class:`pyvista.RectilinearGrid`
        (or a :class:`pyvista.StructuredGrid`) object of this tensor mesh and the
        given models as ``cell_arrays`` of that grid.
        If the mesh is defined on a normal cartesian system then a rectilinear
        grid is generated. Otherwise, a structured grid is generated.

        Parameters
        ----------
        mesh : discretize.TensorMesh
            The tensor mesh to convert to a :class:`pyvista.RectilinearGrid`

        models : dict(numpy.ndarray)
            Name('s) and array('s). Match number of cells
        """
        _vtk, _nps = load_vtk()

        # Deal with dimensionalities
        if mesh.dim >= 1:
            vX = mesh.nodes_x
            xD = len(vX)
            yD, zD = 1, 1
            vY, vZ = np.array([0, 0])
        if mesh.dim >= 2:
            vY = mesh.nodes_y
            yD = len(vY)
        if mesh.dim == 3:
            vZ = mesh.nodes_z
            zD = len(vZ)
        # If axis orientations are standard then use a vtkRectilinearGrid
        if not mesh.reference_is_rotated:
            # Use rectilinear VTK grid.
            # Assign the spatial information.
            output = _vtk.vtkRectilinearGrid()
            output.SetDimensions(xD, yD, zD)
            output.SetXCoordinates(_nps.numpy_to_vtk(vX, deep=1))
            output.SetYCoordinates(_nps.numpy_to_vtk(vY, deep=1))
            output.SetZCoordinates(_nps.numpy_to_vtk(vZ, deep=1))
            return assign_cell_data(output, models=models)
        # Use a structured grid where points are rotated to the cartesian system
        ptsMat = InterfaceVTK.__get_rotated_nodes(mesh)
        # Assign the model('s) to the object
        return InterfaceVTK.__create_structured_grid(
            ptsMat, mesh.shape_cells, models=models
        )

    def __curvilinear_mesh_to_vtk(mesh, models=None):
        """Convert the CurvilinearMesh to a vtk object.

        Constructs a :class:`pyvista.StructuredGrid` of this mesh and the given
        models as ``cell_arrays`` of that object.

        Parameters
        ----------
        mesh : discretize.CurvilinearMesh
            The curvilinear mesh to convert to a :class:`pyvista.StructuredGrid`
        models : dict(numpy.ndarray)
            Name('s) and array('s). Match number of cells
        """
        ptsMat = InterfaceVTK.__get_rotated_nodes(mesh)
        return InterfaceVTK.__create_structured_grid(
            ptsMat, mesh.shape_cells, models=models
        )

    def __cyl_mesh_to_vtk(mesh, models=None):
        """Convert the CylindricalMesh to a vtk object.

        Constructs an vtkUnstructuredGrid made of rational Bezier hexahedrons and wedges.
        Wedges happen on the very internal layer about r=0, and hexes occur elsewhere.
        """
        # # Points
        P = mesh._deflation_matrix("nodes", as_ones=True).T.tocsr()

        if np.any(mesh.h[1] >= np.pi):
            raise NotImplementedError(
                "Exporting cylindrical meshes to vtk with angles larger than 180 degrees"
                " is not yet supported."
            )
        # calculate control points
        dphis_half = mesh.h[1] / 2
        phi_controls = mesh.nodes_y + dphis_half
        if mesh.nodes_x[0] == 0.0:
            Rs, Phis, Zs = np.meshgrid(
                mesh.nodes_x[1:], phi_controls, mesh.nodes_z, indexing="ij"
            )
        else:
            Rs, Phis, Zs = np.meshgrid(
                mesh.nodes_x, phi_controls, mesh.nodes_z, indexing="ij"
            )
        Rs /= np.cos(dphis_half)[None, :, None]

        control_nodes = np.c_[
            Rs.reshape(-1, order="F"),
            Phis.reshape(-1, order="F"),
            Zs.reshape(-1, order="F"),
        ]

        cells = np.arange(mesh.n_cells).reshape(mesh.shape_cells, order="F")
        if mesh.nodes_x[0] == 0.0:
            wedge_cells = cells[0].reshape(-1, order="F")
            hex_cells = (cells[1:]).reshape(-1, order="F")
        else:
            hex_cells = cells.reshape(-1, order="F")

        # Hex Cells...
        # calculate indices
        ir, it, iz = np.unravel_index(hex_cells, shape=mesh.shape_cells, order="F")

        irs = np.stack([ir, ir, ir + 1, ir + 1, ir, ir, ir + 1, ir + 1], axis=-1)
        its = np.stack([it + 1, it, it, it + 1, it + 1, it, it, it + 1], axis=-1)
        izs = np.stack([iz, iz, iz, iz, iz + 1, iz + 1, iz + 1, iz + 1], axis=-1)
        i_hex_nodes = np.ravel_multi_index(
            (irs, its, izs), mesh._shape_total_nodes, order="F"
        )
        i_hex_nodes = P.indices[i_hex_nodes]

        if mesh.nodes_x[0] == 0.0:
            irs = np.stack([ir - 1, ir, ir - 1, ir], axis=-1)
        else:
            irs = np.stack([ir, ir + 1, ir, ir + 1], axis=-1)
        its = np.stack([it, it, it, it], axis=-1)
        izs = np.stack([iz, iz, iz + 1, iz + 1], axis=-1)
        i_hex_control_nodes = np.ravel_multi_index((irs, its, izs), Rs.shape, order="F")

        if mesh.nodes_x[0] == 0.0:
            # Wedge Cells nodes
            # put control points along radial edge for halfway points on the edges...
            Phis, Zs = np.meshgrid(mesh.nodes_y, mesh.nodes_z, indexing="ij")
            Rhalfs = np.full_like(Phis, mesh.nodes_x[1] * 0.5)
            wedge_control_nodes = np.c_[
                Rhalfs.reshape(-1, order="F"),
                Phis.reshape(-1, order="F"),
                Zs.reshape(-1, order="F"),
            ]

            # indices for wedge nodes: for cell 0
            ir, it, iz = np.unravel_index(
                wedge_cells, shape=mesh.shape_cells, order="F"
            )
            irs = np.stack([ir, ir + 1, ir + 1, ir, ir + 1, ir + 1], axis=-1)
            its = np.stack([it, it, it + 1, it, it, it + 1], axis=-1)
            izs = np.stack([iz, iz, iz, iz + 1, iz + 1, iz + 1], axis=-1)
            i_wedge_nodes = np.ravel_multi_index(
                (irs, its, izs), mesh._shape_total_nodes, order="F"
            )
            i_wedge_nodes = P.indices[i_wedge_nodes]

            its = np.stack([it, it + 1, it, it + 1], axis=-1)
            izs = np.stack([iz, iz, iz + 1, iz + 1], axis=-1)
            # wrap mode for the duplicated wedge control nodes
            i_wscn = np.ravel_multi_index(
                (its, izs), Rhalfs.shape, order="F", mode="wrap"
            ) + len(control_nodes)

            irs = np.stack([ir, ir], axis=-1)
            its = np.stack([it, it], axis=-1)
            izs = np.stack([iz, iz + 1], axis=-1)
            i_wccn = np.ravel_multi_index((irs, its, izs), Rs.shape, order="F")
            i_wedge_control_nodes = np.c_[
                i_wscn[:, 0],
                i_wccn[:, 0],
                i_wscn[:, 1],
                i_wscn[:, 2],
                i_wccn[:, 1],
                i_wscn[:, 3],
            ]

        _vtk, _nps = load_vtk()
        ####
        # assemble cells
        cell_types = np.empty(mesh.n_cells, dtype=np.uint8)
        cell_types[hex_cells] = _vtk.VTK_BEZIER_HEXAHEDRON

        cell_con = np.empty((mesh.n_cells, 13), dtype=int)
        cell_con[:, 0] = 12
        cell_con[hex_cells, 1:9] = i_hex_nodes
        cell_con[hex_cells, 9:] = i_hex_control_nodes + mesh.n_nodes
        nodes_cyl = np.r_[mesh.nodes, control_nodes]

        # calculate weights for control points
        control_weights = (
            np.sin(np.pi / 2 - dphis_half)[None, :, None]
            * np.ones((mesh.shape_nodes[0] - 1, mesh.shape_nodes[2]))[:, None, :]
        )
        rational_weights = np.r_[
            np.ones(mesh.n_nodes), control_weights.reshape(-1, order="F")
        ]

        higher_order_degrees = np.empty((mesh.n_cells, 3))
        higher_order_degrees[hex_cells, :] = [2, 1, 1]

        if mesh.nodes_x[0] == 0.0:
            cell_types[wedge_cells] = _vtk.VTK_BEZIER_WEDGE
            cell_con[wedge_cells, 1:7] = i_wedge_nodes
            cell_con[wedge_cells, 7:] = i_wedge_control_nodes + mesh.n_nodes
            nodes_cyl = np.r_[nodes_cyl, wedge_control_nodes]
            rational_weights = np.r_[
                rational_weights, np.ones(len(wedge_control_nodes))
            ]
            higher_order_degrees[wedge_cells] = [2, 2, 1]

        nodes = cyl2cart(nodes_cyl)

        vtk_pts = _vtk.vtkPoints()
        vtk_pts.SetData(_nps.numpy_to_vtk(nodes, deep=True))

        cells = _vtk.vtkCellArray()
        cells.SetNumberOfCells(mesh.n_cells)
        cells.SetCells(
            mesh.n_cells,
            _nps.numpy_to_vtk(
                cell_con.reshape(-1), deep=True, array_type=_vtk.VTK_ID_TYPE
            ),
        )
        cell_types = _nps.numpy_to_vtk(cell_types, deep=True)

        output = _vtk.vtkUnstructuredGrid()
        output.SetPoints(vtk_pts)
        output.SetCells(cell_types, cells)

        vtk_rational_weights = _nps.numpy_to_vtk(rational_weights)
        vtk_higher_order_degrees = _nps.numpy_to_vtk(higher_order_degrees)

        output.GetPointData().SetRationalWeights(vtk_rational_weights)
        output.GetCellData().SetHigherOrderDegrees(vtk_higher_order_degrees)
        # Assign the model('s) to the object
        return assign_cell_data(output, models=models)

    def to_vtk(mesh, models=None):
        """Convert mesh (and models) to corresponding VTK or PyVista data object.

        This method converts a ``discretize`` mesh (and associated models) to its
        corresponding `VTK <https://vtk.org/doc/nightly/html/index.html>`__ or
        `PyVista <https://docs.pyvista.org/>`__ data object.

        Parameters
        ----------
        models : dict of [str, (n_cells) numpy.ndarray], optional
            Models are supplied as a dictionary where the keys are the model
            names. Each model is a 1D :class:`numpy.ndarray` of size (n_cells).

        Returns
        -------
        pyvista.UnstructuredGrid, pyvista.RectilinearGrid or pyvista.StructuredGrid
            The corresponding VTK or PyVista data object for the mesh and its models
        """
        converters = {
            "tree": InterfaceVTK.__tree_mesh_to_vtk,
            "tensor": InterfaceVTK.__tensor_mesh_to_vtk,
            "curv": InterfaceVTK.__curvilinear_mesh_to_vtk,
            "simplex": InterfaceVTK.__simplex_mesh_to_vtk,
            "cyl": InterfaceVTK.__cyl_mesh_to_vtk,
        }
        key = mesh._meshType.lower()
        try:
            convert = converters[key]
        except KeyError:
            raise RuntimeError(
                "Mesh type `{}` is not currently supported for VTK conversion.".format(
                    key
                )
            )
        # Convert the data object then attempt a wrapping with `pyvista`
        cvtd = convert(mesh, models=models)
        try:
            import pyvista

            cvtd = pyvista.wrap(cvtd)
        except ImportError:
            warnings.warn(
                "For easier use of VTK objects, you should install `pyvista` (the VTK interface): pip install pyvista"
            )
        return cvtd

    def toVTK(mesh, models=None):
        """Convert mesh (and models) to corresponding VTK or PyVista data object.

        *toVTK* has been deprecated and replaced by *to_vtk*.

        See Also
        --------
        to_vtk
        """
        warnings.warn(
            "Deprecation Warning: `toVTK` is deprecated, use `to_vtk` instead",
            category=FutureWarning,
        )
        return InterfaceVTK.to_vtk(mesh, models=models)

    @staticmethod
    def _save_unstructured_grid(file_name, vtkUnstructGrid, directory=""):
        """Save an unstructured grid to a vtk file.

        Saves a VTK unstructured grid file (vtu) for an already generated
        :class:`pyvista.UnstructuredGrid` object.

        Parameters
        ----------
        file_name : str
            path to the output vtk file or just its name if directory is specified
        directory : str
            directory where the UBC GIF file lives
        """
        _vtk, _, extra = load_vtk(("VTK_VERSION", "vtkXMLUnstructuredGridWriter"))
        _vtk_version, _vtkUnstWriter = extra

        if not isinstance(vtkUnstructGrid, _vtk.vtkUnstructuredGrid):
            raise RuntimeError(
                "`_save_unstructured_grid` can only handle `vtkUnstructuredGrid` objects. `%s` is not supported."
                % vtkUnstructGrid.__class__
            )
        # Check the extension of the file_name
        fname = os.path.join(directory, file_name)
        ext = os.path.splitext(fname)[1]
        if ext == "":
            fname = fname + ".vtu"
        elif ext not in ".vtu":
            raise IOError("{:s} is an incorrect extension, has to be .vtu".format(ext))
        # Make the writer
        vtuWriteFilter = _vtkUnstWriter()
        if float(_vtk_version.split(".")[0]) >= 6:
            vtuWriteFilter.SetInputDataObject(vtkUnstructGrid)
        else:
            vtuWriteFilter.SetInput(vtkUnstructGrid)
        vtuWriteFilter.SetFileName(fname)
        # Write the file
        vtuWriteFilter.Update()

    @staticmethod
    def _save_structured_grid(file_name, vtkStructGrid, directory=""):
        """Save a structured grid to a vtk file.

        Saves a VTK structured grid file (vtk) for an already generated
        :class:`pyvista.StructuredGrid` object.

        Parameters
        ----------
        file_name : str
            path to the output vtk file or just its name if directory is specified
        directory : str
            directory where the UBC GIF file lives
        """
        _vtk, _, extra = load_vtk(("VTK_VERSION", "vtkXMLStructuredGridWriter"))
        _vtk_version, _vtkStrucWriter = extra

        if not isinstance(vtkStructGrid, _vtk.vtkStructuredGrid):
            raise RuntimeError(
                "`_save_structured_grid` can only handle `vtkStructuredGrid` objects. `{}` is not supported.".format(
                    vtkStructGrid.__class__
                )
            )
        # Check the extension of the file_name
        fname = os.path.join(directory, file_name)
        ext = os.path.splitext(fname)[1]
        if ext == "":
            fname = fname + ".vts"
        elif ext not in ".vts":
            raise IOError("{:s} is an incorrect extension, has to be .vts".format(ext))
        # Make the writer
        writer = _vtkStrucWriter()
        if float(_vtk_version.split(".")[0]) >= 6:
            writer.SetInputDataObject(vtkStructGrid)
        else:
            writer.SetInput(vtkStructGrid)
        writer.SetFileName(fname)
        # Write the file
        writer.Update()

    @staticmethod
    def _save_rectilinear_grid(file_name, vtkRectGrid, directory=""):
        """Save a rectilinear grid to a vtk file.

        Saves a VTK rectilinear file (vtr) ffor an already generated
        :class:`pyvista.RectilinearGrid` object.

        Parameters
        ----------
        file_name : str
            path to the output vtk file or just its name if directory is specified
        directory : str
            directory where the UBC GIF file lives
        """
        _vtk, _, _vtkRectWriter = load_vtk("vtkXMLRectilinearGridWriter")

        if not isinstance(vtkRectGrid, _vtk.vtkRectilinearGrid):
            raise RuntimeError(
                "`_save_rectilinear_grid` can only handle `vtkRectilinearGrid` objects. `{}` is not supported.".format(
                    vtkRectGrid.__class__
                )
            )
        # Check the extension of the file_name
        fname = os.path.join(directory, file_name)
        ext = os.path.splitext(fname)[1]
        if ext == "":
            fname = fname + ".vtr"
        elif ext not in ".vtr":
            raise IOError("{:s} is an incorrect extension, has to be .vtr".format(ext))
        # Write the file.
        vtrWriteFilter = _vtkRectWriter()
        vtrWriteFilter.SetInputDataObject(vtkRectGrid)
        vtrWriteFilter.SetFileName(fname)
        vtrWriteFilter.Update()

    def write_vtk(mesh, file_name, models=None, directory=""):
        """Convert mesh (and models) to corresponding VTK or PyVista data object then writes to file.

        This method converts a ``discretize`` mesh (and associated models) to its
        corresponding `VTK <https://vtk.org/doc/nightly/html/index.html>`__ or
        `PyVista <https://docs.pyvista.org/>`__ data object, then writes to file.
        The output structure will be one of: ``vtkUnstructuredGrid``,
        ``vtkRectilinearGrid`` or ``vtkStructuredGrid``.

        Parameters
        ----------
        file_name : str or file name
            Full path for the output file or just its name if directory is specified
        models : dict of [str, (n_cells) numpy.ndarray], optional
            Models are supplied as a dictionary where the keys are the model
            names. Each model is a 1D :class:`numpy.ndarray` of size (n_cells).
        directory : str, optional
            output directory

        Returns
        -------
        str
            The output of Python's *write* function
        """
        vtkObj = InterfaceVTK.to_vtk(mesh, models=models)
        writers = {
            "vtkUnstructuredGrid": InterfaceVTK._save_unstructured_grid,
            "vtkRectilinearGrid": InterfaceVTK._save_rectilinear_grid,
            "vtkStructuredGrid": InterfaceVTK._save_structured_grid,
        }
        key = vtkObj.GetClassName()
        try:
            write = writers[key]
        except KeyError:
            raise RuntimeError("VTK data type `%s` is not currently supported." % key)
        return write(file_name, vtkObj, directory=directory)

    def writeVTK(mesh, file_name, models=None, directory=""):
        """Convert mesh (and models) to corresponding VTK or PyVista data object then writes to file.

        *writeVTK* has been deprecated and replaced by *write_vtk*

        See Also
        --------
        write_vtk
        """
        warnings.warn(
            "Deprecation Warning: `writeVTK` is deprecated, use `write_vtk` instead",
            category=FutureWarning,
        )
        return InterfaceVTK.write_vtk(
            mesh, file_name, models=models, directory=directory
        )


class InterfaceTensorread_vtk(object):
    """Mixin class for converting vtk to TensorMesh.

    This class provides convenient methods for converting VTK Rectilinear Grid
    files/objects to :class:`~discretize.TensorMesh` objects.
    """

    @classmethod
    def vtk_to_tensor_mesh(TensorMesh, vtrGrid):
        """Convert vtk object to a TensorMesh.

        Convert ``vtkRectilinearGrid`` or :class:`~pyvista.RectilinearGrid` object
        to a :class:`~discretize.TensorMesh` object.

        Parameters
        ----------
        vtuGrid : ``vtkRectilinearGrid`` or :class:`~pyvista.RectilinearGrid`
            A VTK or PyVista rectilinear grid object

        Returns
        -------
        discretize.TensorMesh
            A discretize tensor mesh
        """
        _, _nps = load_vtk()

        # Sort information
        hx = np.abs(np.diff(_nps.vtk_to_numpy(vtrGrid.GetXCoordinates())))
        xR = _nps.vtk_to_numpy(vtrGrid.GetXCoordinates())[0]
        hy = np.abs(np.diff(_nps.vtk_to_numpy(vtrGrid.GetYCoordinates())))
        yR = _nps.vtk_to_numpy(vtrGrid.GetYCoordinates())[0]
        zD = np.diff(_nps.vtk_to_numpy(vtrGrid.GetZCoordinates()))
        # Check the direction of hz
        if np.all(zD < 0):
            hz = np.abs(zD[::-1])
            zR = _nps.vtk_to_numpy(vtrGrid.GetZCoordinates())[-1]
        else:
            hz = np.abs(zD)
            zR = _nps.vtk_to_numpy(vtrGrid.GetZCoordinates())[0]
        origin = np.array([xR, yR, zR])

        # Make the object
        tensMsh = TensorMesh([hx, hy, hz], origin=origin)

        # Grap the models
        models = {}
        for i in np.arange(vtrGrid.GetCellData().GetNumberOfArrays()):
            modelName = vtrGrid.GetCellData().GetArrayName(i)
            if np.all(zD < 0):
                modFlip = _nps.vtk_to_numpy(vtrGrid.GetCellData().GetArray(i))
                tM = tensMsh.reshape(modFlip, "CC", "CC", "M")
                modArr = tensMsh.reshape(tM[:, :, ::-1], "CC", "CC", "V")
            else:
                modArr = _nps.vtk_to_numpy(vtrGrid.GetCellData().GetArray(i))
            models[modelName] = modArr

        # Return the data
        return tensMsh, models

    @classmethod
    def read_vtk(TensorMesh, file_name, directory=""):
        """Read VTK rectilinear file (vtr or xml) and return a discretize tensor mesh (and models).

        This method reads a VTK rectilinear file (vtr or xml format) and returns
        a tuple containing the :class:`~discretize.TensorMesh` as well as a dictionary
        containing any models. The keys in the model dictionary correspond to the file
        names of the models.

        Parameters
        ----------
        file_name : str
            full path to the VTK rectilinear file (vtr or xml) containing the mesh (and
            models) or just the file name if the directory is specified.
        directory : str, optional
            directory where the file lives

        Returns
        -------
        mesh : discretize.TensorMesh
            The tensor mesh object.
        model_dict : dict of [str, (n_cells) numpy.ndarray]
            A dictionary containing the models. The keys correspond to the names of the
            models.
        """
        _, _, _vtkRectReader = load_vtk("vtkXMLRectilinearGridReader")
        fname = os.path.join(directory, file_name)
        # Read the file
        vtrReader = _vtkRectReader()
        vtrReader.SetFileName(fname)
        vtrReader.Update()
        vtrGrid = vtrReader.GetOutput()
        return TensorMesh.vtk_to_tensor_mesh(vtrGrid)

    @classmethod
    def readVTK(TensorMesh, file_name, directory=""):
        """Read VTK rectilinear file (vtr or xml) and return a discretize tensor mesh (and models).

        *readVTK* has been deprecated and replaced by *read_vtk*

        See Also
        --------
        read_vtk
        """
        warnings.warn(
            "Deprecation Warning: `readVTK` is deprecated, use `read_vtk` instead",
            category=FutureWarning,
        )
        return InterfaceTensorread_vtk.read_vtk(
            TensorMesh, file_name, directory=directory
        )


class InterfaceSimplexReadVTK:
    """Mixin class for converting vtk to SimplexMesh.

    This class provides convenient methods for converting VTK Unstructured Grid
    files/objects to :class:`~discretize.SimplexMesh` objects.
    """

    @classmethod
    def vtk_to_simplex_mesh(SimplexMesh, vtuGrid):
        """Convert an unstructured grid of simplices to a SimplexMesh.

        Convert ``vtkUnstructuredGrid`` or :class:`~pyvista.UnstructuredGrid` object
        to a :class:`~discretize.SimplexMesh` object.

        Parameters
        ----------
        vtrGrid : ``vtkUnstructuredGrid`` or :class:`~pyvista.UnstructuredGrid`
            A VTK or PyVista unstructured grid object

        Returns
        -------
        discretize.SimplexMesh
            A discretize tensor mesh
        """
        _, _nps = load_vtk()

        # check if all of the cells are the same type
        cell_types = np.unique(_nps.vtk_to_numpy(vtuGrid.GetCellTypesArray()))
        if len(cell_types) > 1:
            raise ValueError(
                "Incompatible unstructured grid. All cell's must have the same type."
            )
        if cell_types[0] not in [5, 10]:
            raise ValueError("Cell types must be either triangular or tetrahedral")
        if cell_types[0] == 5:
            dim = 2
        else:
            dim = 3
        # then should be safe to move forward
        simplices = _nps.vtk_to_numpy(
            vtuGrid.GetCells().GetConnectivityArray()
        ).reshape(-1, dim + 1)
        points = _nps.vtk_to_numpy(vtuGrid.GetPoints().GetData())
        if dim == 2:
            points = points[:, :-1]

        mesh = SimplexMesh(points, simplices)

        # Grap the models
        models = {}
        for i in np.arange(vtuGrid.GetCellData().GetNumberOfArrays()):
            modelName = vtuGrid.GetCellData().GetArrayName(i)
            modArr = _nps.vtk_to_numpy(vtuGrid.GetCellData().GetArray(i))
            models[modelName] = modArr

        # Return the data
        return mesh, models

    @classmethod
    def read_vtk(SimplexMesh, file_name, directory=""):
        """Read VTK unstructured file (vtu or xml) and return a discretize simplex mesh (and models).

        This method reads a VTK unstructured file (vtu or xml format) and returns
        the :class:`~discretize.SimplexMesh` as well as a dictionary containing any
        models. The keys in the model dictionary correspond to the file names of the
        models.

        Parameters
        ----------
        file_name : str
            full path to the VTK unstructured file (vtr or xml) containing the mesh (and
            models) or just the file name if the directory is specified.
        directory : str, optional
            directory where the file lives

        Returns
        -------
        mesh : discretize.SimplexMesh
            The tensor mesh object.
        model_dict : dict of [str, (n_cells) numpy.ndarray]
            A dictionary containing the models. The keys correspond to the names of the
            models.
        """
        _, _, _vtkUnstReader = load_vtk("vtkXMLUnstructuredGridReader")

        fname = os.path.join(directory, file_name)
        # Read the file
        vtuReader = _vtkUnstReader()
        vtuReader.SetFileName(fname)
        vtuReader.Update()
        vtuGrid = vtuReader.GetOutput()
        return SimplexMesh.vtk_to_simplex_mesh(vtuGrid)
