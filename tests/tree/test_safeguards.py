"""
Test safeguards against accessing certain properties on non-finalized TreeMesh.
"""

import re
import pytest

from discretize import TreeMesh
from discretize.tree_mesh import TreeMeshNotFinalizedError


PROPERTIES = [
    "average_cell_to_face",
    "average_cell_to_face_x",
    "average_cell_to_face_y",
    "average_cell_to_face_z",
    "average_cell_vector_to_face",
    "boundary_edges",
    "boundary_face_outward_normals",
    "boundary_faces",
    "boundary_nodes",
    "cell_centers",
    "cell_volumes",
    "edge_lengths",
    "edges_x",
    "edges_y",
    "edges_z",
    "face_areas",
    "faces_x",
    "faces_y",
    "faces_z",
    "h_gridded",
    "hanging_edges_x",
    "hanging_edges_y",
    "hanging_edges_z",
    "hanging_faces_x",
    "hanging_faces_y",
    "hanging_faces_z",
    "hanging_nodes",
    "nodes",
    "project_edge_to_boundary_edge",
    "project_face_to_boundary_face",
    "project_node_to_boundary_node",
    "stencil_cell_gradient_x",
    "stencil_cell_gradient_y",
    "stencil_cell_gradient_z",
    "total_nodes",
]

INHERITED_PROPERTIES = [
    "boundary_edge_vector_integral",
    "boundary_face_scalar_integral",
    "boundary_node_vector_integral",
    "edges",
    "faces",
    "permute_cells",
    "permute_edges",
    "permute_faces",
]


@pytest.fixture
def mesh():
    """Return a sample TreeMesh"""
    nc = 16
    h = [nc, nc, nc]
    origin = (-32.4, 245.4, 192.3)
    mesh = TreeMesh(h, origin, diagonal_balance=True)
    return mesh


def refine_mesh(mesh):
    """
    Refine the sample tree mesh.

    Don't finalize the mesh.
    """
    origin = mesh.origin
    p1 = (origin[0] + 0.4, origin[1] + 0.4, origin[2] + 0.7)
    p2 = (origin[0] + 0.6, origin[1] + 0.6, origin[2] + 0.9)
    mesh.refine_box(p1, p2, levels=5, finalize=False)


class TestSafeGuards:

    @pytest.mark.parametrize("prop_name", PROPERTIES)
    @pytest.mark.parametrize("refine", [True, False], ids=["refined", "non-refined"])
    def test_errors(self, mesh, prop_name, refine):
        """
        Test error after trying to access the property before finalizing the mesh.

        Run tests by accessing the property before and after mesh refinement.
        """
        if refine:
            refine_mesh(mesh)
        msg = re.escape(f"`TreeMesh.{prop_name}` requires a finalized mesh.")
        with pytest.raises(TreeMeshNotFinalizedError, match=msg):
            getattr(mesh, prop_name)

    @pytest.mark.parametrize("prop_name", PROPERTIES)
    def test_no_errors(self, mesh, prop_name):
        """
        Test if no error is raised when accessing the property on finalized mesh.
        """
        # Refine and finalize mesh
        refine_mesh(mesh)
        mesh.finalize()
        # Accessing the property should not error out
        getattr(mesh, prop_name)

    @pytest.mark.parametrize("prop_name", INHERITED_PROPERTIES)
    @pytest.mark.parametrize("refine", [True, False], ids=["refined", "non-refined"])
    def test_errors_inherited_properties(self, mesh, prop_name, refine):
        """
        Test errors when accessing inherited properties before finalizing the mesh.

        These inherited properties are the ones that ``TreeMesh`` inherit, but
        they depend on the ones defined by it that should not be accessed
        before finalizing the mesh (e.g. ``edges`` and ``faces``).
        """
        if refine:
            refine_mesh(mesh)
        msg = r"\`TreeMesh\.[a-z_]+\`" + re.escape(" requires a finalized mesh.")
        with pytest.raises(TreeMeshNotFinalizedError, match=msg):
            getattr(mesh, prop_name)
