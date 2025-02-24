"""
Test safeguards against accessing certain properties on non-finalized TreeMesh.
"""

import re
import pytest

from discretize import TreeMesh


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
        msg = re.escape(
            f"The `{prop_name}` property cannot be accessed before "
            "the mesh is finalized."
        )
        with pytest.raises(AttributeError, match=msg):
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
