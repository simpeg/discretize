discretize.CurvilinearMesh
==========================

.. currentmodule:: discretize

.. autoclass:: CurvilinearMesh
    :show-inheritance:

.. include:: backreferences/discretize.CurvilinearMesh.examples

.. raw:: html

     <div style='clear:both'></div>

Attributes
^^^^^^^^^^


.. autoattribute:: CurvilinearMesh.area

.. autoattribute:: CurvilinearMesh.average_cell_to_face

.. autoattribute:: CurvilinearMesh.average_cell_vector_to_face

.. autoattribute:: CurvilinearMesh.average_edge_to_cell

.. autoattribute:: CurvilinearMesh.average_edge_to_cell_vector

.. autoattribute:: CurvilinearMesh.average_edge_x_to_cell

.. autoattribute:: CurvilinearMesh.average_edge_y_to_cell

.. autoattribute:: CurvilinearMesh.average_edge_z_to_cell

.. autoattribute:: CurvilinearMesh.average_face_to_cell

.. autoattribute:: CurvilinearMesh.average_face_to_cell_vector

.. autoattribute:: CurvilinearMesh.average_face_x_to_cell

.. autoattribute:: CurvilinearMesh.average_face_y_to_cell

.. autoattribute:: CurvilinearMesh.average_face_z_to_cell

.. autoattribute:: CurvilinearMesh.average_node_to_cell

.. autoattribute:: CurvilinearMesh.average_node_to_edge

.. autoattribute:: CurvilinearMesh.average_node_to_face

.. autoattribute:: CurvilinearMesh.axis_u

.. autoattribute:: CurvilinearMesh.axis_v

.. autoattribute:: CurvilinearMesh.axis_w

.. autoattribute:: CurvilinearMesh.cellGrad

.. autoattribute:: CurvilinearMesh.cellGradBC

.. autoattribute:: CurvilinearMesh.cellGradx

.. autoattribute:: CurvilinearMesh.cellGrady

.. autoattribute:: CurvilinearMesh.cellGradz

.. autoattribute:: CurvilinearMesh.cell_centers

.. autoattribute:: CurvilinearMesh.cell_gradient

.. autoattribute:: CurvilinearMesh.cell_gradient_BC

.. autoattribute:: CurvilinearMesh.cell_gradient_x

.. autoattribute:: CurvilinearMesh.cell_gradient_y

.. autoattribute:: CurvilinearMesh.cell_gradient_z

.. autoattribute:: CurvilinearMesh.cell_volumes

.. autoattribute:: CurvilinearMesh.dim

.. autoattribute:: CurvilinearMesh.edge

.. autoattribute:: CurvilinearMesh.edgeCurl

.. autoattribute:: CurvilinearMesh.edge_curl

.. autoattribute:: CurvilinearMesh.edge_lengths

.. autoattribute:: CurvilinearMesh.edge_tangents

.. autoattribute:: CurvilinearMesh.edges_x

.. autoattribute:: CurvilinearMesh.edges_y

.. autoattribute:: CurvilinearMesh.edges_z

.. autoattribute:: CurvilinearMesh.faceDiv

.. autoattribute:: CurvilinearMesh.faceDivx

.. autoattribute:: CurvilinearMesh.faceDivy

.. autoattribute:: CurvilinearMesh.faceDivz

.. autoattribute:: CurvilinearMesh.face_areas

.. autoattribute:: CurvilinearMesh.face_divergence

.. autoattribute:: CurvilinearMesh.face_normals

.. autoattribute:: CurvilinearMesh.face_x_divergence

.. autoattribute:: CurvilinearMesh.face_y_divergence

.. autoattribute:: CurvilinearMesh.face_z_divergence

.. autoattribute:: CurvilinearMesh.faces_x

.. autoattribute:: CurvilinearMesh.faces_y

.. autoattribute:: CurvilinearMesh.faces_z

.. autoattribute:: CurvilinearMesh.nCx

.. autoattribute:: CurvilinearMesh.nCy

.. autoattribute:: CurvilinearMesh.nCz

.. autoattribute:: CurvilinearMesh.nNx

.. autoattribute:: CurvilinearMesh.nNy

.. autoattribute:: CurvilinearMesh.nNz

.. autoattribute:: CurvilinearMesh.n_cells

.. autoattribute:: CurvilinearMesh.n_edges

.. autoattribute:: CurvilinearMesh.n_edges_per_direction

.. autoattribute:: CurvilinearMesh.n_edges_x

.. autoattribute:: CurvilinearMesh.n_edges_y

.. autoattribute:: CurvilinearMesh.n_edges_z

.. autoattribute:: CurvilinearMesh.n_faces

.. autoattribute:: CurvilinearMesh.n_faces_per_direction

.. autoattribute:: CurvilinearMesh.n_faces_x

.. autoattribute:: CurvilinearMesh.n_faces_y

.. autoattribute:: CurvilinearMesh.n_faces_z

.. autoattribute:: CurvilinearMesh.n_nodes

.. autoattribute:: CurvilinearMesh.nodalGrad

.. autoattribute:: CurvilinearMesh.nodalLaplacian

.. autoattribute:: CurvilinearMesh.nodal_gradient

.. autoattribute:: CurvilinearMesh.nodal_laplacian

.. autoattribute:: CurvilinearMesh.node_list

.. autoattribute:: CurvilinearMesh.nodes

.. autoattribute:: CurvilinearMesh.normals

.. autoattribute:: CurvilinearMesh.orientation

.. autoattribute:: CurvilinearMesh.origin

.. autoattribute:: CurvilinearMesh.reference_is_rotated

.. autoattribute:: CurvilinearMesh.reference_system

.. autoattribute:: CurvilinearMesh.rotation_matrix

.. autoattribute:: CurvilinearMesh.shape_cells

.. autoattribute:: CurvilinearMesh.shape_edges_x

.. autoattribute:: CurvilinearMesh.shape_edges_y

.. autoattribute:: CurvilinearMesh.shape_edges_z

.. autoattribute:: CurvilinearMesh.shape_faces_x

.. autoattribute:: CurvilinearMesh.shape_faces_y

.. autoattribute:: CurvilinearMesh.shape_faces_z

.. autoattribute:: CurvilinearMesh.shape_nodes

.. autoattribute:: CurvilinearMesh.stencil_cell_gradient

.. autoattribute:: CurvilinearMesh.stencil_cell_gradient_x

.. autoattribute:: CurvilinearMesh.stencil_cell_gradient_y

.. autoattribute:: CurvilinearMesh.stencil_cell_gradient_z

.. autoattribute:: CurvilinearMesh.tangents

.. autoattribute:: CurvilinearMesh.vol

.. autoattribute:: CurvilinearMesh.x0



Methods
^^^^^^^





.. automethod:: CurvilinearMesh.copy



.. automethod:: CurvilinearMesh.deserialize



.. automethod:: CurvilinearMesh.getBCProjWF



.. automethod:: CurvilinearMesh.getBCProjWF_simple



.. automethod:: CurvilinearMesh.getEdgeInnerProduct



.. automethod:: CurvilinearMesh.getEdgeInnerProductDeriv



.. automethod:: CurvilinearMesh.getFaceInnerProduct



.. automethod:: CurvilinearMesh.getFaceInnerProductDeriv



.. automethod:: CurvilinearMesh.get_BC_projections



.. automethod:: CurvilinearMesh.get_BC_projections_simple



.. automethod:: CurvilinearMesh.get_edge_inner_product



.. automethod:: CurvilinearMesh.get_edge_inner_product_deriv



.. automethod:: CurvilinearMesh.get_face_inner_product



.. automethod:: CurvilinearMesh.get_face_inner_product_deriv



.. automethod:: CurvilinearMesh.plotGrid



.. automethod:: CurvilinearMesh.plotImage



.. automethod:: CurvilinearMesh.plotSlice



.. automethod:: CurvilinearMesh.plot_3d_slicer



.. automethod:: CurvilinearMesh.plot_grid



.. automethod:: CurvilinearMesh.plot_image



.. automethod:: CurvilinearMesh.plot_slice



.. automethod:: CurvilinearMesh.projectEdgeVector



.. automethod:: CurvilinearMesh.projectFaceVector



.. automethod:: CurvilinearMesh.project_edge_vector



.. automethod:: CurvilinearMesh.project_face_vector



.. automethod:: CurvilinearMesh.r



.. automethod:: CurvilinearMesh.reshape



.. automethod:: CurvilinearMesh.save



.. automethod:: CurvilinearMesh.serialize



.. automethod:: CurvilinearMesh.setCellGradBC



.. automethod:: CurvilinearMesh.set_cell_gradient_BC



.. automethod:: CurvilinearMesh.toVTK



.. automethod:: CurvilinearMesh.to_dict



.. automethod:: CurvilinearMesh.to_vtk



.. automethod:: CurvilinearMesh.validate



.. automethod:: CurvilinearMesh.writeVTK



.. automethod:: CurvilinearMesh.write_vtk



.. raw:: html

     <div style='clear:both'></div>