from __future__ import print_function

from .code_utils import (is_scalar, as_array_n_by_dim, requires)
from .code_utils import (isScalar, asArray_N_x_Dim)

from .matrix_utils import (
    mkvc, sdiag, sdinv, speye, kron3, spzeros, ddx, av,
    av_extrap, ndgrid, ind2sub, sub2ind, get_subarray,
    inverse_3x3_block_diagonal, inverse_2x2_block_diagonal, TensorType,
    make_property_tensor, inverse_property_tensor, Zero,
    Identity
)
from .matrix_utils import (
    sdInv, getSubArray, inv3X3BlockDiagonal, inv2X2BlockDiagonal,
    makePropertyTensor, invPropertyTensor
)

from .mesh_utils import (
    example_curvilinear_grid, mesh_tensor, closest_points, extract_core_mesh,
    random_model, refine_tree_xyz, active_from_xyz, mesh_builder_xyz
)
from .mesh_utils import (
    exampleLrmGrid, meshTensor, closestPoints, ExtractCoreMesh
)

from .curvilinear_utils import volume_tetrahedron, face_info, index_cube
from .curvilinear_utils import volTetra, indexCube, faceInfo

from .interpolation_utils import interpolation_matrix, volume_average
from .interpolation_utils import interpmat

from .coordinate_utils import (
    rotate_points_from_normals, rotation_matrix_from_normals, cyl2cart, cart2cyl
    # rotate_vec_cyl2cart
)
from .coordinate_utils import (rotationMatrixFromNormals, rotatePointsFromNormals)

from .io_utils import download
