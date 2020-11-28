from discretize.utils.code_utils import is_scalar, as_array_n_by_dim, requires
from discretize.utils.matrix_utils import (
    mkvc,
    sdiag,
    sdinv,
    speye,
    kron3,
    spzeros,
    ddx,
    av,
    av_extrap,
    ndgrid,
    ind2sub,
    sub2ind,
    get_subarray,
    inverse_3x3_block_diagonal,
    inverse_2x2_block_diagonal,
    TensorType,
    make_property_tensor,
    inverse_property_tensor,
    Zero,
    Identity,
)
from discretize.utils.mesh_utils import (
    example_curvilinear_grid,
    unpack_widths,
    closest_points_index,
    extract_core_mesh,
    random_model,
    refine_tree_xyz,
    active_from_xyz,
    mesh_builder_xyz,
)
from discretize.utils.curvilinear_utils import volume_tetrahedron, face_info, index_cube
from discretize.utils.interpolation_utils import interpolation_matrix, volume_average
from discretize.utils.coordinate_utils import (
    rotate_points_from_normals,
    rotation_matrix_from_normals,
    cyl2cart,
    cart2cyl,
    cylindrical_to_cartesian,
    cartesian_to_cylindrical,
    # rotate_vec_cyl2cart
)

from discretize.utils.io_utils import download

# DEPRECATIONS
from discretize.utils.code_utils import isScalar, asArray_N_x_Dim
from discretize.utils.matrix_utils import (
    sdInv,
    getSubArray,
    inv3X3BlockDiagonal,
    inv2X2BlockDiagonal,
    makePropertyTensor,
    invPropertyTensor,
)
from discretize.utils.mesh_utils import (
    exampleLrmGrid,
    meshTensor,
    closestPoints,
    ExtractCoreMesh,
)
from discretize.utils.curvilinear_utils import volTetra, indexCube, faceInfo
from discretize.utils.interpolation_utils import interpmat
from discretize.utils.coordinate_utils import (
    rotationMatrixFromNormals,
    rotatePointsFromNormals,
)
