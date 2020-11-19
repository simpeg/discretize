from discretize.tensor_mesh import TensorMesh
from discretize.cylindrical_mesh import CylMesh, CylindricalMesh
from discretize.curvilinear_mesh import CurvilinearMesh
from discretize.base.mesh_io import load_mesh

try:
    from discretize.tree_mesh import TreeMesh
except ImportError as err:
    print(err)
    import os

    # Check if being called from non-standard location (i.e. a git repository)
    # is tree_ext.pyx here? will not be in the folder if installed to site-packages...
    file_test = os.path.dirname(os.path.abspath(__file__)) + "/_extensions/tree_ext.pyx"
    if os.path.isfile(file_test):
        # Then we are being run from a repository
        raise ImportError(
            """
            Unable to import TreeMesh.

            It would appear that discretize is being imported from its repository.
            If this is intentional, you need to run:

            python setup.py build_ext --inplace

            to compile the cython code.
            """
        )
from discretize import tests

__version__ = "0.6.1"
__author__ = "SimPEG Team"
__license__ = "MIT"
__copyright__ = "2013, SimPEG Developers, http://simpeg.xyz"
