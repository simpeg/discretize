# from discretize.BaseMesh import BaseMesh
from discretize.TensorMesh import TensorMesh
from discretize.CylMesh import CylMesh
from discretize.CurvilinearMesh import CurvilinearMesh
from discretize import Tests
from discretize.MeshIO import load_mesh
try:
    from discretize.TreeMesh import TreeMesh
except ImportError as err:
    print(err)
    import os
    # Check if being called from non-standard location (i.e. a git repository)
    # is tree_ext.cpp here? will not be in the folder if installed to site-packages...
    file_test = os.path.dirname(os.path.abspath(__file__))+"/tree_ext.pyx"
    if os.path.isfile(file_test):
        # Then we are being run from a repository
        print(
            """
            Unable to import TreeMesh.

            It would appear that discretize is being imported from its repository.
            If this is intentional, you need to run:

            python setup.py build_ext --inplace

            to compile the cython code.
            """
            )

__version__   = '0.4.7'
__author__    = 'SimPEG Team'
__license__   = 'MIT'
__copyright__ = '2013 - 2019, SimPEG Developers, http://simpeg.xyz'
