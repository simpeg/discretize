import datetime as _datetime

from .BaseMesh import BaseMesh
from .TensorMesh import TensorMesh
from .CylMesh import CylMesh
from .CurvilinearMesh import CurvilinearMesh
from .import Tests
from .MeshIO import load_mesh
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

__version__   = '0.4.0'
__author__    = 'SimPEG Team'
__license__   = 'MIT'
__copyright__ = f"2013 - {_datetime.datetime.now().year}, {__author__}, http://simpeg.xyz"
