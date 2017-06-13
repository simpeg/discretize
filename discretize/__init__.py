from discretize.BaseMesh import BaseMesh
from discretize.TensorMesh import TensorMesh
from discretize.CylMesh import CylMesh
from discretize.CurvilinearMesh import CurvilinearMesh
from discretize import Tests

try:
    from discretize.TreeMesh import TreeMesh
except ImportError:
    print(
        """
        TreeMesh not imported. You need to run:

        python setup.py install

        to build the TreeMesh cython code.
        """
    )

__version__   = '0.1.7'
__author__    = 'SimPEG Team'
__license__   = 'MIT'
__copyright__ = '2013 - 2017, SimPEG Developers, http://simpeg.xyz'
