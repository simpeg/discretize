from discretize.tests import *
from discretize.utils.code_utils import deprecate_module

# note this needs to be a module with an __init__ so we can avoid name clash
# with tests.py in the discretize directory on systems that are agnostic to Case.
deprecate_module("discretize.Tests", "discretize.tests", removal_version="1.0.0")
