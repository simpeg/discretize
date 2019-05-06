# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:55:17 2019

@author: Devin
"""

from discretize import TensorMesh
from pymatsolver import SolverLU
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse as sp
from discretize.utils import sdiag, speye, kron3, spzeros, ddx, av, av_extrap


hx = 10*np.ones(50)
hy = 10*np.ones(50)
mesh = TensorMesh([hx, hy], 'CC')

L = mesh.nodalLaplacian

x = sp.linalg.eigs(L, which='SM')
