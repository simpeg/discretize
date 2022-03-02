#!/usr/bin/env python

"""discretize

Discretization tools for finite volume and inverse problems.
"""

import os

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools_scm import get_version
from Cython.Build import cythonize
import numpy as np

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Natural Language :: English",
]

with open("README.rst") as f:
    LONG_DESCRIPTION = "".join(f.readlines())

ext_kwargs = {}
if os.environ.get("DISC_COV", None) is not None:
    ext_kwargs["define_macros"] = [("CYTHON_TRACE_NOGIL", 1)]

extensions = [
    Extension(
        "discretize._extensions.interputils_cython",
        ["discretize/_extensions/interputils_cython.pyx"],
        include_dirs=[np.get_include()],
        **ext_kwargs
    ),
    Extension(
        "discretize._extensions.tree_ext",
        ["discretize/_extensions/tree_ext.pyx", "discretize/_extensions/tree.cpp"],
        include_dirs=[np.get_include()],
        **ext_kwargs
    )
]

build_requires = [
    "numpy>=1.8",
    "cython>=0.2",
    "setuptools_scm",
]

install_requires = [
    "numpy>=1.8",
    "scipy>=0.13",
]

# scm_version = {
#     "root": ".",
#     "relative_to": __file__,
#     "write_to": os.path.join("discretize", "version.py"),
# }

metadata = dict(
    name="discretize",
    version=get_version(),
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    python_requires=">=3.6",
    setup_requires=build_requires,
    install_requires=install_requires,
    author="SimPEG developers",
    author_email="rowanc1@gmail.com",
    description="Discretization tools for finite volume and inverse problems",
    long_description=LONG_DESCRIPTION,
    license="MIT",
    keywords="finite volume, discretization, pde, ode",
    url="http://simpeg.xyz/",
    download_url="http://github.com/simpeg/discretize",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    #use_scm_version=scm_version,
)

setup(**metadata)
