#!/usr/bin/env python
from __future__ import print_function
"""discretize

Discretization tools for finite volume and inverse problems.
"""

from distutils.core import setup
from setuptools import find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy


class NumpyBuild(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())


CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Natural Language :: English',
]


with open("README.rst") as f:
    LONG_DESCRIPTION = ''.join(f.readlines())

cython_files = [
    "discretize/utils/interputils_cython.pyx",
    "discretize/TreeUtils.pyx"
]

setup(
    name="discretize",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.7',
        'scipy>=0.13',
        'cython',
        'ipython',
        'matplotlib',
        'pymatsolver>=0.1.2',
        'properties[math]'
    ],
    author="Rowan Cockett",
    author_email="rowanc1@gmail.com",
    description="Discretization tools for finite volume and inverse problems",
    long_description=LONG_DESCRIPTION,
    license="MIT",
    keywords="finite volume, discretization, pde, ode",
    url="http://simpeg.xyz/",
    download_url="http://github.com/simpeg/discretize",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3=False,
    setup_requires=['numpy', 'cython'],
    cmdclass={'build_ext': NumpyBuild},
    ext_modules=cythonize(cython_files)
)
