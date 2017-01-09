#!/usr/bin/env python
from __future__ import print_function
"""discretize

Discretization tools for finite volume and inverse problems.
"""

from setuptools import find_packages
from distutils.core import setup

import os
import sys
import numpy

if 'cython' in sys.argv:
    del sys.argv[sys.argv.index('cython')]  # delete the command
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    USE_CYTHON = True
else:
    from setuptools.command.build_ext import build_ext
    from distutils.extension import Extension
    USE_CYTHON = False


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

ext = '.pyx' if USE_CYTHON else '.c'

with open("README.rst") as f:
    LONG_DESCRIPTION = ''.join(f.readlines())

cython_files = [
    "discretize/utils/interputils_cython".replace('/', os.sep),
    "discretize/TreeUtils".replace('/', os.sep)
]

scripts = [s + '.pyx' for s in cython_files] + [s + '.c' for s in cython_files]

if USE_CYTHON:
    extensions = cythonize([s + '.pyx' for s in cython_files])
else:
    extensions = [Extension(cf, [cf+ext]) for cf in cython_files]


class NumpyBuild(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())


setup(
    name="discretize",
    version="0.1.1",
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
    ext_modules=extensions if not USE_CYTHON else cythonize(extensions),
    scripts=scripts,
    cmdclass={'build_ext': NumpyBuild},
    setup_requires=['numpy']
)
