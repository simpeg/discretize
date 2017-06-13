#!/usr/bin/env python
from __future__ import print_function
"""discretize

Discretization tools for finite volume and inverse problems.
"""

from setuptools import find_packages

try:
    from numpy.distutils.core import setup
except Exception:
    raise Exception(
        "Install requires numpy. "
        "If you use conda, `conda install numpy` "
        "or you can use pip, `pip install numpy`"
    )

import os
import sys
import numpy


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


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('discretize')

    return config

setup(
    name="discretize",
    version="0.1.7",
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
    setup_requires=['numpy'],
    configuration=configuration
)
