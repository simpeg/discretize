#!/usr/bin/env python
from __future__ import print_function
"""discretize

Discretization tools for finite volume and inverse problems.
"""

import os
import sys


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


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('discretize')

    return config


with open("README.rst") as f:
    LONG_DESCRIPTION = ''.join(f.readlines())

build_requires = [
    'numpy>=1.8',
    'cython>=0.2',
    ]

install_requires = build_requires + [
    'scipy>=0.13',
    'properties',
    'vectormath',
]

metadata = dict(
    name="discretize",
    version="0.5.0",
    python_requires='>=3.6',
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
)

if len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
        sys.argv[1] in ('--help-commands', 'egg_info', '--version',
            'clean')):
        # For these actions, NumPy is not required.
        #
        # They are required to succeed without Numpy, for example when
        # pip is used to install discretize when Numpy is not yet present in
        # the system.
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup
else:
    if (len(sys.argv) >= 2 and sys.argv[1] in ('bdist_wheel', 'bdist_egg')) or (
                'develop' in sys.argv):
        # bdist_wheel/bdist_egg needs setuptools
        import setuptools

    from numpy.distutils.core import setup

    # Add the configuration to the setup dict when building
    # after numpy is installed
    metadata['configuration'] = configuration

    # A Small hack to remove -std=c99 from c++ compiler options (if present)
    # This should only be if numpy 1.18.0 is installed.
    from numpy.distutils.ccompiler import CCompiler_customize, CCompiler
    from numpy.distutils.ccompiler import replace_method
    _np_customize = CCompiler_customize
    def _simpeg_customize(self, dist, need_cxx=0):
        _np_customize(self, dist, need_cxx)
        if need_cxx:
            # Remove -std=c99 option if present
            try:
                self.compiler_so.remove('-std=c99')
            except (AttributeError, ValueError):
                pass
    replace_method(CCompiler, 'customize', _simpeg_customize)


setup(**metadata)
