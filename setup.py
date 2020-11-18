#!/usr/bin/env python

"""discretize

Discretization tools for finite volume and inverse problems.
"""

import os
import sys


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


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True,
    )

    config.add_subpackage("discretize")

    return config


with open("README.rst") as f:
    LONG_DESCRIPTION = "".join(f.readlines())

build_requires = [
    "numpy>=1.8",
    "cython>=0.2",
]

install_requires = build_requires + [
    "scipy>=0.13",
    "properties",
    "vectormath",
]

metadata = dict(
    name="discretize",
    version="0.6.1",
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
)

if len(sys.argv) >= 2 and (
    "--help" in sys.argv[1:]
    or sys.argv[1] in ("--help-commands", "egg_info", "--version", "clean")
):
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
    if (len(sys.argv) >= 2 and sys.argv[1] in ("bdist_wheel", "bdist_egg")) or (
        "develop" in sys.argv
    ):
        # bdist_wheel/bdist_egg needs setuptools
        import setuptools

    from numpy.distutils.core import setup

    # Add the configuration to the setup dict when building
    # after numpy is installed
    metadata["configuration"] = configuration


setup(**metadata)
