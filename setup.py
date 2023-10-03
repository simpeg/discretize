#!/usr/bin/env python

"""discretize

Discretization tools for finite volume and inverse problems.
"""

import os
import sys

from setuptools import setup, find_packages

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

build_requires = [
    "numpy>=1.20",
    "cython>=0.29.31",
    "setuptools_scm",
]

install_requires = [
    "numpy>=1.20",
    "scipy>=1.8",
]

metadata = dict(
    name="discretize",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.8",
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
    use_scm_version={
        "write_to": os.path.join("src", "discretize", "version.py"),
    },
)
if len(sys.argv) >= 2 and (
    "--help" in sys.argv[1:]
    or sys.argv[1]
    in ("--help-commands", "egg_info", "install_egg_info", "--version", "clean")
):
    # For these actions, build_requires are not required.
    #
    # They are required to succeed without Numpy/Cython, for example when
    # pip is used to install discretize when Numpy/Cython is not yet
    # present in the system.

    # add cython and setuptools_scm to install requires on these commands though
    install_requires = build_requires + install_requires[1:]
    metadata["install_requires"] = install_requires
else:
    from setuptools.extension import Extension
    from setuptools.command.build_ext import build_ext
    from Cython.Build import cythonize
    import numpy as np

    ext_kwargs = {}
    if os.environ.get("DISC_COV", None) is not None:
        ext_kwargs["define_macros"] = [("CYTHON_TRACE_NOGIL", 1)]

    extensions = [
        Extension(
            "discretize._extensions.interputils_cython",
            ["src/discretize/_extensions/interputils_cython.pyx"],
            include_dirs=[np.get_include()],
            **ext_kwargs
        ),
        Extension(
            "discretize._extensions.tree_ext",
            [
                "src/discretize/_extensions/tree_ext.pyx",
                "src/discretize/_extensions/tree.cpp",
            ],
            include_dirs=[np.get_include()],
            **ext_kwargs
        ),
        Extension(
            "discretize._extensions.simplex_helpers",
            ["src/discretize/_extensions/simplex_helpers.pyx"],
            include_dirs=[np.get_include()],
            **ext_kwargs
        ),
    ]

    class build_ext_cpp_standard(build_ext):
        # add compiler specific standard argument specifier
        def build_extension(self, ext):
            # This module requires c++17 standard
            if ext.name == "discretize._extensions.tree_ext":
                comp_type = self.compiler.compiler_type
                if comp_type == "msvc":
                    std_arg = "/std:c++17"
                elif comp_type == "bcpp":
                    raise Exception(
                        "Must use cpp compiler that support C++17 standard."
                    )
                else:
                    std_arg = "-std=c++17"
                ext.extra_compile_args = [
                    std_arg,
                ]
            super().build_extension(ext)

    metadata["ext_modules"] = cythonize(extensions)
    metadata["cmdclass"] = {"build_ext": build_ext_cpp_standard}

setup(**metadata)
