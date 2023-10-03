#!/usr/bin/env python

"""discretize

Discretization tools for finite volume and inverse problems.

"""

import os
import sys
import numpy
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext as _build_ext

ext_kwargs = {}
if os.environ.get("DISC_COV", None) is not None:
    ext_kwargs["define_macros"].append(("CYTHON_TRACE_NOGIL", 1))


extensions = [
    Extension(
        name="discretize._extensions.interputils_cython",
        sources=["src/discretize/_extensions/interputils_cython.pyx"],
        include_dirs=[numpy.get_include()],
        language="c",
        **ext_kwargs,
    ),
    Extension(
        name="discretize._extensions.simplex_helpers",
        sources=["src/discretize/_extensions/simplex_helpers.pyx"],
        include_dirs=[numpy.get_include()],
        language="c",
        **ext_kwargs,
    ),
    Extension(
        name="discretize._extensions.tree_ext",
        sources=[
            "src/discretize/_extensions/tree_ext.pyx",
            "src/discretize/_extensions/tree.cpp",
        ],
        include_dirs=[numpy.get_include()],
        language="c++",
        **ext_kwargs,
    ),
]

# cmdclass['bdist_wheel']
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            self.root_is_pure = False
            _bdist_wheel.finalize_options(self)

except ImportError:
    bdist_wheel = None


# cmdclass['build_ext']
class build_ext(_build_ext):
    # add compiler specific standard argument specifier
    def build_extension(self, ext):
        # This module requires c++17 standard
        if ext.name == "discretize._extensions.tree_ext":
            comp_type = self.compiler.compiler_type
            if comp_type == "msvc":
                std_arg = "/std:c++17"
            elif comp_type == "bcpp":
                raise Exception("Must use cpp compiler that support C++17 standard.")
            else:
                std_arg = "-std=c++17"
            ext.extra_compile_args = [
                std_arg,
            ]
        super().build_extension(ext)


setup_kwargs = {}
setup_kwargs["name"] = "discretize"
setup_kwargs["use_scm_version"] = {
    "write_to": os.path.join("src", "discretize", "version.py")
}
setup_kwargs["cmdclass"] = {"bdist_wheel": bdist_wheel, "build_ext": build_ext}
setup_kwargs["ext_modules"] = cythonize(
    extensions, compiler_directives={"language_level": 3}
)

setup(**setup_kwargs)
