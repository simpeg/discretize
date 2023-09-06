#!/usr/bin/env python

"""discretize

Discretization tools for finite volume and inverse problems.
"""

import os
import sys
import numpy
import platform
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext as _build_ext


# cmdclass['bdist_wheel']
if platform.python_implementation() == 'CPython':
    try:
        import wheel.bdist_wheel
    except:
        pass
    else:
        class bdist_wheel(wheel.bdist_wheel.bdist_wheel):
            def finalize_options(self):
                self.py_limited_api = 'cp3{}'.format(sys.version_info[1])
                super().finalize_options()


# cmdclass['build_ext']
ext_kwargs = {
    "py_limited_api": True,
    "define_macros": [('CYTHON_LIMITED_API', '1')],
}
if os.environ.get("DISC_COV", None) is not None:
    ext_kwargs["define_macros"].append(("CYTHON_TRACE_NOGIL", 1))

extensions = [
        Extension(
            "discretize._extensions.interputils_cython",
            ["src/discretize/_extensions/interputils_cython.pyx"],
            include_dirs=[numpy.get_include()],
            language="c",
            **ext_kwargs
        ),
        Extension(
            "discretize._extensions.simplex_helpers",
            ["src/discretize/_extensions/simplex_helpers.pyx"],
            include_dirs=[numpy.get_include()],
            language="c",
            **ext_kwargs
        ),
        Extension(
            "discretize._extensions.tree_ext",
            ["src/discretize/_extensions/tree_ext.pyx", "src/discretize/_extensions/tree.cpp"],
            include_dirs=[numpy.get_include()],
            language="c++",
            extra_compile_args=['-std=c++17'],
            **ext_kwargs
        )
]

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        import numpy
        self.include_dirs.append(numpy.get_include())


setup_kwargs = {} # see 'pyproject.toml' for other kwargs and settings
setup_kwargs["name"] = "discretize"
setup_kwargs["cmdclass"] = {'bdist_wheel': bdist_wheel,'build_ext': build_ext} 
setup_kwargs["ext_modules"] = cythonize(extensions, compiler_directives={"language_level": 3})

setup(**setup_kwargs)
