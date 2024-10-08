# NumPy include directory
# The try-except is needed because when things are
# split across drives on Windows, there is no relative path and an exception
# gets raised. There may be other such cases, so add a catch-all and switch to
# an absolute path. Relative paths are needed when for example a virtualenv is
# placed inside the source tree; Meson rejects absolute paths to places inside
# the source tree.
# For cross-compilation it is often not possible to run the Python interpreter
# in order to retrieve numpy's include directory. It can be specified in the
# cross file instead:
#   [properties]
#   numpy-include-dir = /abspath/to/host-pythons/site-packages/numpy/core/include
#
# This uses the path as is, and avoids running the interpreter.
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(py,
    [
      '-c',
      '''import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
  '''
    ],
    check: true
  ).stdout().strip()
else
  _incdir_numpy_abs = incdir_numpy
endif
inc_np = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: inc_np)

# Deal with M_PI & friends; add `use_math_defines` to c_args or cpp_args
# Cython doesn't always get this right itself (see, e.g., gh-16800), so
# explicitly add the define as a compiler flag for Cython-generated code.
is_windows = host_machine.system() == 'windows'
if is_windows
  use_math_defines = ['-D_USE_MATH_DEFINES']
else
  use_math_defines = []
endif

numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'
c_undefined_ok = ['-Wno-maybe-uninitialized']

cython_c_args = [numpy_nodepr_api, use_math_defines]

cy_line_trace = get_option('cy_line_trace')
if cy_line_trace
  cython_c_args += ['-DCYTHON_TRACE_NOGIL=1']
endif

cython_cpp_args = cython_c_args

module_path = 'discretize/_extensions'

py.extension_module(
    'interputils_cython',
    'interputils_cython.pyx',
    include_directories: incdir_numpy,
    c_args: cython_c_args,
    install: true,
    subdir: module_path,
    dependencies : [py_dep, np_dep],
)

py.extension_module(
    'tree_ext',
    ['tree_ext.pyx' , 'tree.cpp', 'geom.cpp'],
    include_directories: incdir_numpy,
    cpp_args: cython_cpp_args,
    install: true,
    subdir: module_path,
    dependencies : [py_dep, np_dep],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'simplex_helpers',
    'simplex_helpers.pyx',
    include_directories: incdir_numpy,
    cpp_args: cython_cpp_args,
    install: true,
    subdir: module_path,
    dependencies : [py_dep, np_dep],
    override_options : ['cython_language=cpp'],
)

python_sources = [
  '__init__.py',
]

py.install_sources(
  python_sources,
  subdir: module_path
)