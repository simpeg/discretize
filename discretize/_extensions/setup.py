import os
import os.path

base_path = os.path.abspath(os.path.dirname(__file__))

# Enable line tracing for coverage of cython files conditionally
ext_kwargs = {}
if os.environ.get("DISC_COV", None) is not None:
    ext_kwargs["define_macros"] = [("CYTHON_TRACE_NOGIL", 1)]

def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration("_extensions", parent_package, top_path)

    ext = "tree_ext"
    try:
        from Cython.Build import cythonize

        cythonize(os.path.join(base_path, ext + ".pyx"))
    except ImportError:
        pass

    config.add_extension(
        ext, sources=[ext + ".cpp", "tree.cpp"], include_dirs=[get_numpy_include_dirs()], **ext_kwargs
    )

    ext = "interputils_cython"
    try:
        from Cython.Build import cythonize

        cythonize(os.path.join(base_path, ext + ".pyx"))
    except ImportError:
        pass

    config.add_extension(
        ext, sources=[ext + ".c"], include_dirs=[get_numpy_include_dirs()], **ext_kwargs
    )

    return config
