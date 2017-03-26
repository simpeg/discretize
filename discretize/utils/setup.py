import os
import os.path
base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('utils', parent_package, top_path)

    ext = 'interputils_cython'
    ext2 = 'prodAvec'
    try:
        from Cython.Build import cythonize
        cythonize(os.path.join(base_path, ext+'.pyx'))
        cythonize(os.path.join(base_path, ext2+'.pyx'))
    except ImportError:
        pass

    config.add_extension(
        ext,
        sources=[ext+'.c'],
        include_dirs=[get_numpy_include_dirs()]
    )
    config.add_extension(
        ext2,
        sources=[ext2+'.c'],
        include_dirs=[get_numpy_include_dirs()]
    )

    return config
