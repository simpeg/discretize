def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration("discretize", parent_package, top_path)

    config.add_subpackage("base")
    config.add_subpackage("operators")
    config.add_subpackage("mixins")
    config.add_subpackage("utils")
    config.add_subpackage("_extensions")
    # deprecated
    config.add_subpackage("Tests")

    return config
