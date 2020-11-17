import numpy as np
import warnings

SCALARTYPES = (complex, float, int, np.number)


def is_scalar(f):
    if isinstance(f, SCALARTYPES):
        return True
    elif isinstance(f, np.ndarray) and f.size == 1 and isinstance(f[0], SCALARTYPES):
        return True
    return False


def as_array_n_by_dim(pts, dim):
    if type(pts) == list:
        pts = np.array(pts)
    if not isinstance(pts, np.ndarray):
        raise TypeError("pts must be a numpy array")

    if dim > 1:
        pts = np.atleast_2d(pts)
    elif len(pts.shape) == 1:
        pts = pts[:, np.newaxis]

    if pts.shape[1] != dim:
        raise ValueError(
            "pts must be a column vector of shape (nPts, {0:d}) not ({1:d}, {2:d})".format(
                *((dim,) + pts.shape)
            )
        )

    return pts


def requires(modules):
    """Decorator to wrap functions with soft dependencies.

    This function was inspired by the `requires` function of pysal,
    which is released under the 'BSD 3-Clause "New" or "Revised" License'.

    https://github.com/pysal/pysal/blob/master/pysal/lib/common.py

    Parameters
    ----------
    modules : dict
        Dictionary containing soft dependencies, e.g.,
        {'matplotlib': matplotlib}.

    Returns
    -------
    decorated_function : function
        Original function if all soft dependencies are met, otherwise
        it returns an empty function which prints why it is not running.

    """

    # Check the required modules, add missing ones in the list `missing`.
    missing = []
    for key, item in modules.items():
        if item is False:
            missing.append(key)

    def decorated_function(function):
        """Wrap function."""
        if not missing:
            return function
        else:

            def passer(*args, **kwargs):
                print(("Missing dependencies: {d}.".format(d=missing)))
                print(("Not running `{}`.".format(function.__name__)))

            return passer

    return decorated_function


def deprecate_class(removal_version=None, new_location=None):
    def decorator(cls):
        my_name = cls.__name__
        parent_name = cls.__bases__[0].__name__
        message = f"{my_name} has been deprecated, please use {parent_name}."
        if removal_version is not None:
            message += (
                f" It will be removed in version {removal_version} of discretize."
            )
        else:
            message += " It will be removed in a future version of discretize."

        # stash the original initialization of the class
        cls._old__init__ = cls.__init__

        def __init__(self, *args, **kwargs):
            warnings.warn(message, FutureWarning)
            self._old__init__(*args, **kwargs)

        cls.__init__ = __init__
        if new_location is not None:
            parent_name = f"{new_location}.{parent_name}"
        cls.__doc__ = f""" This class has been deprecated, see `{parent_name}` for documentation"""
        return cls

    return decorator


def deprecate_module(old_name, new_name, removal_version=None):
    message = f"The {old_name} module has been deprecated, please use {new_name}."
    if removal_version is not None:
        message += f" It will be removed in version {removal_version} of discretize"
    else:
        message += " It will be removed in a future version of discretize."
    message += " Please update your code accordingly."
    warnings.warn(message, FutureWarning)


def deprecate_property(new_name, old_name, removal_version=None):
    if removal_version is not None:
        tag = f" It will be removed in version {removal_version} of discretize."
    else:
        tag = " It will be removed in a future version of discretize."

    def get_dep(self):
        class_name = type(self).__name__
        message = (
            f"{class_name}.{old_name} has been deprecated, please use {class_name}.{new_name}."
            + tag
        )
        warnings.warn(message, FutureWarning)
        return getattr(self, new_name)

    def set_dep(self, other):
        class_name = type(self).__name__
        message = (
            f"{class_name}.{old_name} has been deprecated, please use {class_name}.{new_name}."
            + tag
        )
        warnings.warn(message, FutureWarning)
        setattr(self, new_name, other)

    doc = f"`{old_name}` has been deprecated. See `{new_name}` for documentation"

    return property(get_dep, set_dep, None, doc)


def deprecate_method(new_name, old_name, removal_version=None):
    if removal_version is not None:
        tag = f" It will be removed in version {removal_version} of discretize."
    else:
        tag = " It will be removed in a future version of discretize."

    def new_method(self, *args, **kwargs):
        class_name = type(self).__name__
        warnings.warn(
            f"{class_name}.{old_name} has been deprecated, please use {class_name}.{new_name}."
            + tag,
            FutureWarning,
        )
        return getattr(self, new_name)(*args, **kwargs)

    doc = f"`{old_name}` has been deprecated. See `{new_name}` for documentation"
    new_method.__doc__ = doc
    return new_method


def deprecate_function(new_function, old_name, removal_version=None):
    new_name = new_function.__name__
    if removal_version is not None:
        tag = f" It will be removed in version {removal_version} of discretize."
    else:
        tag = " It will be removed in a future version of discretize."

    def dep_function(*args, **kwargs):
        warnings.warn(
            f"{old_name} has been deprecated, please use {new_name}." + tag,
            FutureWarning,
        )
        return new_function(*args, **kwargs)

    doc = f"`{old_name}` has been deprecated. See `{new_name}` for documentation"
    dep_function.__doc__ = doc
    return dep_function


# DEPRECATIONS
isScalar = deprecate_function(is_scalar, "isScalar", removal_version="1.0.0")
asArray_N_x_Dim = deprecate_function(
    as_array_n_by_dim, "asArray_N_x_Dim", removal_version="1.0.0"
)
