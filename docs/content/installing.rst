.. _api_installing:

Installing
**********

Which Python?
=============

Currently, ``discretize`` is tested on python 3.8, 3.9, 3.10, and 3.11. We recommend that you
use the latest version of Python available on `Anaconda <https://www.anaconda.com/download>`_.

Installing Python
-----------------

Python is available on all major operating systems, but if you are getting started with python
it is best to use a package manager such as
`Anaconda <https://www.anaconda.com/download>`_.
You can download the package manager and use it to install the dependencies above.

.. note::
    When using Continuum Anaconda, make sure to run::

        conda update conda
        conda update anaconda

Dependencies
============

``discretize``'s runtime requirements are:

- `numpy <http://www.numpy.org>`_ 1.22.4 (or greater)
- `scipy <https://docs.scipy.org/doc/scipy/reference>`_ 1.8 (or greater)

Additional functionality is provided when the following optional packages
are installed:

- `matplotlib <https://matplotlib.org/>`_
- `pyvista <https://pyvista.org/>`_
- `vtk <https://vtk.org/>`_
- `omf <https://omf.readthedocs.io/en/latest/>`_

We also recommend installing:

- `pymatsolver <https://pymatsolver.readthedocs.io/en/latest/>`_ 0.1.2 (or greater)

Installing discretize
=====================

``discretize`` is available on conda-forge and using the ``conda`` (or ``mamba``) package manager
is our recommended way of installing `discretize``::

    conda install -c conda-forge discretize

``discretize`` is also available on pypi::

    pip install discretize

There are currently pre-built wheels for windows available on pypi, but other operating
systems will require a build.

Installing from Source
----------------------
.. attention::
    Install ``discretize`` from the source code only if you need to run the development version. Otherwise it's usually better to install it from ``conda-forge``.

As ``discretize`` contains several compiled extensions and is not a pure python pacakge,
installing ``discretize`` from the source code requires a C/C++ compiler capable of
using a C++ 17 standard.

``discretize`` uses a ``pyproject.toml`` file to define the build and install steps. As such
there is no ``setup.py`` file to run. You must use ``pip`` to install ``discretize``. As long as
you have an available compiler you should be able to install ``discretize`` from the source as::

    pip install .

Editable Installs
^^^^^^^^^^^^^^^^^
If you are an active developer of ``discretize``, and find yourself modifying the code often,
you might want to install it from source, in an editable installation. ``discretize`` uses
``meson-python`` to build the external modules and install the package. As such, there are a few extra
steps to take. First, make sure you have the runtime dependencies installed in your environment (see Dependencies listed above).
Then you must install some packages needed to build ``discretize`` in your environment. You can do so with ``pip``::

    pip install meson-python meson ninja cython setuptools_scm

Or with ``conda`` (or ``mamba``)::

    conda install -c conda-forge meson-python meson ninja cython setuptools_scm

This will allow you to use the build backend required by `discretize`.

Finally, you should then be able to perform an editable install using the source code::

    pip install --no-build-isolation --editable .


This builds and installs the local directory to your active python environment in an
"editable" mode; when source code is changed, you will be able to make use of it immediately. It also builds against the packages installed
in your environment instead of creating and isolated environment to build a wheel for the package.

Testing your installation
=========================

Head over to the :ref:`sphx_glr_examples` and download and run any of the notebooks or python scripts.
