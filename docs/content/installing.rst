.. _api_installing:

Installing
**********

Which Python?
=============

Currently, `discretize` will run on Python 2.7, 3.5, 3.6 and 3.7. We recommend that you
use the latest version of Python available on `Anaconda <https://www.anaconda.com/download>`_.

We will be `dropping support for Python 2.7 by 2020 <https://python3statement.org/>`.

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

- `numpy <http://www.numpy.org>`_ 1.8 (or greater)
- `scipy <https://docs.scipy.org/doc/scipy/reference>`_ 0.13 (or greater)
- `matplotlib <https://matplotlib.org>`_ 1.3 (or greater)
- `cython <https://cython.org/>`_ 0.20 (or greater)
- `pymatsolver <https://pymatsolver.readthedocs.io/en/latest/>`_ 0.1.2 (or greater)
- `properties[math] <http://propertiespy.readthedocs.io>`_


Installing discretize
=====================

discretize is on pypi::

    pip install discretize

.. attention:: Windows users

    If the pip install fails, please try installing the most recent version of
    Visual Studio Community from https://visualstudio.microsoft.com/vs/community/

    Within the **Python development** options, ensure that the following are included

        - Cookiecutter template support
        - Python web support
        - Python 3 64-bit

    .. image:: ../images/visual-studio-community.png
        :align: center



Installing from Source
----------------------

First (you need git)::

    git clone https://github.com/simpeg/discretize

Second (from the root of the discretize repository)::

    python setup.py build_ext --inline

This builds the cython extensions. You will also need to add
the discretize directory to your PYTHON_PATH.


Testing your installation
=========================

Head over to the :ref:`examples` and download and run any of the notebooks or python scripts.
