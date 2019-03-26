.. _api_installing:

Getting Started
***************

Dependencies
============

- Python 2.7, 3.4 or 3.5
- NumPy 1.8 (or greater)
- SciPy 0.13 (or greater)
- matplotlib 1.3 (or greater)
- pymatsolver 0.1.2 (or greater)
- Cython 0.20 (or greater)
- properties[math]

Development Dependencies
------------------------
- sphinx
- sphinx_rtd_theme
- sphinx-gallery
- pillow
- nose-cov
- pylint

Installing Python
=================

Python is available on all major operating systems, but if you are getting started with python
it is best to use a package manager such as
`Continuum Anaconda <https://www.anaconda.com/download>`_.
You can download the package manager and use it to install the dependencies above.

.. note::
    When using Continuum Anaconda, make sure to run::

        conda update conda
        conda update anaconda


Installing discretize
=====================

discretize is on pip::

    pip install discretize


Installing from Source
----------------------

First (you need git)::

    git clone https://github.com/simpeg/discretize

Second (from the root of the discretize repository)::

    python setup.py build_ext --inplace

This builds the cython extensions. You will also need to add
the discretize directory to your PYTHON_PATH.

.. attention:: Windows users

	A common error when installing the setup.py is:
	``Missing linker, needs MSC v.1500 (Microsoft Visual C++ 2008) Runtime Library``

	The missing library can be found `here <https://www.microsoft.com/en-ca/download/details.aspx?id=29>`

Useful Links
============
An enormous amount of information (including tutorials and examples) can be found on the official websites of the packages

* `Python Website <https://www.python.org/>`_
* `Numpy Website <http://www.numpy.org/>`_
* `SciPy Website <http://www.scipy.org/>`_
* `Matplotlib <http://matplotlib.org/>`_

Python for scientific computing
-------------------------------

* `Python for Scientists <https://sites.google.com/site/pythonforscientists/>`_ Links to commonly used packages, Matlab to Python comparison
* `Python Wiki <http://wiki.python.org/moin/NumericAndScientific>`_ Lists packages and resources for scientific computing in Python

Numpy and Matlab
----------------

* `NumPy for Matlab Users <https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html>`_
* `Python vs Matlab <https://sites.google.com/site/pythonforscientists/python-vs-matlab>`_

Lessons in Python
-----------------

* `Software Carpentry <http://swcarpentry.github.io/python-novice-inflammation/>`_
* `Introduction to NumPy and Matplotlib <https://www.youtube.com/watch?v=3Fp1zn5ao2M>`_

Editing Python
--------------

There are numerous ways to edit and test Python (see `PythonWiki <http://wiki.python.org/moin/PythonEditors>`_ for an overview) and in our group at least the following options are being used:

* `Sublime <http://www.sublimetext.com/>`_
* `iPython Notebook <http://ipython.org/notebook.html>`_
* `iPython <http://ipython.org/>`__
