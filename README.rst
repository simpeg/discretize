discretize
==========

.. image:: https://img.shields.io/pypi/v/discretize.svg
    :target: https://pypi.python.org/pypi/discretize
    :alt: Latest PyPI version

.. image:: https://img.shields.io/github/license/simpeg/simpeg.svg
    :target: https://github.com/simpeg/discretize/blob/master/LICENSE
    :alt: MIT license

.. image:: https://api.travis-ci.org/simpeg/discretize.svg?branch=master
    :target: https://travis-ci.org/simpeg/discretize
    :alt: Travis CI build status

.. image:: https://codecov.io/gh/simpeg/discretize/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/simpeg/discretize
    :alt: Coverage status

.. image:: https://api.codacy.com/project/badge/Grade/644262e9ee5d4fa79b7041e1ad61f131
    :target: https://www.codacy.com/app/lindseyheagy/discretize?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=simpeg/discretize&amp;utm_campaign=Badge_Grade
    :alt: codacy status

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.596411.svg
   :target: https://doi.org/10.5281/zenodo.596411

.. image:: https://img.shields.io/badge/Slack-simpeg-4B0082.svg?logo=slack
    :target: http://slack.simpeg.xyz

.. image:: https://img.shields.io/badge/Google%20group-simpeg-da5247.svg
    :target: https://groups.google.com/forum/#!forum/simpeg


**discretize** - A python package for finite volume discretization.

The vision is to create a package for finite volume simulation with a
focus on large scale inverse problems.
This package has the following features:

* modular with respect to the spacial discretization
* built with the inverse problem in mind
* supports 1D, 2D and 3D problems
* access to sparse matrix operators
* access to derivatives to mesh variables

.. image:: https://raw.githubusercontent.com/simpeg/figures/master/finitevolume/cell-anatomy-tensor.png

Currently, discretize supports:

* Tensor Meshes (1D, 2D and 3D)
* Cylindrically Symmetric Meshes
* QuadTree and OcTree Meshes (2D and 3D)
* Logically Rectangular Meshes (2D and 3D)

Installing
^^^^^^^^^^

**discretize** is on pypi

.. code:: shell

    pip install discretize

To install from source

.. code:: shell

    git clone https://github.com/simpeg/discretize.git
    python setup.py install

Citing discretize
^^^^^^^^^^^^^^^^^

Please cite the SimPEG paper when using discretize in your work:


    Cockett, R., Kang, S., Heagy, L. J., Pidlisecky, A., & Oldenburg, D. W. (2015). SimPEG: An open source framework for simulation and gradient based parameter estimation in geophysical applications. Computers & Geosciences.

**BibTex:**

.. code:: Latex

    @article{cockett2015simpeg,
      title={SimPEG: An open source framework for simulation and gradient based parameter estimation in geophysical applications},
      author={Cockett, Rowan and Kang, Seogi and Heagy, Lindsey J and Pidlisecky, Adam and Oldenburg, Douglas W},
      journal={Computers \& Geosciences},
      year={2015},
      publisher={Elsevier}
    }

Links
^^^^^

Website:
http://simpeg.xyz

Documentation:
http://discretize.simpeg.xyz

Code:
https://github.com/simpeg/discretize

Tests:
https://travis-ci.org/simpeg/discretize

Bugs & Issues:
https://github.com/simpeg/discretize/issues

Questions:
https://groups.google.com/forum/#!forum/simpeg

Chat:
http://slack.simpeg.xyz/

