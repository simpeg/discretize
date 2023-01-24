.. image:: https://raw.github.com/simpeg/discretize/main/docs/images/discretize-logo.png
    :alt: Discretize Logo

discretize
==========

.. image:: https://img.shields.io/pypi/v/discretize.svg
    :target: https://pypi.python.org/pypi/discretize
    :alt: Latest PyPI version

.. image:: https://anaconda.org/conda-forge/discretize/badges/version.svg
    :target: https://anaconda.org/conda-forge/discretize
    :alt: Latest conda-forge version

.. image:: https://img.shields.io/github/license/simpeg/simpeg.svg
    :target: https://github.com/simpeg/discretize/blob/main/LICENSE
    :alt: MIT license

.. image:: https://dev.azure.com/simpeg/discretize/_apis/build/status/simpeg.discretize?branchName=main
    :target: https://dev.azure.com/simpeg/discretize/_build/latest?definitionId=1&branchName=main
    :alt: Azure pipelines build status

.. image:: https://codecov.io/gh/simpeg/discretize/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/simpeg/discretize
    :alt: Coverage status

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.596411.svg
   :target: https://doi.org/10.5281/zenodo.596411

.. image:: https://img.shields.io/discourse/users?server=http%3A%2F%2Fsimpeg.discourse.group%2F
    :target: http://simpeg.discourse.group/

.. image:: https://img.shields.io/badge/Slack-simpeg-4B0082.svg?logo=slack
    :target: http://slack.simpeg.xyz

.. image:: https://img.shields.io/badge/Youtube%20channel-GeoSci.xyz-FF0000.svg?logo=youtube
    :target: https://www.youtube.com/channel/UCBrC4M8_S4GXhyHht7FyQqw


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
* Triangular (2D) and Tetrahedral (3D) Meshes

Installing
^^^^^^^^^^
**discretize** is on conda-forge

.. code:: shell

    conda install -c conda-forge discretize

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
http://simpeg.discourse.group/

Chat:
http://slack.simpeg.xyz/
