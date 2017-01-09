.. _examples_quadtree_hanging:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    discretize/examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


QuadTree: Hanging Nodes
=======================

You can give the refine method a function, which is evaluated on every
cell of the TreeMesh.

Occasionally it is useful to initially refine to a constant level
(e.g. 3 in this 32x32 mesh). This means the function is first evaluated
on an 8x8 mesh (2^3).



.. plot::

    from discretize import examples
    examples.quadtree_hanging.run()

.. literalinclude:: ../../../discretize/examples/quadtree_hanging.py
    :language: python
    :linenos:
