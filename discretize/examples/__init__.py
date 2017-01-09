from __future__ import print_function
# Run this file to add imports.

##### AUTOIMPORTS #####
from discretize.examples import Mesh_Basic_ForwardDC
from discretize.examples import Mesh_Basic_PlotImage
from discretize.examples import Mesh_Basic_Types
from discretize.examples import Mesh_Operators_CahnHilliard
from discretize.examples import Mesh_Plot_Cyl
from discretize.examples import Mesh_QuadTree_Creation
from discretize.examples import Mesh_QuadTree_FaceDiv
from discretize.examples import Mesh_QuadTree_HangingNodes
from discretize.examples import Mesh_Tensor_Creation
from discretize.examples import Utils_plot2Ddata
from discretize.examples import Utils_surface2ind_topo

__examples__ = ["Mesh_Basic_ForwardDC", "Mesh_Basic_PlotImage", "Mesh_Basic_Types", "Mesh_Operators_CahnHilliard", "Mesh_Plot_Cyl", "Mesh_QuadTree_Creation", "Mesh_QuadTree_FaceDiv", "Mesh_QuadTree_HangingNodes", "Mesh_Tensor_Creation", "Utils_plot2Ddata", "Utils_surface2ind_topo"]

##### AUTOIMPORTS #####

if __name__ == '__main__':
    """

        Run the following to create the examples documentation and add to the imports at the top.

    """

    import shutil
    import os
    from discretize import examples

    # Create the examples dir in the docs folder.
    fName = os.path.realpath(__file__)
    doc_examples_dir = os.path.sep.join(fName.split(os.path.sep)[:-3] + ['docs', 'content', 'examples'])
    shutil.rmtree(doc_examples_dir)
    os.makedirs(doc_examples_dir)

    # Get all the python examples in this folder
    thispath = os.path.sep.join(fName.split(os.path.sep)[:-1])
    exfiles  = [f[:-3] for f in os.listdir(thispath) if
                os.path.isfile(os.path.join(thispath, f)) and f.endswith('.py')
                and not f.startswith('_')]

    # Add the imports to the top in the AUTOIMPORTS section
    f = open(fName, 'r')
    inimports = False
    out = ''
    for line in f:
        if not inimports:
            out += line

        if line == "##### AUTOIMPORTS #####\n":
            inimports = not inimports
            if inimports:
                out += '\n'.join(["from discretize.examples import {0!s}".format(_)
                                  for _ in exfiles])
                out += '\n\n__examples__ = ["' + '", "'.join(exfiles)+ '"]\n'
                out += '\n##### AUTOIMPORTS #####\n'
    f.close()

    f = open(fName, 'w')
    f.write(out)
    f.close()

    def _makeExample(filePath, runFunction):
        """Makes the example given a path of the file and the run function."""
        filePath = os.path.realpath(filePath)
        name = filePath.split(os.path.sep)[-1].rstrip('.pyc').rstrip('.py')

        docstr = runFunction.__doc__
        if docstr is None:
            doc = '{0!s}\n{1!s}'.format(name.replace('_',' '), '='*len(name))
        else:
            doc = '\n'.join([_[8:].rstrip() for _ in docstr.split('\n')])

        out = """.. _examples_{0!s}:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    discretize/examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..

{1!s}

.. plot::

    from discretize import examples
    examples.{2!s}.run()

.. literalinclude:: ../../../discretize/examples/{3!s}.py
    :language: python
    :linenos:
""".format(name, doc, name, name)

        rst = (os.path.sep.join((filePath.split(os.path.sep)[:-3] +
               ['docs', 'content', 'examples', name + '.rst'])))

        print('Creating: {0!s}.rst'.format(name))
        f = open(rst, 'w')
        f.write(out)
        f.close()

    for ex in dir(examples):
        if ex.startswith('_') or ex.startswith('print_function'):
            continue
        E = getattr(examples, ex)
        _makeExample(E.__file__, E.run)
