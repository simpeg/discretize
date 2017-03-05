"""
Cyl: Plot Grid
================

You can use :code:`mesh.plotGrid()` to plot a z and theta slice through a
3D cyl mesh.

If you wish to plot only one slice, you can use
:code:`mesh.plotGrid(slice='z')` or :code:`mesh.plotGrid(slice='theta')`

"""
import discretize
import matplotlib.pyplot as plt


def run(plotIt=True):
    mesh = discretize.CylMesh([7, 6, 10])
    mesh.plotGrid()
    # mesh.plotGrid(slice='z')
    # mesh.plotGrid(slice='theta')

if __name__ == '__main__':
    run()
    plt.show()
