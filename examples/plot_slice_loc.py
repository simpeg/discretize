"""
Basic: plot_slice
================

You can use M.plot_slice to plot images.
"""
import discretize
import matplotlib.pyplot as plt
import numpy as np

def run(plotIt=True):
    M = discretize.TensorMesh([32, 32, 32])
    v = discretize.utils.random_model(M.vnC, seed=789)
    v = discretize.utils.mkvc(v)

    O = discretize.TreeMesh([32, 32,32])

    def function(cell):
        if (
            cell.center[0] < 0.75
            and cell.center[0] > 0.25
            and cell.center[1] < 0.75
            and cell.center[1] > 0.25
            and cell.center[2] < 0.75
            and cell.center[2] > 0.25
        ):
            return 6
        if (
            cell.center[0] < 0.85
            and cell.center[0] > 0.15
            and cell.center[1] < 0.85
            and cell.center[1] > 0.15
            and cell.center[2] < 0.85
            and cell.center[2] > 0.15
        ):
            return 5
        if (
            cell.center[0] < 0.9
            and cell.center[0] > 0.1
            and cell.center[1] < 0.9
            and cell.center[1] > 0.1
            and cell.center[2] < 0.9
            and cell.center[2] > 0.1
        ):
            return 4
        return 3

    O.refine(function)

    P = M.getInterpolationMat(O.gridCC, "CC")

    ov = P * v

    if not plotIt:
        return

    xslice = 0.75
    yslice = 0.25
    zslice = 0.9
    print('xslice index:',int(np.argmin(np.abs(M.cell_centers_x - xslice))))
    print('yslice index:',int(np.argmin(np.abs(M.cell_centers_y - yslice))))
    print('zslice index:',int(np.argmin(np.abs(M.cell_centers_z - zslice))))

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    out = M.plot_slice(v, ax=axes[0], normal='X', ind=23)
    cb = plt.colorbar(out[0], ax=axes[0])
    cb.set_label("Random Field")
    axes[0].set_title("x-normal slice")

    out = M.plot_slice(v, ax=axes[1], normal='Y', ind=7)
    cb = plt.colorbar(out[0], ax=axes[1])
    cb.set_label("Random Field")
    axes[1].set_title("y-normal slice")

    out = M.plot_slice(v, ax=axes[2], normal='Z', ind=28)
    cb = plt.colorbar(out[0], ax=axes[2])
    cb.set_label("Random Field")
    axes[2].set_title("z-normal slice")

    ## Now with slice_loc

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    out = M.plot_slice(v, ax=axes[0], normal='X', slice_loc=xslice)
    cb = plt.colorbar(out[0], ax=axes[0])
    cb.set_label("Random Field")
    axes[0].set_title("x-normal slice")

    out = M.plot_slice(v, ax=axes[1], normal='Y', slice_loc=yslice)
    cb = plt.colorbar(out[0], ax=axes[1])
    cb.set_label("Random Field")
    axes[1].set_title("y-normal slice")

    out = M.plot_slice(v, ax=axes[2], normal='Z', slice_loc=zslice)
    cb = plt.colorbar(out[0], ax=axes[2])
    cb.set_label("Random Field")
    axes[2].set_title("z-normal slice")

    # try it on a TreeMesh
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    out = O.plot_slice(ov, ax=axes[0], normal='X', slice_loc=xslice)
    cb = plt.colorbar(out[0], ax=axes[0])
    cb.set_label("Random Field")
    axes[0].set_title("x-normal slice")

    out = O.plot_slice(ov, ax=axes[1], normal='Y', slice_loc=yslice)
    cb = plt.colorbar(out[0], ax=axes[1])
    cb.set_label("Random Field")
    axes[1].set_title("y-normal slice")

    out = O.plot_slice(ov, ax=axes[2], normal='Z', slice_loc=zslice)
    cb = plt.colorbar(out[0], ax=axes[2])
    cb.set_label("Random Field")
    axes[2].set_title("z-normal slice")

if __name__ == "__main__":
    run()
    plt.show()
