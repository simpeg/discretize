"""
.. _pyvista_demo_ref:

3D Visualization with PyVista
=============================

The example demonstrates the how to use the VTK interface via the
`pyvista library <http://docs.pyvista.org>`__ .
To run this example, you will need to `install pyvista <http://docs.pyvista.org/getting-started/installation.html>`__ .

- contributed by `@banesullivan <https://github.com/banesullivan>`_

Using the inversion result from the example notebook
`plot_laguna_del_maule_inversion.ipynb <http://docs.simpeg.xyz/content/examples/20-published/plot_laguna_del_maule_inversion.html>`_

"""
# sphinx_gallery_thumbnail_number = 2
import os
import tarfile
import discretize
import pyvista as pv
import numpy as np

# Set a documentation friendly plotting theme
pv.set_plot_theme("document")

print("PyVista Version: {}".format(pv.__version__))

###############################################################################
# Download and load data
# ----------------------
#
# In the following we load the :code:`mesh` and :code:`Lpout` that you would
# get from running the laguna-del-maule inversion notebook as well as some of
# the raw data for the topography surface and gravity observations.

# Download Topography and Observed gravity data
url = "https://storage.googleapis.com/simpeg/Chile_GRAV_4_Miller/Chile_GRAV_4_Miller.tar.gz"
downloads = discretize.utils.download(url, overwrite=True)
basePath = downloads.split(".")[0]

# unzip the tarfile
tar = tarfile.open(downloads, "r")
tar.extractall()
tar.close()

# Download the inverted model
f = discretize.utils.download(
    "https://storage.googleapis.com/simpeg/laguna_del_maule_slicer.tar.gz",
    overwrite=True,
)
tar = tarfile.open(f, "r")
tar.extractall()
tar.close()

# Load the mesh/data
mesh = discretize.load_mesh(os.path.join("laguna_del_maule_slicer", "mesh.json"))
models = {"Lpout": np.load(os.path.join("laguna_del_maule_slicer", "Lpout.npy"))}


###############################################################################
# Create PyVista data objects
# ---------------------------
#
# Here we start making PyVista data objects of all the spatially referenced
# data.

# Get the PyVista dataset of the inverted model
dataset = mesh.to_vtk(models)
dataset.set_active_scalars('Lpout')

###############################################################################

# Load topography points from text file as XYZ numpy array
topo_pts = np.loadtxt("Chile_GRAV_4_Miller/LdM_topo.topo", skiprows=1)
# Create the topography points and apply an elevation filter
topo = pv.PolyData(topo_pts).delaunay_2d().elevation()

###############################################################################

# Load the gravity data from text file as XYZ+attributes numpy array
grav_data = np.loadtxt("Chile_GRAV_4_Miller/LdM_grav_obs.grv", skiprows=1)
print("gravity file shape: ", grav_data.shape)
# Use the points to create PolyData
grav = pv.PolyData(grav_data[:, 0:3])
# Add the data arrays
grav.point_data["comp-1"] = grav_data[:, 3]
grav.point_data["comp-2"] = grav_data[:, 4]
grav.set_active_scalars('comp-1')

###############################################################################
# Plot the topographic surface and the gravity data

p = pv.Plotter()
p.add_mesh(topo, color="grey")
p.add_mesh(
    grav, point_size=15, render_points_as_spheres=True,
    scalar_bar_args={"title": "Observed Gravtiy Data"}
)
# Use a non-phot-realistic shading technique to show topographic relief
p.enable_eye_dome_lighting()
p.show(window_size=[1024, 768])


###############################################################################
# Visualize Using PyVista
# -----------------------
#
# Here we visualize all the data in 3D!

# Create display parameters for inverted model
dparams = dict(
    show_edges=False,
    cmap="bwr",
    clim=[-0.6, 0.6],
)

# Apply a threshold filter to remove topography
#  no arguments will remove the NaN values
dataset_t = dataset.threshold()

# Extract volumetric threshold
threshed = dataset_t.threshold(-0.2, invert=True)

# Create the rendering scene
p = pv.Plotter()
# add a grid axes
p.show_grid()

# Add spatially referenced data to the scene
p.add_mesh(dataset_t.slice("x"), **dparams)
p.add_mesh(dataset_t.slice("y"), **dparams)
p.add_mesh(threshed, **dparams)
p.add_mesh(
    topo,
    opacity=0.75,
    color="grey",
    # cmap='gist_earth', clim=[1.7e+03, 3.104e+03],
)
p.add_mesh(grav, cmap="viridis", point_size=15, render_points_as_spheres=True)

# Here is a nice camera position we manually found:
cpos = [
    (395020.7332989303, 6039949.0452080015, 20387.583125699253),
    (364528.3152860675, 6008839.363092581, -3776.318305935185),
    (-0.3423732500124074, -0.34364514928896667, 0.8744647328772646),
]
p.camera_position = cpos


# Render the scene!
p.show(window_size=[1024, 768])
