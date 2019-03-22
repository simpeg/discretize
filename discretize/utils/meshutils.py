import numpy as np
from scipy import optimize
import scipy.ndimage as ndi
import scipy.sparse as sp

from .matutils import ndgrid
from .codeutils import asArray_N_x_Dim
from .codeutils import isScalar

import sys
if sys.version_info < (3,):
    num_types = [int, long, float]
else:
    num_types = [int, float]


def exampleLrmGrid(nC, exType):
    assert type(nC) == list, "nC must be a list containing the number of nodes"
    assert len(nC) == 2 or len(nC) == 3, "nC must either two or three dimensions"
    exType = exType.lower()

    possibleTypes = ['rect', 'rotate']
    assert exType in possibleTypes, "Not a possible example type."

    if exType == 'rect':
        return list(ndgrid([np.cumsum(np.r_[0, np.ones(nx)/nx]) for nx in nC], vector=False))
    elif exType == 'rotate':
        if len(nC) == 2:
            X, Y = ndgrid([np.cumsum(np.r_[0, np.ones(nx)/nx]) for nx in nC], vector=False)
            amt = 0.5-np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
            amt[amt < 0] = 0
            return [X + (-(Y - 0.5))*amt, Y + (+(X - 0.5))*amt]
        elif len(nC) == 3:
            X, Y, Z = ndgrid([np.cumsum(np.r_[0, np.ones(nx)/nx]) for nx in nC], vector=False)
            amt = 0.5-np.sqrt((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2)
            amt[amt < 0] = 0
            return [X + (-(Y - 0.5))*amt, Y + (-(Z - 0.5))*amt, Z + (-(X - 0.5))*amt]


def random_model(shape, seed=None, anisotropy=None, its=100, bounds=None):
    """
        Create a random model by convolving a kernel with a
        uniformly distributed model.

        :param tuple shape: shape of the model.
        :param int seed: pick which model to produce, prints the seed if you don't choose.
        :param numpy.ndarray anisotropy: this is the (3 x n) blurring kernel that is used.
        :param int its: number of smoothing iterations
        :param list bounds: bounds on the model, len(list) == 2
        :rtype: numpy.ndarray
        :return: M, the model


        .. plot::

            import matplotlib.pyplot as plt
            import discretize
            plt.colorbar(plt.imshow(discretize.utils.random_model((50, 50), bounds=[-4, 0])))
            plt.title('A very cool, yet completely random model.')
            plt.show()


    """
    if bounds is None:
        bounds = [0, 1]

    if seed is None:
        seed = np.random.randint(1e3)
        print('Using a seed of: ', seed)

    if type(shape) in num_types:
        shape = (shape, ) # make it a tuple for consistency

    np.random.seed(seed)
    mr = np.random.rand(*shape)
    if anisotropy is None:
        if len(shape) is 1:
            smth = np.array([1, 10., 1], dtype=float)
        elif len(shape) is 2:
            smth = np.array([[1, 7, 1], [2, 10, 2], [1, 7, 1]], dtype=float)
        elif len(shape) is 3:
            kernal = np.array([1, 4, 1], dtype=float).reshape((1, 3))
            smth = np.array(sp.kron(sp.kron(kernal, kernal.T).todense()[:], kernal).todense()).reshape((3, 3, 3))
    else:
        assert len(anisotropy.shape) is len(shape), 'Anisotropy must be the same shape.'
        smth = np.array(anisotropy, dtype=float)

    smth = smth/smth.sum() # normalize
    mi = mr
    for i in range(its):
        mi = ndi.convolve(mi, smth)

    # scale the model to live between the bounds.
    mi = (mi - mi.min())/(mi.max()-mi.min()) # scaled between 0 and 1
    mi = mi*(bounds[1]-bounds[0])+bounds[0]

    return mi



def meshTensor(value):
    """**meshTensor** takes a list of numbers and tuples
    that have the form::

        mT = [ float, (cellSize, numCell), (cellSize, numCell, factor) ]

    For example, a time domain mesh code needs
    many time steps at one time::

        [(1e-5, 30), (1e-4, 30), 1e-3]

    Means take 30 steps at 1e-5 and then 30 more at 1e-4,
    and then one step of 1e-3.

    Tensor meshes can also be created by increase factors::

        [(10.0, 5, -1.3), (10.0, 50), (10.0, 5, 1.3)]

    When there is a third number in the tuple, it
    refers to the increase factor, if this number
    is negative this section of the tensor is flipped right-to-left.

    .. plot::

        import discretize
        tx = [(10.0, 10, -1.3), (10.0, 40), (10.0, 10, 1.3)]
        ty = [(10.0, 10, -1.3), (10.0, 40)]
        M = discretize.TensorMesh([tx, ty])
        M.plotGrid(showIt=True)

    """
    if type(value) is not list:
        raise Exception('meshTensor must be a list of scalars and tuples.')

    proposed = []
    for v in value:
        if isScalar(v):
            proposed += [float(v)]
        elif type(v) is tuple and len(v) == 2:
            proposed += [float(v[0])]*int(v[1])
        elif type(v) is tuple and len(v) == 3:
            start = float(v[0])
            num = int(v[1])
            factor = float(v[2])
            pad = ((np.ones(num)*np.abs(factor))**(np.arange(num)+1))*start
            if factor < 0: pad = pad[::-1]
            proposed += pad.tolist()
        else:
            raise Exception('meshTensor must contain only scalars and len(2) or len(3) tuples.')

    return np.array(proposed)


def closestPoints(mesh, pts, gridLoc='CC'):
    """Move a list of points to the closest points on a grid.

    :param BaseMesh mesh: The mesh
    :param numpy.ndarray pts: Points to move
    :param str gridLoc: ['CC', 'N', 'Fx', 'Fy', 'Fz', 'Ex', 'Ex', 'Ey', 'Ez']
    :rtype: numpy.ndarray
    :return: nodeInds
    """

    pts = asArray_N_x_Dim(pts, mesh.dim)
    grid = getattr(mesh, 'grid' + gridLoc)
    nodeInds = np.empty(pts.shape[0], dtype=int)

    for i, pt in enumerate(pts):
        if mesh.dim == 1:
            nodeInds[i] = ((pt - grid)**2).argmin()
        else:
            nodeInds[i] = ((np.tile(pt, (grid.shape[0], 1)) - grid)**2).sum(axis=1).argmin()

    return nodeInds


def ExtractCoreMesh(xyzlim, mesh, meshType='tensor'):
    """
    Extracts Core Mesh from Global mesh

    :param numpy.ndarray xyzlim: 2D array [ndim x 2]
    :param BaseMesh mesh: The mesh

    This function ouputs::

        - actind: corresponding boolean index from global to core
        - meshcore: core sdiscretize mesh
    """
    import discretize
    if mesh.dim == 1:
        xyzlim = xyzlim.flatten()
        xmin, xmax = xyzlim[0], xyzlim[1]

        xind = np.logical_and(mesh.vectorCCx > xmin, mesh.vectorCCx < xmax)

        xc = mesh.vectorCCx[xind]

        hx = mesh.hx[xind]

        x0 = [xc[0]-hx[0]*0.5]

        meshCore = discretize.TensorMesh([hx], x0=x0)

        actind = (mesh.gridCC > xmin) & (mesh.gridCC < xmax)

    elif mesh.dim == 2:
        xmin, xmax = xyzlim[0, 0], xyzlim[0, 1]
        ymin, ymax = xyzlim[1, 0], xyzlim[1, 1]

        xind = np.logical_and(mesh.vectorCCx > xmin, mesh.vectorCCx < xmax)
        yind = np.logical_and(mesh.vectorCCy > ymin, mesh.vectorCCy < ymax)

        xc = mesh.vectorCCx[xind]
        yc = mesh.vectorCCy[yind]

        hx = mesh.hx[xind]
        hy = mesh.hy[yind]

        x0 = [xc[0]-hx[0]*0.5, yc[0]-hy[0]*0.5]

        meshCore = discretize.TensorMesh([hx, hy], x0=x0)

        actind = (
            (mesh.gridCC[:, 0] > xmin) & (mesh.gridCC[:, 0] < xmax) &
            (mesh.gridCC[:, 1] > ymin) & (mesh.gridCC[:, 1] < ymax)
        )

    elif mesh.dim == 3:
        xmin, xmax = xyzlim[0, 0], xyzlim[0, 1]
        ymin, ymax = xyzlim[1, 0], xyzlim[1, 1]
        zmin, zmax = xyzlim[2, 0], xyzlim[2, 1]

        xind = np.logical_and(mesh.vectorCCx > xmin, mesh.vectorCCx < xmax)
        yind = np.logical_and(mesh.vectorCCy > ymin, mesh.vectorCCy < ymax)
        zind = np.logical_and(mesh.vectorCCz > zmin, mesh.vectorCCz < zmax)

        xc = mesh.vectorCCx[xind]
        yc = mesh.vectorCCy[yind]
        zc = mesh.vectorCCz[zind]

        hx = mesh.hx[xind]
        hy = mesh.hy[yind]
        hz = mesh.hz[zind]

        x0 = [xc[0]-hx[0]*0.5, yc[0]-hy[0]*0.5, zc[0]-hz[0]*0.5]

        meshCore = discretize.TensorMesh([hx, hy, hz], x0=x0)

        actind = (
            (mesh.gridCC[:, 0] > xmin) & (mesh.gridCC[:, 0] < xmax) &
            (mesh.gridCC[:, 1] > ymin) & (mesh.gridCC[:, 1] < ymax) &
            (mesh.gridCC[:, 2] > zmin) & (mesh.gridCC[:, 2] < zmax)
        )

    else:
        raise Exception("Not implemented!")

    return actind, meshCore


# HELPER FUNCTIONS TO CREATE MESH
def get_domain(x0=0, freq=1, rho=0.3, limits=None, min_width=None,
               fact_min=0.2, fact_neg=5, fact_pos=None):
    r"""Get domain extent and minimum cell width as a function of skin depth.

    Returns the extent of the calculation domain and the minimum cell width as
    a multiple of the skin depth, with possible user restrictions on minimum
    calculation domain and range of possible minimum cell widths.

    .. math::

            \delta &= 503.3 \sqrt{\frac{\rho}{f}} , \\
            x_\text{start} &= x_0-k_\text{neg}\delta , \\
            x_\text{end} &= x_0+k_\text{pos}\delta , \\
            h_\text{min} &= k_\text{min} \delta .


    Parameters
    ----------

    x0 : float
        Center of the calculation domain. Normally the source location.
        Default is 0.

    freq : float
        Frequency (Hz) to calculate the skin depth.
        Default is 1 Hz.

    rho : float, optional
        Resistivity (Ohm m) to calculate skin depth.
        Default is 0.3 Ohm m (sea water).

    limits : None or list
        [start, end] of model domain. This extent represents the minimum extent
        of the domain. The domain is therefore only adjusted if it has to reach
        outside of [start, end].
        Default is None.

    min_width : None or list
        [min, max] for the minimum cell width. This extent is the range which
        ``h_min`` can take.
        Default is None.

    fact_min, fact_neg, fact_pos : floats
        The skin depth is multiplied with these factors to estimate:

            - Minimum cell width (``fact_min``, default 0.2)
            - Domain-start (``fact_neg``, default 5), and
            - Domain-end (``fact_pos``, defaults to ``fact_neg``).


    Returns
    -------

    h_min : float
        Minimum cell width.

    domain : list
        Start- and end-points of calculation domain.

    """

    # Set fact_pos to fact_neg if not provided
    if fact_pos is None:
        fact_pos = fact_neg

    # Calculate the skin depth
    skind = 503.3*np.sqrt(rho/freq)

    # Estimate minimum cell width
    h_min = fact_min*skind
    if min_width is not None:  # Respect user input
        h_min = np.clip(h_min, *min_width)

    # Estimate calculation domain
    domain = [x0-fact_neg*skind, x0+fact_pos*skind]
    if limits is not None:  # Respect user input
        domain = [min(limits[0], domain[0]), max(limits[1], domain[1])]

    return h_min, domain


def get_stretched_h(h_min, domain, nx, x0=0, x1=None, resp_domain=False):
    """Return cell widths for a stretched grid within the domain.

    Returns ``nx`` cell widths within ``domain``, where the minimum cell width
    is ``h_min``. The cells are not stretched within ``x0`` and ``x1``, and
    outside uses a power-law stretching. The actual stretching factor and the
    number of cells left and right of ``x0`` and ``x1`` are find in a
    minimization process.

    The domain is not completely respected. The starting point of the domain
    is, but the endpoint of the domain might slightly shift (this is more
    likely the case for small ``nx``, for big ``nx`` the shift should be
    small). The new endpoint can be obtained with ``domain[0]+np.sum(hx)``. If
    you want the domain to be respected absolutely, set ``resp_domain=True``.
    However, be aware that this will introduce one stretch-factor which is
    different from the other stretch factors, to accommodate the restriction.
    This one-off factor is between the left- and right-side of ``x0``, or, if
    ``x1`` is provided, just after ``x1``.


    Parameters
    ----------

    h_min : float
        Minimum cell width.

    domain : list
        [start, end] of model domain.

    nx : int
        Number of cells.

    x0 : float
        Center of the grid. ``x0`` is restricted to ``domain``.
        Default is 0.

    x1 : float
        If provided, then no stretching is applied between ``x0`` and ``x1``.
        The non-stretched part starts at ``x0`` and stops at the first possible
        location at or after ``x1``. ``x1`` is restricted to ``domain``.

    resp_domain : bool
        If False (default), then the domain-end might shift slightly to assure
        that the same stretching factor is applied throughout. If set to True,
        however, the domain is respected absolutely. This will introduce one
        stretch-factor which is different from the other stretch factors, to
        accommodate the restriction. This one-off factor is between the left-
        and right-side of ``x0``, or, if ``x1`` is provided, just after ``x1``.


    Returns
    -------
    hx : ndarray
        Cell widths of mesh.

    """

    # Cast to arrays
    domain = np.array(domain, dtype=float)
    x0 = np.array(x0, dtype=float)
    x0 = np.clip(x0, *domain)  # Restrict to model domain
    if x1 is not None:
        x1 = np.array(x1, dtype=float)
        x1 = np.clip(x1, *domain)  # Restrict to model domain

    # If x1 is provided (a part is not stretched)
    if x1 is not None:

        # Store original values
        xlim_orig = domain.copy()
        nx_orig = int(nx)
        x0_orig = x0.copy()

        # Get number of non-stretched cells
        n_nos = int(np.ceil((x1-x0)/h_min))-1
        # Note that wee subtract one cell, because the standard scheme provides
        # one h_min-cell.

        # Reset x0, because the first h_min comes from normal scheme
        x0 += h_min

        # Reset xmax for normal scheme
        domain[1] -= n_nos*h_min

        # Reset nx for normal scheme
        nx -= n_nos

        # If there are not enough points reset to standard procedure
        # This five is arbitrary. However, nx should be much bigger than five
        # anyways, otherwise stretched grid doesn't make sense.
        if nx <= 5:
            print("Warning :: Not enough points for non-stretched part,"
                  "ignoring therefore `x1`.")
            domain = xlim_orig
            nx = nx_orig
            x0 = x0_orig
            x1 = None

    # Get stretching factor (a = 1+alpha).
    if h_min == 0 or h_min > np.diff(domain)/nx:
        # If h_min is bigger than the domain-extent divided by nx, no
        # stretching is required at all.
        alpha = 0
    else:

        # Wrap _get_dx into a minimization function to call with fsolve.
        def find_alpha(alpha, h_min, args):
            """Find alpha such that min(hx) = h_min."""
            return min(get_hx(alpha, *args))/h_min-1

        # Search for best alpha, must be at least 0
        args = (domain, nx, x0)
        alpha = max(0, optimize.fsolve(find_alpha, 0.02, (h_min, args)))

    # With alpha get actual cell spacing with `resp_domain` to respect the
    # users decision.
    hx = get_hx(alpha, domain, nx, x0, resp_domain)

    # Add the non-stretched center if x1 is provided
    if x1 is not None:
        hx = np.r_[hx[: np.argmin(hx)], np.ones(n_nos)*h_min,
                   hx[np.argmin(hx):]]

    # Print mesh dimensions
    print(f"extent : {domain[0]:>10,.1f} - {domain[0]+np.sum(hx):<10,.1f}; "
          f"min/max width: {min(hx):>6,.1f} - {max(hx):<6,.1f}; "
          f"stretching: {1+np.squeeze(alpha):.3f}")

    return hx


def get_hx(alpha, domain, nx, x0, resp_domain=True):
    r"""Return cell widths for given input.

    Find the number of cells left and right of ``x0``, ``nl`` and ``nr``
    respectively, for the provided alpha. For this, we solve

    .. math::   \frac{x_\text{max}-x_0}{x_0-x_\text{min}} =
                \frac{a^{nr}-1}{a^{nl}-1}

    where :math:`a = 1+\alpha`.


    Parameters
    ----------

    alpha : float
        Stretching factor ``a`` is given by ``a=1+alpha``.

    domain : list
        [start, end] of model domain.

    nx : int
        Number of cells.

    x0 : float
        Center of the grid. ``x0`` is restricted to ``domain``.

    resp_domain : bool
        If False (default), then the domain-end might shift slightly to assure
        that the same stretching factor is applied throughout. If set to True,
        however, the domain is respected absolutely. This will introduce one
        stretch-factor which is different from the other stretch factors, to
        accommodate the restriction. This one-off factor is between the left-
        and right-side of ``x0``, or, if ``x1`` is provided, just after ``x1``.


    Returns
    -------
    hx : ndarray
        Cell widths of mesh.

    """
    if alpha <= 0.:  # If alpha <= 0: equal spacing (no stretching at all)
        hx = np.ones(nx)*np.diff(domain)/nx

    else:            # Get stretched hx
        a = alpha+1

        # Get hx depending if x0 is on the domain boundary or not.
        if x0 == domain[0] or x0 == domain[1]:
            # Get al a's
            alr = np.diff(domain)*alpha/(a**nx-1)*a**np.arange(nx)
            if x0 == domain[1]:
                alr = alr[::-1]

            # Calculate differences
            hx = alr*np.diff(domain)/sum(alr)

        else:
            # Find number of elements left and right by solving:
            #     (xmax-x0)/(x0-xmin) = a**nr-1/(a**nl-1)
            nr = np.arange(2, nx+1)
            er = (domain[1]-x0)/(x0-domain[0]) - (a**nr[::-1]-1)/(a**nr-1)
            nl = np.argmin(abs(np.floor(er)))+1
            nr = nx-nl

            # Get all a's
            al = a**np.arange(nl-1, -1, -1)
            ar = a**np.arange(1, nr+1)

            # Calculate differences
            if resp_domain:
                # This version honours domain[0] and domain[1], but to achieve
                # this it introduces one stretch-factor which is different from
                # all the others between al to ar.
                hx = np.r_[al*(x0-domain[0])/sum(al),
                           ar*(domain[1]-x0)/sum(ar)]
            else:
                # This version moves domain[1], but each stretch-factor is
                # exactly the same.
                fact = (x0-domain[0])/sum(al)  # Take distance from al.
                hx = np.r_[al, ar]*fact

                # Note: this hx is equivalent as providing the following h
                # to TensorMesh:
                # h = [(h_min, nl-1, -a), (h_min, n_nos+1), (h_min, nr, a)]

    return hx
