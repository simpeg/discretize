# cython: embedsignature=True, language_level=3
# cython: linetrace=True
import numpy as np
import cython
cimport numpy as np
import scipy.sparse as sp

def _interp_point_1D(np.ndarray[np.float64_t, ndim=1] x, float xr_i):
    """
        given a point, xr_i, this will find which two integers it lies between.

        :param numpy.ndarray x: Tensor vector of 1st dimension of grid.
        :param float xr_i: Location of a point
        :rtype: int,int,float,float
        :return: index1, index2, portion1, portion2
    """
    cdef IIFF xs
    _get_inds_ws(x,xr_i,&xs)
    return xs.i1,xs.i2,xs.w1,xs.w2

cdef struct IIFF:
    np.int64_t i1,i2
    np.float64_t w1,w2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.int64_t _bisect_left(np.float64_t[:] a, np.float64_t x) nogil:
    cdef np.int64_t lo, hi, mid
    lo = 0
    hi = a.shape[0]
    while lo < hi:
      mid = (lo+hi)//2
      if a[mid] < x: lo = mid+1
      else: hi = mid
    return lo

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.int64_t _bisect_right(np.float64_t[:] a, np.float64_t x) nogil:
    cdef np.int64_t lo, hi, mid
    lo = 0
    hi = a.shape[0]
    while lo < hi:
      mid = (lo+hi)//2
      if x < a[mid]: hi = mid
      else: lo = mid+1
    return lo

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _get_inds_ws(np.float64_t[:] x, np.float64_t xp, IIFF* out) nogil:
    cdef np.int64_t ind = _bisect_right(x,xp)
    cdef np.int64_t nx = x.shape[0]
    out.i2 = ind
    out.i1 = ind-1
    out.i2 = max(min(out.i2,nx-1),0)
    out.i1 = max(min(out.i1,nx-1),0)
    if(out.i1==out.i2):
        out.w1 = 0.5
    else:
        out.w1 = (x[out.i2]-xp)/(x[out.i2]-x[out.i1])
    out.w2 = 1-out.w1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def _interpmat1D(np.ndarray[np.float64_t, ndim=1] locs,
                   np.ndarray[np.float64_t, ndim=1] x):
    cdef int nx = x.size
    cdef IIFF xs
    cdef int npts = locs.shape[0]
    cdef int i

    cdef np.ndarray[np.int64_t,ndim=1] inds = np.empty(npts*2,dtype=np.int64)
    cdef np.ndarray[np.float64_t,ndim=1] vals = np.empty(npts*2,dtype=np.float64)
    for i in range(npts):
        _get_inds_ws(x,locs[i],&xs)

        inds[2*i  ] = xs.i1
        inds[2*i+1] = xs.i2
        vals[2*i  ] = xs.w1
        vals[2*i+1] = xs.w2

    return inds,vals

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def _interpmat2D(np.ndarray[np.float64_t, ndim=2] locs,
                   np.ndarray[np.float64_t, ndim=1] x,
                   np.ndarray[np.float64_t, ndim=1] y):
    cdef int nx,ny
    nx,ny = len(x),len(y)
    cdef int npts = locs.shape[0]
    cdef int i
    cdef IIFF xs,ys

    cdef np.ndarray[np.int64_t,ndim=2] inds = np.empty((npts*4,2),dtype=np.int64)
    cdef np.ndarray[np.float64_t,ndim=1] vals = np.empty(npts*4,dtype=np.float64)
    for i in range(npts):
        _get_inds_ws(x,locs[i,0],&xs)
        _get_inds_ws(y,locs[i,1],&ys)

        inds[4*i  ,0] = xs.i1
        inds[4*i+1,0] = xs.i1
        inds[4*i+2,0] = xs.i2
        inds[4*i+3,0] = xs.i2
        inds[4*i  ,1] = ys.i1
        inds[4*i+1,1] = ys.i2
        inds[4*i+2,1] = ys.i1
        inds[4*i+3,1] = ys.i2

        vals[4*i  ] = xs.w1*ys.w1
        vals[4*i+1] = xs.w1*ys.w2
        vals[4*i+2] = xs.w2*ys.w1
        vals[4*i+3] = xs.w2*ys.w2

    return inds,vals

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def _interpmat3D(np.ndarray[np.float64_t, ndim=2] locs,
                 np.ndarray[np.float64_t, ndim=1] x,
                 np.ndarray[np.float64_t, ndim=1] y,
                 np.ndarray[np.float64_t, ndim=1] z):

    cdef int nx,ny,nz
    nx,ny,nz = len(x),len(y),len(z)
    cdef IIFF xs,ys,zs
    cdef int npts = locs.shape[0]
    cdef int i

    cdef np.ndarray[np.int64_t,ndim=2] inds = np.empty((npts*8,3),dtype=np.int64)
    cdef np.ndarray[np.float64_t,ndim=1] vals = np.empty(npts*8,dtype=np.float64)
    for i in range(npts):
        _get_inds_ws(x,locs[i,0],&xs)
        _get_inds_ws(y,locs[i,1],&ys)
        _get_inds_ws(z,locs[i,2],&zs)

        inds[8*i  ,0] = xs.i1
        inds[8*i+1,0] = xs.i1
        inds[8*i+2,0] = xs.i2
        inds[8*i+3,0] = xs.i2
        inds[8*i+4,0] = xs.i1
        inds[8*i+5,0] = xs.i1
        inds[8*i+6,0] = xs.i2
        inds[8*i+7,0] = xs.i2

        inds[8*i  ,1] = ys.i1
        inds[8*i+1,1] = ys.i2
        inds[8*i+2,1] = ys.i1
        inds[8*i+3,1] = ys.i2
        inds[8*i+4,1] = ys.i1
        inds[8*i+5,1] = ys.i2
        inds[8*i+6,1] = ys.i1
        inds[8*i+7,1] = ys.i2

        inds[8*i  ,2] = zs.i1
        inds[8*i+1,2] = zs.i1
        inds[8*i+2,2] = zs.i1
        inds[8*i+3,2] = zs.i1
        inds[8*i+4,2] = zs.i2
        inds[8*i+5,2] = zs.i2
        inds[8*i+6,2] = zs.i2
        inds[8*i+7,2] = zs.i2

        vals[8*i  ] = xs.w1*ys.w1*zs.w1
        vals[8*i+1] = xs.w1*ys.w2*zs.w1
        vals[8*i+2] = xs.w2*ys.w1*zs.w1
        vals[8*i+3] = xs.w2*ys.w2*zs.w1
        vals[8*i+4] = xs.w1*ys.w1*zs.w2
        vals[8*i+5] = xs.w1*ys.w2*zs.w2
        vals[8*i+6] = xs.w2*ys.w1*zs.w2
        vals[8*i+7] = xs.w2*ys.w2*zs.w2

    return inds,vals

@cython.boundscheck(False)
@cython.cdivision(True)
def _tensor_volume_averaging(mesh_in, mesh_out, values=None, output=None):

    cdef np.int32_t[:] i1_in, i1_out, i2_in, i2_out, i3_in, i3_out
    cdef np.float64_t[:] w1, w2, w3
    w1 = np.array([1.0], dtype=np.float64)
    w2 = np.array([1.0], dtype=np.float64)
    w3 = np.array([1.0], dtype=np.float64)
    i1_in = np.array([0], dtype=np.int32)
    i1_out = np.array([0], dtype=np.int32)
    i2_in = np.array([0], dtype=np.int32)
    i2_out = np.array([0], dtype=np.int32)
    i3_in = np.array([0], dtype=np.int32)
    i3_out = np.array([0], dtype=np.int32)
    cdef int dim = mesh_in.dim
    w1, i1_in, i1_out = _volume_avg_weights(mesh_in.nodes_x, mesh_out.nodes_x)
    if dim > 1:
        w2, i2_in, i2_out = _volume_avg_weights(mesh_in.nodes_y, mesh_out.nodes_y)
    if dim > 2:
        w3, i3_in, i3_out = _volume_avg_weights(mesh_in.nodes_z, mesh_out.nodes_z)

    cdef (np.int32_t, np.int32_t, np.int32_t) w_shape = (w1.shape[0], w2.shape[0], w3.shape[0])
    cdef (np.int32_t, np.int32_t, np.int32_t) mesh_in_shape
    cdef (np.int32_t, np.int32_t, np.int32_t) mesh_out_shape

    nCv_in = [len(h) for h in mesh_in.h]
    nCv_out = [len(h) for h in mesh_out.h]
    if dim == 1:
        mesh_in_shape = (nCv_in[0], 1, 1)
        mesh_out_shape = (nCv_out[0], 1, 1)
    elif dim == 2:
        mesh_in_shape = (nCv_in[0], nCv_in[1], 1)
        mesh_out_shape = (nCv_out[0], nCv_out[1], 1)
    elif dim == 3:
        mesh_in_shape = (*nCv_in, )
        mesh_out_shape = (*nCv_out, )

    cdef np.float64_t[::1, :, :] val_in
    cdef np.float64_t[::1, :, :] val_out
    cdef int i1, i2, i3, i1i, i2i, i3i, i1o, i2o, i3o
    cdef np.float64_t w_3, w_32
    cdef np.float64_t[::1, :, :] vol = mesh_out.cell_volumes.reshape(mesh_out_shape, order='F').astype(np.float64)

    if values is not None:
        # If given a values array, do the operation
        val_in = values.reshape(mesh_in_shape, order='F').astype(np.float64)
        if output is None:
            v_o = np.zeros(mesh_out_shape, order='F')
        else:
            v_o = output.reshape(mesh_out_shape, order='F')
            v_o.fill(0)
        val_out = v_o
        for i3 in range(w_shape[2]):
            i3i = i3_in[i3]
            i3o = i3_out[i3]
            w_3 = w3[i3]
            for i2 in range(w_shape[1]):
                i2i = i2_in[i2]
                i2o = i2_out[i2]
                w_32 = w_3*w2[i2]
                for i1 in range(w_shape[0]):
                    i1i = i1_in[i1]
                    i1o = i1_out[i1]
                    val_out[i1o, i2o, i3o] += w_32*w1[i1]*val_in[i1i, i2i, i3i]/vol[i1o, i2o, i3o]
        return v_o.reshape(-1, order='F')

    # Else, build and return a sparse matrix representing the operation
    i_i = np.empty(w_shape, dtype=np.int32, order='F')
    i_o = np.empty(w_shape, dtype=np.int32, order='F')
    ws = np.empty(w_shape, dtype=np.float64, order='F')
    cdef np.int32_t[::1,:,:] i_in = i_i
    cdef np.int32_t[::1,:,:] i_out = i_o
    cdef np.float64_t[::1, :, :] w = ws
    for i3 in range(w.shape[2]):
        i3i = i3_in[i3]
        i3o = i3_out[i3]
        w_3 = w3[i3]
        for i2 in range(w.shape[1]):
            i2i = i2_in[i2]
            i2o = i2_out[i2]
            w_32 = w_3*w2[i2]
            for i1 in range(w.shape[0]):
                i1i = i1_in[i1]
                i1o = i1_out[i1]
                w[i1, i2, i3] = w_32*w1[i1]/vol[i1o, i2o, i3o]
                i_in[i1, i2, i3] = (i3i*mesh_in_shape[1] + i2i)*mesh_in_shape[0] + i1i
                i_out[i1, i2, i3] = (i3o*mesh_out_shape[1] + i2o)*mesh_out_shape[0] + i1o
    ws = ws.reshape(-1, order='F')
    i_i = i_i.reshape(-1, order='F')
    i_o = i_o.reshape(-1, order='F')
    A = sp.csr_matrix((ws, (i_o, i_i)), shape=(mesh_out.nC, mesh_in.nC))
    return A

@cython.boundscheck(False)
def _volume_avg_weights(np.float64_t[:] x1, np.float64_t[:] x2):
    cdef int n1 = x1.shape[0]
    cdef int n2 = x2.shape[0]
    cdef np.float64_t[:] xs = np.empty(n1 + n2)
    # Fill xs with uniques and truncate
    cdef int i1, i2, i, ii
    i1 = i2 = i = 0
    while i1<n1 or i2<n2:
        if i1<n1 and i2<n2:
            if x1[i1]<x2[i2]:
                xs[i] = x1[i1]
                i1 += 1
            elif x1[i1]>x2[i2]:
                xs[i] = x2[i2]
                i2 += 1
            else:
                xs[i] = x1[i1]
                i1 += 1
                i2 += 1
        elif i1<n1 and i2==n2:
            xs[i] = x1[i1]
            i1 += 1
        elif i2<n2 and i1==n1:
            xs[i] = x2[i2]
            i2 += 1
        i += 1
    cdef int nh = i-1
    hs = np.empty(nh)
    ix1 = np.empty(nh, dtype=np.int32)
    ix2 = np.empty(nh, dtype=np.int32)

    cdef np.float64_t[:] _hs = hs
    cdef np.int32_t[:] _ix1 = ix1
    cdef np.int32_t[:] _ix2 = ix2
    cdef np.float64_t center

    i1 = i2 = ii = 0
    for i in range(nh):
        center = 0.5*(xs[i]+xs[i+1])
        if x2[0] <= center and center <= x2[n2-1]:
            _hs[ii] = xs[i+1]-xs[i]
            while i1<n1-1 and center>=x1[i1]:
                i1 += 1
            while i2<n2-1 and center>=x2[i2]:
                i2 += 1
            _ix1[ii] = min(max(i1-1, 0), n1-1)
            _ix2[ii] = min(max(i2-1, 0), n2-1)
            ii += 1

    hs = hs[:ii]
    ix1 = ix1[:ii]
    ix2 = ix2[:ii]
    return hs, ix1, ix2
