# from __future__ import division
import numpy as np
import cython
cimport numpy as np
# from libcpp.vector cimport vector

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
