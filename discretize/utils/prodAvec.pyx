import numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def prodAvec(float [:, :] A, double [:] x):
    """
    Do the product A times a vector in cython with OpenMP
    """
    cdef double [:] vec = np.zeros(A.shape[1])
    cdef double [:] y =  np.zeros(A.shape[0])
    cdef int ii, jj
    cdef int I, J

    I = A.shape[0]
    J = A.shape[1]
    for ii in range(I):
        for jj in range(J):
            y[ii] += A[ii,jj]*x[jj]

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def prodAtvec(float [:, :] A, double [:] x):
    """
    Do the product A transposed times a vector in
    cython with OpenMP
    """
    cdef double [:] vec = np.zeros(A.shape[0])
    cdef double [:] y =  np.zeros(A.shape[1])
    cdef int ii, jj
    cdef int I, J

    I = A.shape[1]
    J = A.shape[0]
    for ii in range(I):
        for jj in range(J):
            y[ii] += A[jj, ii]*x[jj]
    return y


