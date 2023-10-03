cimport numpy as np
cdef np.int64_t _bisect_left(np.float64_t[:] a, np.float64_t x) nogil
cdef np.int64_t _bisect_right(np.float64_t[:] a, np.float64_t x) nogil
