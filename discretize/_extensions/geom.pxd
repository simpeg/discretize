from libcpp cimport bool

cdef extern from "geom.h":
    ctypedef int int_t
    cdef cppclass Ball:
        Ball(int_t dim, double * x0, double r)

    cdef cppclass Line:
        Line(int_t dim, double * x0, double *x1, bool segment)

    cdef cppclass Box:
        Box(int_t dim, double * x0, double *x1)

    cdef cppclass Plane:
        Plane(int_t dim, double * origin, double *normal)

    cdef cppclass Triangle:
        Triangle(int_t dim, double * x0, double *x1, double *x2)

    cdef cppclass VerticalTriangularPrism:
        VerticalTriangularPrism(int_t dim, double * x0, double *x1, double *x2, double h)

    cdef cppclass Tetrahedron:
        Tetrahedron(int_t dim, double * x0, double *x1, double *x2, double *x3)