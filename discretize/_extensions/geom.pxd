from libcpp cimport bool

cdef extern from "geom.h":
    ctypedef int int_t
    cdef cppclass Ball:
        Ball() except +
        Ball(int_t dim, double * x0, double r) except +

    cdef cppclass Line:
        Line() except +
        Line(int_t dim, double * x0, double *x1) except +

    cdef cppclass Box:
        Box() except +
        Box(int_t dim, double * x0, double *x1) except +

    cdef cppclass Plane:
        Plane() except +
        Plane(int_t dim, double * origin, double *normal) except +

    cdef cppclass Triangle:
        Triangle() except +
        Triangle(int_t dim, double * x0, double *x1, double *x2) except +

    cdef cppclass VerticalTriangularPrism:
        VerticalTriangularPrism() except +
        VerticalTriangularPrism(int_t dim, double * x0, double *x1, double *x2, double h) except +

    cdef cppclass Tetrahedron:
        Tetrahedron() except +
        Tetrahedron(int_t dim, double * x0, double *x1, double *x2, double *x3) except +