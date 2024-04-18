#ifndef __GEOM_H
#define __GEOM_H
// simple geometric objects for intersection tests with an aabb

typedef std::size_t int_t;

class Geometric{
    public:
        int_t dim;

        Geometric();
        Geometric(int_t dim);
        virtual bool intersects_cell(double *a, double *b) const = 0;
};

class Ball : public Geometric{
    public:
        double *x0;
        double r;
        double rsq;

        Ball();
        Ball(int_t dim, double* x0, double r);
        virtual bool intersects_cell(double *a, double *b) const;
};

class Line : public Geometric{
    public:
        double *x0;
        double *x1;
        double inv_dx[3];

        Line();
        Line(int_t dim, double* x0, double *x1);
        virtual bool intersects_cell(double *a, double *b) const;
};

class Box : public Geometric{
    public:
        double *x0;
        double *x1;

        Box();
        Box(int_t dim, double* x0, double *x1);
        virtual bool intersects_cell(double *a, double *b) const;
};

class Plane : public Geometric{
    public:
        double *origin;
        double *normal;

        Plane();
        Plane(int_t dim, double* origin, double *normal);
        virtual bool intersects_cell(double *a, double *b) const;
};

class Triangle : public Geometric{
    public:
        double *x0;
        double *x1;
        double *x2;
        double e0[3];
        double e1[3];
        double e2[3];
        double normal[3];

        Triangle();
        Triangle(int_t dim, double* x0, double *x1, double *x2);
        virtual bool intersects_cell(double *a, double *b) const;
};

class VerticalTriangularPrism : public Triangle{
    public:
        double h;

        VerticalTriangularPrism();
        VerticalTriangularPrism(int_t dim, double* x0, double *x1, double *x2, double h);
        virtual bool intersects_cell(double *a, double *b) const;
};

class Tetrahedron : public Geometric{
    public:
        double *x0;
        double *x1;
        double *x2;
        double *x3;
        double edge_tans[6][3];
        double face_normals[4][3];

        Tetrahedron();
        Tetrahedron(int_t dim, double* x0, double *x1, double *x2, double *x3);
        virtual bool intersects_cell(double *a, double *b) const;
 };

#endif