#include <algorithm>
#include <limits>
#include "geom.h"
#include <algorithm>
#include <limits>

// Define the 3D cross product as a pre-processor macro
#define CROSS3D(e0, e1, out) \
    out[0] = e0[1] * e1[2] - e0[2] * e1[1]; \
    out[1] = e0[2] * e1[0] - e0[0] * e1[2]; \
    out[2] = e0[0] * e1[1] - e0[1] * e1[0];

// simple geometric objects for intersection tests with an aabb

Geometric::Geometric(){
    dim = 0;
}

Geometric::Geometric(int_t dim){
    this->dim = dim;
}

Ball::Ball() : Geometric(){
    x0 = NULL;
    r = 0;
    rsq = 0;
}

Ball::Ball(int_t dim, double* x0, double r) : Geometric(dim){
    this->x0 = x0;
    this->r = r;
    this->rsq = r * r;
}

bool Ball::intersects_cell(double *a, double *b) const{
    // check if I intersect the ball
    double dx;
    double r2_test = 0.0;
    for(int_t i=0; i<dim; ++i){
        dx = std::max(a[i], std::min(x0[i], b[i])) - x0[i];
        r2_test += dx * dx;
    }
    return r2_test < rsq;
}

Line::Line() : Geometric(){
    x0 = NULL;
    x1 = NULL;
    for(int_t i=0; i<3; ++i) inv_dx[i] = 1;
}

Line::Line(int_t dim, double* x0, double *x1) : Geometric(dim){
    this->x0 = x0;
    this->x1 = x1;
    for(int_t i=0; i<dim; ++i){
        inv_dx[i] = 1.0/(x1[i] - x0[i]);
    }
}

bool Line::intersects_cell(double *a, double *b) const{
    double t_near = -std::numeric_limits<double>::infinity();
    double t_far = std::numeric_limits<double>::infinity();
    double t0, t1;

    for(int_t i=0; i<dim; ++i){
        // do a quick test if the line has no chance of intersecting the aabb
        if(x0[i] == x1[i] && (x0[i] < a[i] || x0[i] > b[i])){
            return false;
        }
        if(std::max(x0[i], x1[i]) < a[i]){
            return false;
        }
        if(std::min(x0[i], x1[i]) > b[i]){
            return false;
        }
        if (x0[i] != x1[i]){
            t0 = (a[i] - x0[i]) * inv_dx[i];
            t1 = (b[i] - x0[i]) * inv_dx[i];
            if (t0 > t1){
                std::swap(t0, t1);
            }
            t_near = std::max(t_near, t0);
            t_far = std::min(t_far, t1);
            if (t_near > t_far || t_far < 0 || t_near > 1){
                return false;
            }
        }
    }
    return true;
}

Box::Box() : Geometric(){
    x0 = NULL;
    x1 = NULL;
}

Box::Box(int_t dim, double* x0, double *x1) : Geometric(dim){
    this->x0 = x0;
    this->x1 = x1;
}

bool Box::intersects_cell(double *a, double *b) const{
    for(int_t i=0; i<dim; ++i){
        if(std::max(x0[i], x1[i]) < a[i]){
            return false;
        }
        if(std::min(x0[i], x1[i]) > b[i]){
            return false;
        }
    }
    return true;
}

Plane::Plane() : Geometric(){
    origin = NULL;
    normal = NULL;
}

Plane::Plane(int_t dim, double* origin, double *normal) : Geometric(dim){
    this->origin = origin;
    this->normal = normal;
}

bool Plane::intersects_cell(double *a, double *b) const{
    double center;
    double half_width;
    double s = 0.0;
    double r = 0.0;
    for(int_t i=0;i<dim;++i){
        center = (b[i] + a[i]) * 0.5;
        half_width = center - a[i];
        r += half_width * std::abs(normal[i]);
        s += normal[i] * (center - origin[i]);
    }
    return std::abs(s) <= r;
}

Triangle::Triangle() : Geometric(){
    x0 = NULL;
    x1 = NULL;
    x2 = NULL;
    for(int_t i=0; i<3; ++i){
        e0[i] = 0.0;
        e1[i] = 0.0;
        e2[i] = 0.0;
        normal[i] = 0.0;
    }
}

Triangle::Triangle(int_t dim, double* x0, double *x1, double *x2) : Geometric(dim){
    this->x0 = x0;
    this->x1 = x1;
    this->x2 = x2;

    for(int_t i=0; i<dim; ++i){
      e0[i] = x1[i] - x0[i];
      e1[i] = x2[i] - x1[i];
      e2[i] = x2[i] - x0[i];
    }
    if(dim > 2){
        normal[0] = e0[1] * e1[2] - e0[2] * e1[1];
        normal[1] = e0[2] * e1[0] - e0[0] * e1[2];
        normal[2] = e0[0] * e1[1] - e0[1] * e1[0];
    }
}

bool Triangle::intersects_cell(double *a, double *b) const{
    double center;
    double v0[3], v1[3], v2[3], half[3];
    double vmin, vmax;
    double p0, p1, p2, pmin, pmax, rad;
    for(int_t i=0; i < dim; ++i){
        center = 0.5 * (b[i] + a[i]);
        v0[i] = x0[i] - center;
        v1[i] = x1[i] - center;
        vmin = std::min(v0[i], v1[i]);
        vmax = std::max(v0[i], v1[i]);
        v2[i] = x2[i] - center;
        vmin = std::min(vmin, v2[i]);
        vmax = std::max(vmax, v2[i]);
        half[i] = center - a[i];

        // Bounding box check
        if (vmin > half[i] || vmax < -half[i]){
            return false;
        }
    }
    // first do the 3 edge cross tests that apply in 2D and 3D

    // edge 0 cross z_hat
    //p0 = e0[1] * v0[0] - e0[0] * v0[1];
    p1 = e0[1] * v1[0] - e0[0] * v1[1];
    p2 = e0[1] * v2[0] - e0[0] * v2[1];
    pmin = std::min(p1, p2);
    pmax = std::max(p1, p2);
    rad = std::abs(e0[1]) * half[0] + std::abs(e0[0]) * half[1];
    if (pmin > rad || pmax < -rad){
        return false;
    }

    // edge 1 cross z_hat
    p0 = e1[1] * v0[0] - e1[0] * v0[1];
    p1 = e1[1] * v1[0] - e1[0] * v1[1];
    //p2 = e1[1] * v2[0] - e1[0] * v2[1];
    pmin = std::min(p0, p1);
    pmax = std::max(p0, p1);
    rad = std::abs(e1[1]) * half[0] + std::abs(e1[0]) * half[1];
    if (pmin > rad || pmax < -rad){
        return false;
    }

    // edge 2 cross z_hat
    //p0 = e2[1] * v0[0] - e2[0] * v0[1];
    p1 = e2[1] * v1[0] - e2[0] * v1[1];
    p2 = e2[1] * v2[0] - e2[0] * v2[1];
    pmin = std::min(p1, p2);
    pmax = std::max(p1, p2);
    rad = std::abs(e2[1]) * half[0] + std::abs(e2[0]) * half[1];
    if (pmin > rad || pmax < -rad){
        return false;
    }

    if(dim > 2){
        // edge 0 cross x_hat
        p0 = e0[2] * v0[1] - e0[1] * v0[2];
        //p1 = e0[2] * v1[1] - e0[1] * v1[2];
        p2 = e0[2] * v2[1] - e0[1] * v2[2];
        pmin = std::min(p0, p2);
        pmax = std::max(p0, p2);
        rad = std::abs(e0[2]) * half[1] + std::abs(e0[1]) * half[2];
        if (pmin > rad || pmax < -rad){
            return false;
        }
        // edge 0 cross y_hat
        p0 = -e0[2] * v0[0] + e0[0] * v0[2];
        //p1 = -e0[2] * v1[0] + e0[0] * v1[2];
        p2 = -e0[2] * v2[0] + e0[0] * v2[2];
        pmin = std::min(p0, p2);
        pmax = std::max(p0, p2);
        rad = std::abs(e0[2]) * half[0] + std::abs(e0[0]) * half[2];
        if (pmin > rad || pmax < -rad){
            return false;
        }
        // edge 1 cross x_hat
        p0 = e1[2] * v0[1] - e1[1] * v0[2];
        //p1 = e1[2] * v1[1] - e1[1] * v1[2];
        p2 = e1[2] * v2[1] - e1[1] * v2[2];
        pmin = std::min(p0, p2);
        pmax = std::max(p0, p2);
        rad = std::abs(e1[2]) * half[1] + std::abs(e1[1]) * half[2];
        if (pmin > rad || pmax < -rad){
            return false;
        }
        // edge 1 cross y_hat
        p0 = -e1[2] * v0[0] + e1[0] * v0[2];
        //p1 = -e1[2] * v1[0] + e1[0] * v1[2];
        p2 = -e1[2] * v2[0] + e1[0] * v2[2];
        pmin = std::min(p0, p2);
        pmax = std::max(p0, p2);
        rad = std::abs(e1[2]) * half[0] + std::abs(e1[0]) * half[2];
        if (pmin > rad || pmax < -rad){
            return false;
        }
        // edge 2 cross x_hat
        p0 = e2[2] * v0[1] - e2[1] * v0[2];
        p1 = e2[2] * v1[1] - e2[1] * v1[2];
        //p2 = e2[2] * v2[1] - e2[1] * v2[2];
        pmin = std::min(p0, p1);
        pmax = std::max(p0, p1);
        rad = std::abs(e2[2]) * half[1] + std::abs(e2[1]) * half[2];
        if (pmin > rad || pmax < -rad){
            return false;
        }
        // edge 2 cross y_hat
        p0 = -e2[2] * v0[0] + e2[0] * v0[2];
        p1 = -e2[2] * v1[0] + e2[0] * v1[2];
        //p2 = -e2[2] * v2[0] + e2[0] * v2[2];
        pmin = std::min(p0, p1);
        pmax = std::max(p0, p1);
        rad = std::abs(e2[2]) * half[0] + std::abs(e2[0]) * half[2];
        if (pmin > rad || pmax < -rad){
            return false;
        }

        // triangle normal axis
        pmin = 0.0;
        pmax = 0.0;
        for(int_t i=0; i<dim; ++i){
            if(normal[i] > 0){
                pmin += normal[i] * (-half[i] - v0[i]);
                pmax += normal[i] * (half[i] - v0[i]);
            }else{
                pmin += normal[i] * (half[i] - v0[i]);
                pmax += normal[i] * (-half[i] - v0[i]);
            }
        }
        if (pmin > 0 || pmax < 0){
            return false;
        }
    }
    return true;
}

VerticalTriangularPrism::VerticalTriangularPrism() : Triangle(){
    h = 0;
}

VerticalTriangularPrism::VerticalTriangularPrism(int_t dim, double* x0, double *x1, double *x2, double h) : Triangle(dim, x0, x1, x2){
    this->h = h;
}

bool VerticalTriangularPrism::intersects_cell(double *a, double *b) const{
    double center;
    double v0[3], v1[3], v2[3], half[3];
    double vmin, vmax;
    double p0, p1, p2, p3, pmin, pmax, rad;
    for(int_t i=0; i < dim; ++i){
        center = 0.5 * (a[i] + b[i]);
        v0[i] = x0[i] - center;
        v1[i] = x1[i] - center;
        vmin = std::min(v0[i], v1[i]);
        vmax = std::max(v0[i], v1[i]);
        v2[i] = x2[i] - center;
        vmin = std::min(vmin, v2[i]);
        vmax = std::max(vmax, v2[i]);
        if(i == 2){
            vmax += h;
        }
        half[i] = center - a[i];

        // Bounding box check
        if (vmin > half[i] || vmax < -half[i]){
            return false;
        }
    }
    // first do the 3 edge cross tests that apply in 2D and 3D

    // edge 0 cross z_hat
    //p0 = e0[1] * v0[0] - e0[0] * v0[1];
    p1 = e0[1] * v1[0] - e0[0] * v1[1];
    p2 = e0[1] * v2[0] - e0[0] * v2[1];
    pmin = std::min(p1, p2);
    pmax = std::max(p1, p2);
    rad = std::abs(e0[1]) * half[0] + std::abs(e0[0]) * half[1];
    if (pmin > rad || pmax < -rad){
        return false;
    }

    // edge 1 cross z_hat
    p0 = e1[1] * v0[0] - e1[0] * v0[1];
    p1 = e1[1] * v1[0] - e1[0] * v1[1];
    //p2 = e1[1] * v2[0] - e1[0] * v2[1];
    pmin = std::min(p0, p1);
    pmax = std::max(p0, p1);
    rad = std::abs(e1[1]) * half[0] + std::abs(e1[0]) * half[1];
    if (pmin > rad || pmax < -rad){
        return false;
    }

    // edge 2 cross z_hat
    //p0 = e2[1] * v0[0] - e2[0] * v0[1];
    p1 = e2[1] * v1[0] - e2[0] * v1[1];
    p2 = e2[1] * v2[0] - e2[0] * v2[1];
    pmin = std::min(p1, p2);
    pmax = std::max(p1, p2);
    rad = std::abs(e2[1]) * half[0] + std::abs(e2[0]) * half[1];
    if (pmin > rad || pmax < -rad){
        return false;
    }

    // edge 0 cross x_hat
    p0 = e0[2] * v0[1] - e0[1] * v0[2];
    p1 = e0[2] * v0[1] - e0[1] * (v0[2] + h);
    p2 = e0[2] * v2[1] - e0[1] * v2[2];
    p3 = e0[2] * v2[1] - e0[1] * (v2[2] + h);
    pmin = std::min(std::min(std::min(p0, p1), p2), p3);
    pmax = std::max(std::max(std::max(p0, p1), p2), p3);
    rad = std::abs(e0[2]) * half[1] + std::abs(e0[1]) * half[2];
    if (pmin > rad || pmax < -rad){
        return false;
    }
    // edge 0 cross y_hat
    p0 = -e0[2] * v0[0] + e0[0] * v0[2];
    p1 = -e0[2] * v0[0] + e0[0] * (v0[2] + h);
    p2 = -e0[2] * v2[0] + e0[0] * v2[2];
    p3 = -e0[2] * v2[0] + e0[0] * (v2[2] + h);
    pmin = std::min(std::min(std::min(p0, p1), p2), p3);
    pmax = std::max(std::max(std::max(p0, p1), p2), p3);
    rad = std::abs(e0[2]) * half[0] + std::abs(e0[0]) * half[2];
    if (pmin > rad || pmax < -rad){
        return false;
    }
    // edge 1 cross x_hat
    p0 = e1[2] * v0[1] - e1[1] * v0[2];
    p1 = e1[2] * v0[1] - e1[1] * (v0[2] + h);
    p2 = e1[2] * v2[1] - e1[1] * v2[2];
    p3 = e1[2] * v2[1] - e1[1] * (v2[2] + h);
    pmin = std::min(std::min(std::min(p0, p1), p2), p3);
    pmax = std::max(std::max(std::max(p0, p1), p2), p3);
    rad = std::abs(e1[2]) * half[1] + std::abs(e1[1]) * half[2];
    if (pmin > rad || pmax < -rad){
        return false;
    }
    // edge 1 cross y_hat
    p0 = -e1[2] * v0[0] + e1[0] * v0[2];
    p1 = -e1[2] * v0[0] + e1[0] * (v0[2] + h);
    p2 = -e1[2] * v2[0] + e1[0] * v2[2];
    p3 = -e1[2] * v2[0] + e1[0] * (v2[2] + h);
    pmin = std::min(std::min(std::min(p0, p1), p2), p3);
    pmax = std::max(std::max(std::max(p0, p1), p2), p3);
    rad = std::abs(e1[2]) * half[0] + std::abs(e1[0]) * half[2];
    if (pmin > rad || pmax < -rad){
        return false;
    }
    // edge 2 cross x_hat
    p0 = e2[2] * v0[1] - e2[1] * v0[2];
    p1 = e2[2] * v0[1] - e2[1] * (v0[2] + h);
    p2 = e2[2] * v1[1] - e2[1] * v1[2];
    p3 = e2[2] * v1[1] - e2[1] * (v1[2] + h);
    pmin = std::min(std::min(std::min(p0, p1), p2), p3);
    pmax = std::max(std::max(std::max(p0, p1), p2), p3);
    rad = std::abs(e2[2]) * half[1] + std::abs(e2[1]) * half[2];
    if (pmin > rad || pmax < -rad){
        return false;
    }
    // edge 2 cross y_hat
    p0 = -e2[2] * v0[0] + e2[0] * v0[2];
    p1 = -e2[2] * v0[0] + e2[0] * (v0[2] + h);
    p2 = -e2[2] * v1[0] + e2[0] * v1[2];
    p3 = -e2[2] * v1[0] + e2[0] * (v1[2] + h);
    pmin = std::min(std::min(std::min(p0, p1), p2), p3);
    pmax = std::max(std::max(std::max(p0, p1), p2), p3);
    rad = std::abs(e2[2]) * half[0] + std::abs(e2[0]) * half[2];
    if (pmin > rad || pmax < -rad){
        return false;
    }

    // triangle normal axis
    p0 = normal[0] * v0[0] + normal[1] * v0[1] + normal[2] * v0[2];
    p1 = normal[0] * v0[0] + normal[1] * v0[1] + normal[2] * (v0[2] + h);
    pmin = std::min(p0, p1);
    pmax = std::max(p0, p1);
    rad = std::abs(normal[0]) * half[0] + std::abs(normal[1]) * half[1] + std::abs(normal[2]) * half[2];
    if (pmin > rad || pmax < -rad){
        return false;
    }
    // the axes defined by the three vertical prism faces
    // should already be tested by the e0, e1, e2 cross z_hat tests
    return true;
}

Tetrahedron::Tetrahedron() : Geometric(){
    x0 = NULL;
    x1 = NULL;
    x2 = NULL;
    x3 = NULL;
    for(int_t i=0; i<6; ++i){
        for(int_t j=0; j<3; ++j){
            edge_tans[i][j] = 0.0;
        }
    }
    for(int_t i=0; i<4; ++i){
        for(int_t j=0; j<3; ++j){
            face_normals[i][j] = 0.0;
        }
    }
}

Tetrahedron::Tetrahedron(int_t dim, double* x0, double *x1, double *x2, double *x3) : Geometric(dim){
    this->x0 = x0;
    this->x1 = x1;
    this->x2 = x2;
    this->x3 = x3;
    for(int_t i=0; i<dim; ++i){
        edge_tans[0][i] = x1[i] - x0[i];
        edge_tans[1][i] = x2[i] - x0[i];
        edge_tans[2][i] = x2[i] - x1[i];
        edge_tans[3][i] = x3[i] - x0[i];
        edge_tans[4][i] = x3[i] - x1[i];
        edge_tans[5][i] = x3[i] - x2[i];
    }
    // cross e0, e1 (x0, x1, x2)
    CROSS3D(edge_tans[0], edge_tans[1], face_normals[0])

    // cross e0, e3 (x0, x1, x3)
    CROSS3D(edge_tans[0], edge_tans[3], face_normals[1])

    // cross e1, e3 (x0, x2, x3)
    CROSS3D(edge_tans[1], edge_tans[3], face_normals[2])

    // cross e2, e5 (x1, x2, x3)
    CROSS3D(edge_tans[2], edge_tans[5], face_normals[3])
}

bool Tetrahedron::intersects_cell(double *a, double *b) const{
    double v0[3], v1[3], v2[3], v3[3], half[3];
    double p0, p1, p2, p3, pmin, pmax, rad;
    double center;
    for(int_t i=0; i < dim; ++i){
        center = 0.5 * (a[i] + b[i]);
        v0[i] = x0[i] - center;
        v1[i] = x1[i] - center;
        v2[i] = x2[i] - center;
        v3[i] = x3[i] - center;
        half[i] = center - a[i];
        pmin = std::min(std::min(std::min(v0[i], v1[i]), v2[i]), v3[i]);
        pmax = std::max(std::max(std::max(v0[i], v1[i]), v2[i]), v3[i]);
        // Bounding box check
        if (pmin > half[i] || pmax < -half[i]){
            return false;
        }
    }
    // first do the 3 edge cross tests that apply in 2D and 3D
    const double *axis;

    for(int_t i=0; i<6; ++i){
        // edge cross [1, 0, 0]
        p0 = edge_tans[i][2] * v0[1] - edge_tans[i][1] * v0[2];
        p1 = edge_tans[i][2] * v1[1] - edge_tans[i][1] * v1[2];
        p2 = edge_tans[i][2] * v2[1] - edge_tans[i][1] * v2[2];
        p3 = edge_tans[i][2] * v3[1] - edge_tans[i][1] * v3[2];
        pmin = std::min(std::min(std::min(p0, p1), p2), p3);
        pmax = std::max(std::max(std::max(p0, p1), p2), p3);
        rad = std::abs(edge_tans[i][2]) * half[1] + std::abs(edge_tans[i][1]) * half[2];
        if (pmin > rad || pmax < -rad){
            return false;
        }

        p0 = -edge_tans[i][2] * v0[0] + edge_tans[i][0] * v0[2];
        p1 = -edge_tans[i][2] * v1[0] + edge_tans[i][0] * v1[2];
        p2 = -edge_tans[i][2] * v2[0] + edge_tans[i][0] * v2[2];
        p3 = -edge_tans[i][2] * v3[0] + edge_tans[i][0] * v3[2];
        pmin = std::min(std::min(std::min(p0, p1), p2), p3);
        pmax = std::max(std::max(std::max(p0, p1), p2), p3);
        rad = std::abs(edge_tans[i][2]) * half[0] + std::abs(edge_tans[i][0]) * half[2];
        if (pmin > rad || pmax < -rad){
            return false;
        }

        p0 = edge_tans[i][1] * v0[0] - edge_tans[i][0] * v0[1];
        p1 = edge_tans[i][1] * v1[0] - edge_tans[i][0] * v1[1];
        p2 = edge_tans[i][1] * v2[0] - edge_tans[i][0] * v2[1];
        p3 = edge_tans[i][1] * v3[0] - edge_tans[i][0] * v3[1];
        pmin = std::min(std::min(std::min(p0, p1), p2), p3);
        pmax = std::max(std::max(std::max(p0, p1), p2), p3);
        rad = std::abs(edge_tans[i][1]) * half[0] + std::abs(edge_tans[i][0]) * half[1];
        if (pmin > rad || pmax < -rad){
            return false;
        }
    }
    // triangle face normals
    for(int_t i=0; i<4; ++i){
        axis = face_normals[i];
        p0 = axis[0] * v0[0] + axis[1] * v0[1] + axis[2] * v0[2];
        p1 = axis[0] * v1[0] + axis[1] * v1[1] + axis[2] * v1[2];
        p2 = axis[0] * v2[0] + axis[1] * v2[1] + axis[2] * v2[2];
        p3 = axis[0] * v3[0] + axis[1] * v3[1] + axis[2] * v3[2];
        pmin = std::min(std::min(std::min(p0, p1), p2), p3);
        pmax = std::max(std::max(std::max(p0, p1), p2), p3);
        rad = std::abs(axis[0]) * half[0] + std::abs(axis[1]) * half[1] + std::abs(axis[2]) * half[2];
        if (pmin > rad || pmax < -rad){
            return false;
        }
    }
    return true;
}