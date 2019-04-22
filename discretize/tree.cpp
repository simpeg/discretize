#include <vector>
#include <map>
#include "tree.h"
#include <iostream>
#include <algorithm>

Node::Node(){
    location_ind[0] = 0;
    location_ind[1] = 0;
    location_ind[2] = 0;
    location[0] = 0;
    location[1] = 0;
    location[2] = 0;
    key = 0;
    reference = 0;
    index = 0;
    hanging = false;
    parents[0] = NULL;
    parents[1] = NULL;
    parents[2] = NULL;
    parents[3] = NULL;
};

Node::Node(int_t ix, int_t iy, int_t iz, double* xs, double *ys, double *zs){
    location_ind[0] = ix;
    location_ind[1] = iy;
    location_ind[2] = iz;
    location[0] = xs[ix];
    location[1] = ys[iy];
    location[2] = zs[iz];
    key = key_func(ix, iy, iz);
    reference = 0;
    index = 0;
    hanging = false;
    parents[0] = NULL;
    parents[1] = NULL;
    parents[2] = NULL;
    parents[3] = NULL;
};

Edge::Edge(){
    location_ind[0] = 0;
    location_ind[1] = 0;
    location_ind[2] = 0;
    location[0] = 0;
    location[1] = 0;
    location[2] = 0;
    key = 0;
    index = 0;
    reference = 0;
    length = 0.0;
    hanging = false;
    points[0] = NULL;
    points[1] = NULL;
    parents[0] = NULL;
    parents[1] = NULL;
};

Edge::Edge(Node& p1, Node& p2){
      points[0] = &p1;
      points[1] = &p2;
      int_t ix, iy, iz;
      ix = (p1.location_ind[0]+p2.location_ind[0])/2;
      iy = (p1.location_ind[1]+p2.location_ind[1])/2;
      iz = (p1.location_ind[2]+p2.location_ind[2])/2;
      key = key_func(ix, iy, iz);
      location_ind[0] = ix;
      location_ind[1] = iy;
      location_ind[2] = iz;

      location[0] = (p1[0]+p2[0]) * 0.5;
      location[1] = (p1[1]+p2[1]) * 0.5;
      location[2] = (p1[2]+p2[2]) * 0.5;
      length = (p2[0]-p1[0])
                + (p2[1]-p1[1])
                + (p2[2]-p1[2]);
      reference = 0;
      index = 0;
      hanging = false;
      parents[0] = NULL;
      parents[1] = NULL;
}

Face::Face(){
    location_ind[0] = 0;
    location_ind[1] = 0;
    location_ind[2] = 0;
    location[0] = 0.0;
    location[1] = 0.0;
    location[2] = 0.0;
    key = 0;
    reference = 0;
    index = 0;
    area = 0;
    hanging = false;
    points[0] = NULL;
    points[1] = NULL;
    points[2] = NULL;
    points[3] = NULL;
    edges[0] = NULL;
    edges[1] = NULL;
    edges[2] = NULL;
    edges[3] = NULL;
    parent = NULL;
}

Face::Face(Node& p1, Node& p2, Node& p3, Node& p4){
    points[0] = &p1;
    points[1] = &p2;
    points[2] = &p3;
    points[3] = &p4;
    int_t ix, iy, iz;
    ix = (p1.location_ind[0]+p2.location_ind[0]+p3.location_ind[0]+p4.location_ind[0])/4;
    iy = (p1.location_ind[1]+p2.location_ind[1]+p3.location_ind[1]+p4.location_ind[1])/4;
    iz = (p1.location_ind[2]+p2.location_ind[2]+p3.location_ind[2]+p4.location_ind[2])/4;
    key = key_func(ix, iy, iz);
    location_ind[0] = ix;
    location_ind[1] = iy;
    location_ind[2] = iz;

    location[0] = (p1[0]+p2[0]+p3[0]+p4[0]) * 0.25;
    location[1] = (p1[1]+p2[1]+p3[1]+p4[1]) * 0.25;
    location[2] = (p1[2]+p2[2]+p3[2]+p4[2]) * 0.25;
    area = ((p2[0]-p1[0]) + (p2[1]-p1[1]) + (p2[2]-p1[2])) *
           ((p3[0]-p1[0]) + (p3[1]-p1[1]) + (p3[2]-p1[2]));
    reference = 0;
    index = 0;
    hanging = false;
    parent = NULL;
    edges[0] = NULL;
    edges[1] = NULL;
    edges[2] = NULL;
    edges[3] = NULL;
}

Node * set_default_node(node_map_t& nodes, int_t x, int_t y, int_t z,
                        double *xs, double *ys, double *zs){
  int_t key = key_func(x, y, z);
  Node * point;
  if(nodes.count(key) == 0){
    point = new Node(x, y, z, xs, ys, zs);
    nodes[key] = point;
  }
  else{
    point = nodes[key];
  }
  return point;
}

Edge * set_default_edge(edge_map_t& edges, Node& p1, Node& p2){
  int_t xC = (p1.location_ind[0]+p2.location_ind[0])/2;
  int_t yC = (p1.location_ind[1]+p2.location_ind[1])/2;
  int_t zC = (p1.location_ind[2]+p2.location_ind[2])/2;
  int_t key = key_func(xC, yC, zC);
  Edge * edge;
  if(edges.count(key) == 0){
    edge = new Edge(p1, p2);
    edges[key] = edge;
  }
  else{
    edge = edges[key];
  }
  return edge;
};

Face * set_default_face(face_map_t& faces, Node& p1, Node& p2, Node& p3, Node& p4){
    int_t x, y, z, key;
    x = (p1.location_ind[0]+p2.location_ind[0]+p3.location_ind[0]+p4.location_ind[0])/4;
    y = (p1.location_ind[1]+p2.location_ind[1]+p3.location_ind[1]+p4.location_ind[1])/4;
    z = (p1.location_ind[2]+p2.location_ind[2]+p3.location_ind[2]+p4.location_ind[2])/4;
    key = key_func(x, y, z);
    Face * face;
    if(faces.count(key) == 0){
        face = new Face(p1, p2, p3, p4);
        faces[key] = face;
    }
    else{
        face = faces[key];
    }
    return face;
}

Cell::Cell(Node *pts[8], int_t ndim, int_t maxlevel, function func){
    n_dim = ndim;
    int_t n_points = 1<<n_dim;
    for(int_t i = 0; i < n_points; ++i)
        points[i] = pts[i];
    level = 0;
    max_level = maxlevel;
    parent = NULL;
    test_func = func;
    Node p1 = *pts[0];
    Node p2 = *pts[n_points - 1];
    location_ind[0] = (p1.location_ind[0]+p2.location_ind[0])/2;
    location_ind[1] = (p1.location_ind[1]+p2.location_ind[1])/2;
    location_ind[2] = (p1.location_ind[2]+p2.location_ind[2])/2;
    location[0] = (p1[0]+p2[0]) * 0.5;
    location[1] = (p1[1]+p2[1]) * 0.5;
    location[2] = (p1[2]+p2[2]) * 0.5;
    volume = (p2[0]-p1[0]) * (p2[1]-p1[1]);
    if(n_dim==3)
        volume *= (p2[2] - p1[2]);
    key = key_func(location_ind[0], location_ind[1], location_ind[2]);
    for(int_t i = 0; i < n_points; ++i)
        children[i] = NULL;
    for(int_t i = 0; i < 2*n_dim; ++i)
        neighbors[i] = NULL;
};

Cell::Cell(Node *pts[8], Cell *parent){
    n_dim = parent->n_dim;
    int_t n_points = 1<<n_dim;
    for(int_t i = 0; i < n_points; ++i)
        points[i] = pts[i];
    level = parent->level + 1;
    max_level = parent->max_level;
    test_func = parent->test_func;
    Node p1 = *pts[0];
    Node p2 = *pts[n_points - 1];
    location_ind[0] = (p1.location_ind[0]+p2.location_ind[0])/2;
    location_ind[1] = (p1.location_ind[1]+p2.location_ind[1])/2;
    location_ind[2] = (p1.location_ind[2]+p2.location_ind[2])/2;
    location[0] = (p1[0]+p2[0]) * 0.5;
    location[1] = (p1[1]+p2[1]) * 0.5;
    location[2] = (p1[2]+p2[2]) * 0.5;

    volume = (p2[0]-p1[0]) * (p2[1]-p1[1]);
    if(n_dim == 3)
        volume *= (p2[2] - p1[2]);

    key = key_func(location_ind[0], location_ind[1], location_ind[2]);

    for(int_t i = 0; i < n_points; ++i)
        children[i] = NULL;
    for(int_t i = 0; i < 2*n_dim; ++i)
        neighbors[i] = NULL;
};

void Cell::spawn(node_map_t& nodes, Cell *kids[8], double *xs, double *ys, double *zs){
    /*      z0              z0+dz/2          z0+dz
        p03--p13--p04    p20--p21--p22   p07--p27--p08
        |     |    |     |     |    |    |     |    |
        p10--p11--p12    p17--p18--p19   p24--p25--p26
        |     |    |     |     |    |    |     |    |
        p01--p09--p02    p14--p15--p16   p05--p23--p06
    */
    Node *p1 = points[0];
    Node *p2 = points[1];
    Node *p3 = points[2];
    Node *p4 = points[3];

    int_t x0, y0, xC, yC, xF, yF, z0;

    x0 = p1->location_ind[0];
    y0 = p1->location_ind[1];
    xF = p4->location_ind[0];
    yF = p4->location_ind[1];
    z0 = p1->location_ind[2];
    xC = location_ind[0];
    yC = location_ind[1];

    Node *p9, *p10, *p11, *p12, *p13;
    p9  = set_default_node(nodes, xC, y0, z0, xs, ys, zs);
    p10 = set_default_node(nodes, x0, yC, z0, xs, ys, zs);
    p11 = set_default_node(nodes, xC, yC, z0, xs, ys, zs);
    p12 = set_default_node(nodes, xF, yC, z0, xs, ys, zs);
    p13 = set_default_node(nodes, xC, yF, z0, xs, ys, zs);

    //Increment node references for new nodes
    p9->reference += 2;
    p10->reference += 2;
    p11->reference += 4;
    p12->reference += 2;
    p13->reference += 2;

    if(n_dim>2){
        Node *p5 = points[4];
        Node *p6 = points[5];
        Node *p7 = points[6];
        Node *p8 = points[7];

        int_t zC, zF;

        zF = p8->location_ind[2];
        zC = location_ind[2];

        Node *p14, *p15, *p16, *p17, *p18, *p19, *p20, *p21, *p22;
        Node *p23, *p24, *p25, *p26, *p27;

        p14 = set_default_node(nodes, x0, y0, zC, xs, ys, zs);
        p15 = set_default_node(nodes, xC, y0, zC, xs, ys, zs);
        p16 = set_default_node(nodes, xF, y0, zC, xs, ys, zs);
        p17 = set_default_node(nodes, x0, yC, zC, xs, ys, zs);
        p18 = set_default_node(nodes, xC, yC, zC, xs, ys, zs);
        p19 = set_default_node(nodes, xF, yC, zC, xs, ys, zs);
        p20 = set_default_node(nodes, x0, yF, zC, xs, ys, zs);
        p21 = set_default_node(nodes, xC, yF, zC, xs, ys, zs);
        p22 = set_default_node(nodes, xF, yF, zC, xs, ys, zs);

        p23 = set_default_node(nodes, xC, y0, zF, xs, ys, zs);
        p24 = set_default_node(nodes, x0, yC, zF, xs, ys, zs);
        p25 = set_default_node(nodes, xC, yC, zF, xs, ys, zs);
        p26 = set_default_node(nodes, xF, yC, zF, xs, ys, zs);
        p27 = set_default_node(nodes, xC, yF, zF, xs, ys, zs);

        //Increment node references
        p14->reference += 2;
        p15->reference += 4;
        p16->reference += 2;
        p17->reference += 4;
        p18->reference += 8;
        p19->reference += 4;
        p20->reference += 2;
        p21->reference += 4;
        p22->reference += 2;

        p23->reference += 2;
        p24->reference += 2;
        p25->reference += 4;
        p26->reference += 2;
        p27->reference += 2;

        Node * pQC1[8] = { p1, p9,p10,p11,p14,p15,p17,p18};
        Node * pQC2[8] = { p9, p2,p11,p12,p15,p16,p18,p19};
        Node * pQC3[8] = {p10,p11, p3,p13,p17,p18,p20,p21};
        Node * pQC4[8] = {p11,p12,p13, p4,p18,p19,p21,p22};
        Node * pQC5[8] = {p14,p15,p17,p18, p5,p23,p24,p25};
        Node * pQC6[8] = {p15,p16,p18,p19,p23, p6,p25,p26};
        Node * pQC7[8] = {p17,p18,p20,p21,p24,p25, p7,p27};
        Node * pQC8[8] = {p18,p19,p21,p22,p25,p26,p27, p8};

        kids[0] = new Cell(pQC1, this);
        kids[1] = new Cell(pQC2, this);
        kids[2] = new Cell(pQC3, this);
        kids[3] = new Cell(pQC4, this);
        kids[4] = new Cell(pQC5, this);
        kids[5] = new Cell(pQC6, this);
        kids[6] = new Cell(pQC7, this);
        kids[7] = new Cell(pQC8, this);
    }
    else{
        Node * pQC1[8] = { p1, p9,p10,p11, NULL, NULL, NULL, NULL};
        Node * pQC2[8] = { p9, p2,p11,p12, NULL, NULL, NULL, NULL};
        Node * pQC3[8] = {p10,p11, p3,p13, NULL, NULL, NULL, NULL};
        Node * pQC4[8] = {p11,p12,p13, p4, NULL, NULL, NULL, NULL};
        kids[0] = new Cell(pQC1, this);
        kids[1] = new Cell(pQC2, this);
        kids[2] = new Cell(pQC3, this);
        kids[3] = new Cell(pQC4, this);
    }
};

void Cell::set_neighbor(Cell * other, int_t position){
    if(other==NULL){
        return;
    }
    if(level != other->level){
        neighbors[position] = other;
    }else{
        neighbors[position] = other;
        other->neighbors[position^1] = this;
    }
};

void Cell::shift_centers(double *shift){
    for(int_t id = 0; id<n_dim; ++id){
        location[id] += shift[id];
    }
    if(!is_leaf()){
        for(int_t i = 0; i < (1<<n_dim); ++i){
            children[i]->shift_centers(shift);
        }
    }
}

void Cell::insert_cell(node_map_t& nodes, double *new_cell, int_t p_level, double *xs, double *ys, double *zs){
    //Inserts a cell at min(max_level,p_level) that contains the given point
    if(p_level > level){
        // Need to go look in children,
        // Need to spawn children if i don't have any...
        if(is_leaf()){
            divide(nodes, xs, ys, zs, true);
        }
        int ix = new_cell[0] > children[0]->points[3]->location[0];
        int iy = new_cell[1] > children[0]->points[3]->location[1];
        int iz = n_dim>2 && new_cell[2]>children[0]->points[7]->location[2];
        children[ix + 2*iy + 4*iz]->insert_cell(nodes, new_cell, p_level, xs, ys, zs);
    }
};

void Cell::divide(node_map_t& nodes, double* xs, double* ys, double* zs, bool force, bool balance){
    bool do_splitting = false;
    if(level == max_level){
        do_splitting = false;
    }else if(force){
        do_splitting = true;
    }else{
        if(is_leaf()){
            // Only need to call the function if I am a leaf...
            int_t test_level = (*test_func)(this);
            do_splitting = test_level > level;
        }
    }
    if(do_splitting){
        //If i haven't already been split...
        if(is_leaf()){
            spawn(nodes, children, xs, ys, zs);

            //If I need to be split, and my neighbor is below my level
            //Then it needs to be split
            //-x,+x,-y,+y,-z,+z
            if(balance){
                for(int_t i = 0; i < 2*n_dim; ++i){
                    if(neighbors[i] != NULL && neighbors[i]->level < level){
                        neighbors[i]->divide(nodes, xs, ys, zs, true);
                    }
                }
            }

            //Set children's neighbors (first do the easy ones)
            // all of the children live next to each other
            children[0]->set_neighbor(children[1], 1);
            children[0]->set_neighbor(children[2], 3);
            children[1]->set_neighbor(children[3], 3);
            children[2]->set_neighbor(children[3], 1);

            if(n_dim == 3){
                children[4]->set_neighbor(children[5], 1);
                children[4]->set_neighbor(children[6], 3);
                children[5]->set_neighbor(children[7], 3);
                children[6]->set_neighbor(children[7], 1);

                children[0]->set_neighbor(children[4], 5);
                children[1]->set_neighbor(children[5], 5);
                children[2]->set_neighbor(children[6], 5);
                children[3]->set_neighbor(children[7], 5);
            }

            // -x direction
            if(neighbors[0]!=NULL && !(neighbors[0]->is_leaf())){
                children[0]->set_neighbor(neighbors[0]->children[1], 0);
                children[2]->set_neighbor(neighbors[0]->children[3], 0);
            }
            else{
                children[0]->set_neighbor(neighbors[0], 0);
                children[2]->set_neighbor(neighbors[0], 0);
            }
            // +x direction
            if(neighbors[1]!=NULL && !neighbors[1]->is_leaf()){
                children[1]->set_neighbor(neighbors[1]->children[0], 1);
                children[3]->set_neighbor(neighbors[1]->children[2], 1);
            }else{
                children[1]->set_neighbor(neighbors[1], 1);
                children[3]->set_neighbor(neighbors[1], 1);
            }
            // -y direction
            if(neighbors[2]!=NULL && !neighbors[2]->is_leaf()){
                children[0]->set_neighbor(neighbors[2]->children[2], 2);
                children[1]->set_neighbor(neighbors[2]->children[3], 2);
            }else{
                children[0]->set_neighbor(neighbors[2], 2);
                children[1]->set_neighbor(neighbors[2], 2);
            }
            // +y direction
            if(neighbors[3]!=NULL && !neighbors[3]->is_leaf()){
                children[2]->set_neighbor(neighbors[3]->children[0], 3);
                children[3]->set_neighbor(neighbors[3]->children[1], 3);
            }else{
                children[2]->set_neighbor(neighbors[3], 3);
                children[3]->set_neighbor(neighbors[3], 3);
            }
            if(n_dim==3){
                // -x direction
                if(neighbors[0]!=NULL && !(neighbors[0]->is_leaf())){
                    children[4]->set_neighbor(neighbors[0]->children[5], 0);
                    children[6]->set_neighbor(neighbors[0]->children[7], 0);
                }
                else{
                    children[4]->set_neighbor(neighbors[0], 0);
                    children[6]->set_neighbor(neighbors[0], 0);
                }
                // +x direction
                if(neighbors[1]!=NULL && !neighbors[1]->is_leaf()){
                    children[5]->set_neighbor(neighbors[1]->children[4], 1);
                    children[7]->set_neighbor(neighbors[1]->children[6], 1);
                }else{
                    children[5]->set_neighbor(neighbors[1], 1);
                    children[7]->set_neighbor(neighbors[1], 1);
                }
                // -y direction
                if(neighbors[2]!=NULL && !neighbors[2]->is_leaf()){
                    children[4]->set_neighbor(neighbors[2]->children[6], 2);
                    children[5]->set_neighbor(neighbors[2]->children[7], 2);
                }else{
                    children[4]->set_neighbor(neighbors[2], 2);
                    children[5]->set_neighbor(neighbors[2], 2);
                }
                // +y direction
                if(neighbors[3]!=NULL && !neighbors[3]->is_leaf()){
                    children[6]->set_neighbor(neighbors[3]->children[4], 3);
                    children[7]->set_neighbor(neighbors[3]->children[5], 3);
                }else{
                    children[6]->set_neighbor(neighbors[3], 3);
                    children[7]->set_neighbor(neighbors[3], 3);
                }
                // -z direction
                if(neighbors[4]!=NULL && !neighbors[4]->is_leaf()){
                    children[0]->set_neighbor(neighbors[4]->children[4], 4);
                    children[1]->set_neighbor(neighbors[4]->children[5], 4);
                    children[2]->set_neighbor(neighbors[4]->children[6], 4);
                    children[3]->set_neighbor(neighbors[4]->children[7], 4);
                }else{
                    children[0]->set_neighbor(neighbors[4], 4);
                    children[1]->set_neighbor(neighbors[4], 4);
                    children[2]->set_neighbor(neighbors[4], 4);
                    children[3]->set_neighbor(neighbors[4], 4);
                }
                // +z direction
                if(neighbors[5]!=NULL && !neighbors[5]->is_leaf()){
                    children[4]->set_neighbor(neighbors[5]->children[0], 5);
                    children[5]->set_neighbor(neighbors[5]->children[1], 5);
                    children[6]->set_neighbor(neighbors[5]->children[2], 5);
                    children[7]->set_neighbor(neighbors[5]->children[3], 5);
                }else{
                    children[4]->set_neighbor(neighbors[5], 5);
                    children[5]->set_neighbor(neighbors[5], 5);
                    children[6]->set_neighbor(neighbors[5], 5);
                    children[7]->set_neighbor(neighbors[5], 5);
                }
            }
        }
    }
    if(!force){
        if(!is_leaf()){
            for(int_t i = 0; i < (1<<n_dim); ++i){
                children[i]->divide(nodes, xs, ys, zs);
            }
        }
    }
};


void Cell::set_test_function(function func){
    test_func = func;
    if(!is_leaf()){
        for(int_t i = 0; i < (1<<n_dim); ++i){
            children[i]->set_test_function(func);
        }
    }
}

void Cell::build_cell_vector(cell_vec_t& cells){
    if(this->is_leaf()){
        cells.push_back(this);
        return;
    }
    for(int_t i = 0; i < (1<<n_dim); ++i){
        children[i]->build_cell_vector(cells);
    }
}

Cell* Cell::containing_cell(double x, double y, double z){
    if(is_leaf()){
      return this;
    }
    int ix = x > children[0]->points[3]->location[0];
    int iy = y > children[0]->points[3]->location[1];
    int iz = n_dim>2 && z>children[0]->points[7]->location[2];
    return children[ix + 2*iy + 4*iz]->containing_cell(x, y, z);
};

Cell::~Cell(){
        if(is_leaf()){
            return;
        }
        for(int_t i = 0; i< (1<<n_dim); ++i){
            delete children[i];
        }
};

Tree::Tree(){
    nx = 0;
    ny = 0;
    nz = 0;
    n_dim = 0;
    max_level = 0;
};

void Tree::set_dimension(int_t dim){
    n_dim = dim;
}

void Tree::set_levels(int_t l_x, int_t l_y, int_t l_z){
    int_t min_l = std::min(l_x, l_y);
    if(n_dim == 3) min_l = std::min(min_l, l_z);
    max_level = min_l;
    if(l_x != l_y || (n_dim==3 && l_y!=l_z)) ++max_level;

    nx = 2<<l_x;
    ny = 2<<l_y;
    nz = (n_dim == 3)? 2<<l_z : 0;

    nx_roots = 1<<(l_x-(max_level-1));
    ny_roots = 1<<(l_y-(max_level-1));
    nz_roots = (n_dim==3)? 1<<(l_z-(max_level-1)) : 1;

    if (l_x==l_y && (n_dim==2 || l_y==l_z)){
        --nx_roots;
        --ny_roots;
        --nz_roots;
    }
    ixs = new int_t[nx_roots+1];
    iys = new int_t[ny_roots+1];
    izs = new int_t[nz_roots+1];

    int_t min_n = 2<<min_l;
    for(int_t i=0; i<nx_roots+1; ++i){
        ixs[i] = min_n*i;
    }
    for(int_t i=0; i<ny_roots+1; ++i){
        iys[i] = min_n*i;
    }
    for(int_t i=0; i<nz_roots+1; ++i){
        izs[i] = min_n*i;
    }
    if(n_dim == 2) nz_roots = 1;

    // Initialize root cell container
    roots.resize(nz_roots);
    for(int_t iz=0; iz<nz_roots; ++iz){
        roots[iz].resize(ny_roots);
        for(int_t iy=0; iy<ny_roots; ++iy){
            roots[iz][iy].resize(nx_roots);
            for(int_t ix=0; ix<nx_roots; ++ix){
                roots[iz][iy][ix] = NULL;
            }
        }
    }
};

void Tree::set_xs(double *x, double *y, double *z){
    xs = x;
    ys = y;
    zs = z;
}

void Tree::initialize_roots(){
    if(roots[0][0][0]==NULL){
        //Create grid of root nodes
        std::vector<std::vector<std::vector<Node *> > > points;
        if(n_dim == 2){
            points.resize(1);
        }else{
            points.resize(nz_roots+1);
        }
        for(int_t iz = 0; iz<points.size(); ++iz){
            points[iz].resize(ny_roots+1);
            for(int_t iy = 0; iy<ny_roots+1; ++iy){
                points[iz][iy].resize(nx_roots+1);
                for(int_t ix = 0; ix<nx_roots+1; ++ix){
                    points[iz][iy][ix] = new Node(ixs[ix], iys[iy], izs[iz],
                                                  xs, ys, zs);
                    nodes[points[iz][iy][ix]->key] = points[iz][iy][ix];
                }
            }
        }

        // Create grid of root cells
        for (int_t iz = 0; iz<nz_roots; ++iz){
            for (int_t iy = 0; iy<ny_roots; ++iy){
                for (int_t ix = 0; ix<nx_roots; ++ix){
                    Node *ps[8];
                    ps[0] = points[iz][iy  ][ix  ];
                    ps[1] = points[iz][iy  ][ix+1];
                    ps[2] = points[iz][iy+1][ix  ];
                    ps[3] = points[iz][iy+1][ix+1];
                    if (n_dim == 3){
                        ps[4] = points[iz+1][iy  ][ix  ];
                        ps[5] = points[iz+1][iy  ][ix+1];
                        ps[6] = points[iz+1][iy+1][ix  ];
                        ps[7] = points[iz+1][iy+1][ix+1];
                    }
                    roots[iz][iy][ix] = new Cell(ps, n_dim, max_level, NULL);
                    if (nx==ny && (n_dim==2 || ny==nz)){
                        roots[iz][iy][ix]->level = 0;
                    }else{
                        roots[iz][iy][ix]->level = 1;
                    }
                    for(int_t i = 0; i < (1<<n_dim); ++i){
                        ps[i]->reference += 1;
                    }
                }
            }
        }
        // Set root cell neighbors
        // +x neighbors
        for(int_t iz=0; iz<nz_roots; ++iz)
            for (int_t iy=0; iy<ny_roots; ++iy)
                for(int_t ix=0; ix<nx_roots-1; ++ix)
                    roots[iz][iy][ix]->set_neighbor(roots[iz][iy][ix+1], 1);
        // +y neighbors
        for(int_t iz=0; iz<nz_roots; ++iz)
            for(int_t iy=0; iy<ny_roots-1; ++iy)
                for(int_t ix=0; ix<nx_roots; ++ix)
                    roots[iz][iy][ix]->set_neighbor(roots[iz][iy+1][ix], 3);
        // +z neighbors
        for(int_t iz=0; iz<nz_roots-1; ++iz)
            for(int_t iy=0; iy<ny_roots; ++iy)
                for(int_t ix=0; ix<nx_roots; ++ix)
                    roots[iz][iy][ix]->set_neighbor(roots[iz+1][iy][ix], 5);
    }
}

void Tree::insert_cell(double *new_center, int_t p_level){
    // find containing root
    int_t ix = 0;
    int_t iy = 0;
    int_t iz = 0;
    while (new_center[0]>=xs[ixs[ix+1]] && ix<nx_roots-1){
        ++ix;
    }
    while (new_center[1]>=ys[iys[iy+1]] && iy<ny_roots-1){
        ++iy;
    }
    if(n_dim == 3){
        while(new_center[2]>=zs[izs[iz+1]] && iz<nz_roots-1){
            ++iz;
        }
    }
    roots[iz][iy][ix]->insert_cell(nodes, new_center, p_level, xs, ys, zs);
}

void Tree::build_tree_from_function(function test_func){
    //Must set the test_func of all of the roots before I can start dividing
    for(int_t iz=0; iz<nz_roots; ++iz)
        for(int_t iy=0; iy<ny_roots; ++iy)
            for(int_t ix=0; ix<nx_roots; ++ix)
                roots[iz][iy][ix]->set_test_function(test_func);
    //Now we can divide
    for(int_t iz=0; iz<nz_roots; ++iz)
        for(int_t iy=0; iy<ny_roots; ++iy)
            for(int_t ix=0; ix<nx_roots; ++ix)
                roots[iz][iy][ix]->divide(nodes, xs, ys, zs);
};

void Tree::finalize_lists(){
    for(int_t iz=0; iz<nz_roots; ++iz)
        for(int_t iy=0; iy<ny_roots; ++iy)
            for(int_t ix=0; ix<nx_roots; ++ix)
                roots[iz][iy][ix]->build_cell_vector(cells);
    if(n_dim == 3){
        // Generate Faces and edges
        for(std::vector<Cell *>::size_type i = 0; i != cells.size(); i++){
            Cell *cell = cells[i];
            Node *p[8];
            for(int_t it = 0; it < 8; ++it)
                p[it] = cell->points[it];

            Edge *ex[4];
            Edge *ey[4];
            Edge *ez[4];

            ex[0] = set_default_edge(edges_x, *p[0], *p[1]);
            ex[1] = set_default_edge(edges_x, *p[2], *p[3]);
            ex[2] = set_default_edge(edges_x, *p[4], *p[5]);
            ex[3] = set_default_edge(edges_x, *p[6], *p[7]);

            ey[0] = set_default_edge(edges_y, *p[0], *p[2]);
            ey[1] = set_default_edge(edges_y, *p[1], *p[3]);
            ey[2] = set_default_edge(edges_y, *p[4], *p[6]);
            ey[3] = set_default_edge(edges_y, *p[5], *p[7]);

            ez[0] = set_default_edge(edges_z, *p[0], *p[4]);
            ez[1] = set_default_edge(edges_z, *p[1], *p[5]);
            ez[2] = set_default_edge(edges_z, *p[2], *p[6]);
            ez[3] = set_default_edge(edges_z, *p[3], *p[7]);

            Face *fx1, *fx2, *fy1, *fy2, *fz1, *fz2;
            fx1 = set_default_face(faces_x, *p[0], *p[2], *p[4], *p[6]);
            fx2 = set_default_face(faces_x, *p[1], *p[3], *p[5], *p[7]);
            fy1 = set_default_face(faces_y, *p[0], *p[1], *p[4], *p[5]);
            fy2 = set_default_face(faces_y, *p[2], *p[3], *p[6], *p[7]);
            fz1 = set_default_face(faces_z, *p[0], *p[1], *p[2], *p[3]);
            fz2 = set_default_face(faces_z, *p[4], *p[5], *p[6], *p[7]);

            fx1->edges[0] = ez[0];
            fx1->edges[1] = ey[2];
            fx1->edges[2] = ez[2];
            fx1->edges[3] = ey[0];

            fx2->edges[0] = ez[1];
            fx2->edges[1] = ey[3];
            fx2->edges[2] = ez[3];
            fx2->edges[3] = ey[1];

            fy1->edges[0] = ez[0];
            fy1->edges[1] = ex[2];
            fy1->edges[2] = ez[1];
            fy1->edges[3] = ex[0];

            fy2->edges[0] = ez[2];
            fy2->edges[1] = ex[3];
            fy2->edges[2] = ez[3];
            fy2->edges[3] = ex[1];

            fz1->edges[0] = ey[0];
            fz1->edges[1] = ex[1];
            fz1->edges[2] = ey[1];
            fz1->edges[3] = ex[0];

            fz2->edges[0] = ey[2];
            fz2->edges[1] = ex[3];
            fz2->edges[2] = ey[3];
            fz2->edges[3] = ex[2];

            cell->faces[0] = fx1;
            cell->faces[1] = fx2;
            cell->faces[2] = fy1;
            cell->faces[3] = fy2;
            cell->faces[4] = fz1;
            cell->faces[5] = fz2;

            for(int_t it = 0; it < 4; ++it){
                cell->edges[it    ] = ex[it];
                cell->edges[it + 4] = ey[it];
                cell->edges[it + 8] = ez[it];
            }

            for(int_t it = 0; it < 6; ++it)
                cell->faces[it]->reference++;
            for(int_t it = 0; it < 12; ++it)
                cell->edges[it]->reference++;

        }

        // Process hanging x faces
        for(face_it_type it = faces_x.begin(); it != faces_x.end(); ++it){
            Face *face = it->second;
            if(face->reference < 2){
                int_t x;
                x = face->location_ind[0];
                if(x==0 || x==nx) continue; // Face was on the outside, and is not hanging

                if(nodes.count(face->key)) continue; // I will have children (there is a node at my center)
                Node *node;

                //Find Parent
                int_t ip;
                for(int_t i = 0; i < 4; ++i){
                    node = face->points[i];
                    ip = i;
                    if(faces_x.count(node->key)){
                        face->parent = faces_x[node->key];
                        break;
                    }
                }

                //all of my edges are hanging, and most of my points
                for(int_t i = 0; i < 4; ++i){
                    face->edges[i]->hanging = true;
                    face->points[i]->hanging = true;
                }
                // the point oposite the parent node key should not be hanging
                // and also label the edges' parents
                if(face->points[ip^3]->reference != 6)
                    face->points[ip^3]->hanging = false;

                face->edges[0]->parents[0] = face->parent->edges[0];
                face->edges[0]->parents[1] = face->parent->edges[((ip&1)^1)<<1]; //2020

                face->edges[1]->parents[0] = face->parent->edges[1];
                face->edges[1]->parents[1] = face->parent->edges[ip>>1<<1^1]; //1133

                face->edges[2]->parents[0] = face->parent->edges[((ip&1)^1)<<1]; //2020
                face->edges[2]->parents[1] = face->parent->edges[2];

                face->edges[3]->parents[0] = face->parent->edges[ip>>1<<1^1]; //1133
                face->edges[3]->parents[1] = face->parent->edges[3];

                face->points[ip^1]->parents[0] = face->parent->points[(ip&1)^1]; //1010
                face->points[ip^1]->parents[1] = face->parent->points[(ip&1)^3]; //3232
                face->points[ip^1]->parents[2] = face->parent->points[(ip&1)^1]; //1010
                face->points[ip^1]->parents[3] = face->parent->points[(ip&1)^3]; //3232

                face->points[ip^2]->parents[0] = face->parent->points[(ip>>1^1)<<1]; //2200
                face->points[ip^2]->parents[1] = face->parent->points[(ip>>1^1)<<1^1]; //3311
                face->points[ip^2]->parents[2] = face->parent->points[(ip>>1^1)<<1]; //2200
                face->points[ip^2]->parents[3] = face->parent->points[(ip>>1^1)<<1^1]; //3311

                face->hanging = true;
                hanging_faces_x.push_back(face);
                for(int_t i = 0; i < 4; ++i)
                    node->parents[i] = face->parent->points[i];
            }
        }

        // Process hanging y faces
        for(face_it_type it = faces_y.begin(); it != faces_y.end(); ++it){
            Face *face = it->second;
            if(face->reference < 2){
                int_t y;
                y = face->location_ind[1];
                if(y==0 || y==ny) continue; // Face was on the outside, and is not hanging
                if(nodes.count(face->key)) continue; // I will have children (there is a node at my center)
                Node *node;

                //Find Parent
                int_t ip;
                for(int_t i = 0; i < 4; ++i){
                    node = face->points[i];
                    ip = i;
                    if(faces_y.count(node->key)){
                        face->parent = faces_y[node->key];
                        break;
                    }
                }
                //all of my edges are hanging, and most of my points
                for(int_t i = 0; i < 4; ++i){
                    face->edges[i]->hanging = true;
                    face->points[i]->hanging = true;
                }
                // the point oposite the parent node key should not be hanging
                // and also label the edges' parents
                if(face->points[ip^3]->reference != 6)
                    face->points[ip^3]->hanging = false;

                face->edges[0]->parents[0] = face->parent->edges[0];
                face->edges[0]->parents[1] = face->parent->edges[((ip&1)^1)<<1]; //2020

                face->edges[1]->parents[0] = face->parent->edges[1];
                face->edges[1]->parents[1] = face->parent->edges[ip>>1<<1^1]; //1133

                face->edges[2]->parents[0] = face->parent->edges[((ip&1)^1)<<1]; //2020
                face->edges[2]->parents[1] = face->parent->edges[2];

                face->edges[3]->parents[0] = face->parent->edges[ip>>1<<1^1]; //1133
                face->edges[3]->parents[1] = face->parent->edges[3];

                face->points[ip^1]->parents[0] = face->parent->points[(ip&1)^1]; //1010
                face->points[ip^1]->parents[1] = face->parent->points[(ip&1)^3]; //3232
                face->points[ip^1]->parents[2] = face->parent->points[(ip&1)^1]; //1010
                face->points[ip^1]->parents[3] = face->parent->points[(ip&1)^3]; //3232

                face->points[ip^2]->parents[0] = face->parent->points[(ip>>1^1)<<1]; //2200
                face->points[ip^2]->parents[1] = face->parent->points[(ip>>1^1)<<1^1]; //3311
                face->points[ip^2]->parents[2] = face->parent->points[(ip>>1^1)<<1]; //2200
                face->points[ip^2]->parents[3] = face->parent->points[(ip>>1^1)<<1^1]; //3311

                face->hanging = true;
                hanging_faces_y.push_back(face);
                for(int_t i = 0; i < 4; ++i){
                    node->parents[i] = face->parent->points[i];
                }
            }
        }

        // Process hanging z faces
        for(face_it_type it = faces_z.begin(); it != faces_z.end(); ++it){
            Face *face = it->second;
            if(face->reference < 2){
                int_t z;
                z = face->location_ind[2];
                if(z==0 || z==nz){
                    // Face was on the outside, and is not hanging
                    continue;
                }
                //check if I am a parent or a child
                if(nodes.count(face->key)){
                    // I will have children (there is a node at my center)
                    continue;
                }
                Node *node;

                //Find Parent
                int_t ip;
                for(int_t i = 0; i < 4; ++i){
                    node = face->points[i];
                    ip = i;
                    if(faces_z.count(node->key)){
                        face->parent = faces_z[node->key];
                        ip = i;
                        break;
                    }
                }
                //all of my edges are hanging, and most of my points
                for(int_t i = 0; i < 4; ++i){
                    face->edges[i]->hanging = true;
                    face->points[i]->hanging = true;
                }
                // the point oposite the parent node key should not be hanging
                // most of the time
                // and also label the edges' parents
                if(face->points[ip^3]->reference != 6)
                    face->points[ip^3]->hanging = false;

                face->edges[0]->parents[0] = face->parent->edges[0];
                face->edges[0]->parents[1] = face->parent->edges[((ip&1)^1)<<1]; //2020

                face->edges[1]->parents[0] = face->parent->edges[1];
                face->edges[1]->parents[1] = face->parent->edges[ip>>1<<1^1]; //1133

                face->edges[2]->parents[0] = face->parent->edges[((ip&1)^1)<<1]; //2020
                face->edges[2]->parents[1] = face->parent->edges[2];

                face->edges[3]->parents[0] = face->parent->edges[ip>>1<<1^1]; //1133
                face->edges[3]->parents[1] = face->parent->edges[3];

                face->points[ip^1]->parents[0] = face->parent->points[(ip&1)^1]; //1010
                face->points[ip^1]->parents[1] = face->parent->points[(ip&1)^3]; //3232
                face->points[ip^1]->parents[2] = face->parent->points[(ip&1)^1]; //1010
                face->points[ip^1]->parents[3] = face->parent->points[(ip&1)^3]; //3232

                face->points[ip^2]->parents[0] = face->parent->points[(ip>>1^1)<<1]; //2200
                face->points[ip^2]->parents[1] = face->parent->points[(ip>>1^1)<<1^1]; //3311
                face->points[ip^2]->parents[2] = face->parent->points[(ip>>1^1)<<1]; //2200
                face->points[ip^2]->parents[3] = face->parent->points[(ip>>1^1)<<1^1]; //3311

                face->hanging = true;
                hanging_faces_z.push_back(face);
                for(int_t i = 0; i < 4; ++i){
                    node->parents[i] = face->parent->points[i];
                }
            }
        }

    }
    else{
        //Generate Edges (and 1 face for consistency)
        for(std::vector<Cell *>::size_type i=0; i != cells.size(); i++){
            Cell *cell = cells[i];
            Node *p[4];
            for(int_t i = 0; i < 4; ++i)
                p[i] = cell->points[i];
            Edge *e[4];
            e[0] = set_default_edge(edges_x, *p[0], *p[1]);
            e[1] = set_default_edge(edges_x, *p[2], *p[3]);
            e[2] = set_default_edge(edges_y, *p[0], *p[2]);
            e[3] = set_default_edge(edges_y, *p[1], *p[3]);

            Face *face = set_default_face(faces_z, *p[0], *p[1], *p[2], *p[3]);
            for(int_t i = 0; i < 4; ++i){
                cell->edges[i] = e[i];
                face->edges[i] = e[i];
                e[i]->reference++;
            }

            face->hanging=false;
        }

        //Process hanging x edges
        for(edge_it_type it = edges_x.begin(); it != edges_x.end(); ++it){
            Edge *edge = it->second;
            if(edge->reference < 2){
                int_t y = edge->location_ind[1];
                if(y==0 || y==ny) continue; //I am on the boundary
                if(nodes.count(edge->key)) continue; //I am a parent
                //I am a hanging edge find my parent
                Node *node;
                if(edges_x.count(edge->points[0]->key)){
                    node = edge->points[0];
                }else{
                    node = edge->points[1];
                }
                edge->parents[0] = edges_x[node->key];
                edge->parents[1] = edge->parents[0];

                node->hanging = true;
                for(int_t i = 0; i<4; ++i)
                    node->parents[i] = edge->parents[0]->points[i%2];
                edge->hanging = true;
            }
        }

        //Process hanging y edges
        for(edge_it_type it = edges_y.begin(); it != edges_y.end(); ++it){
            Edge *edge = it->second;
            if(edge->reference < 2){
                int_t x = edge->location_ind[0];
                if(x==0 || x==nx) continue; //I am on the boundary
                if(nodes.count(edge->key)) continue; //I am a parent
                //I am a hanging edge find my parent
                Node *node;
                if(edges_y.count(edge->points[0]->key)){
                    node = edge->points[0];
                }else{
                    node = edge->points[1];
                }
                edge->parents[0] = edges_y[node->key];
                edge->parents[1] = edge->parents[0];

                node->hanging = true;
                for(int_t i = 0; i < 4; ++i)
                    node->parents[i] = edge->parents[0]->points[i%2];
                edge->hanging = true;
            }
        }
    }
    //List hanging edges x
    for(edge_it_type it = edges_x.begin(); it != edges_x.end(); ++it){
        Edge *edge = it->second;
        if(edge->hanging){
            hanging_edges_x.push_back(edge);
        }
    }
    //List hanging edges y
    for(edge_it_type it = edges_y.begin(); it != edges_y.end(); ++it){
        Edge *edge = it->second;
        if(edge->hanging){
            hanging_edges_y.push_back(edge);
        }
    }
    if(n_dim==3){
        //List hanging edges z
        for(edge_it_type it = edges_z.begin(); it != edges_z.end(); ++it){
            Edge *edge = it->second;
            if(edge->hanging){
                hanging_edges_z.push_back(edge);
            }
        }
    }

    //List hanging nodes
    for(node_it_type it = nodes.begin(); it != nodes.end(); ++it){
        Node *node = it->second;
        if(node->hanging){
            hanging_nodes.push_back(node);
        }
    }
}

void Tree::number(){
    //Number Nodes
    int_t ii, ih;
    ii = 0;
    ih = nodes.size() - hanging_nodes.size();
    for(node_it_type it = nodes.begin(); it != nodes.end(); ++it){
        Node *node = it->second;
        if(node->hanging){
            node->index = ih;
            ++ih;
        }else{
            node->index = ii;
            ++ii;
        }
    }

    //Number Cells
    for(std::vector<Cell *>::size_type i = 0; i != cells.size(); ++i)
        cells[i]->index = i;

    //Number edges_x
    ii = 0;
    ih = edges_x.size() - hanging_edges_x.size();
    for(edge_it_type it = edges_x.begin(); it != edges_x.end(); ++it){
        Edge *edge = it->second;
        if(edge->hanging){
          edge->index = ih;
          ++ih;
        }else{
          edge->index = ii;
          ++ii;
        }
    }
    //Number edges_y
    ii = 0;
    ih = edges_y.size() - hanging_edges_y.size();
    for(edge_it_type it = edges_y.begin(); it != edges_y.end(); ++it){
        Edge *edge = it->second;
        if(edge->hanging){
          edge->index = ih;
          ++ih;
        }else{
          edge->index = ii;
          ++ii;
        }
    }

    if(n_dim==3){
        //Number faces_x
        ii = 0;
        ih = faces_x.size() - hanging_faces_x.size();
        for(face_it_type it = faces_x.begin(); it != faces_x.end(); ++it){
            Face *face = it->second;
            if(face->hanging){
                face->index = ih;
                ++ih;
            }else{
                face->index = ii;
                ++ii;
            }
        }
        //Number faces_y
        ii = 0;
        ih = faces_y.size() - hanging_faces_y.size();
        for(face_it_type it = faces_y.begin(); it != faces_y.end(); ++it){
            Face *face = it->second;
            if(face->hanging){
                face->index = ih;
                ++ih;
            }else{
                face->index = ii;
                ++ii;
            }
        }

        //Number faces_z
        ii = 0;
        ih = faces_z.size() - hanging_faces_z.size();
        for(face_it_type it = faces_z.begin(); it != faces_z.end(); ++it){
            Face *face = it->second;
            if(face->hanging){
                face->index = ih;
                ++ih;
            }else{
                face->index = ii;
                ++ii;
            }
        }

        //Number edges_z
        ii = 0;
        ih = edges_z.size() - hanging_edges_z.size();
        for(edge_it_type it = edges_z.begin(); it != edges_z.end(); ++it){
            Edge *edge = it->second;
            if(edge->hanging){
              edge->index = ih;
              ++ih;
            }else{
              edge->index = ii;
              ++ii;
            }
        }
    }else{
        //Ensure Fz and cells are numbered the same in 2D
        for(std::vector<Cell *>::size_type i = 0; i != cells.size(); ++i)
            faces_z[cells[i]->key]->index = cells[i]->index;
    }

};

Tree::~Tree(){
    if (roots.size() == 0){
        return;
    }
    for(int_t iz=0; iz<nz_roots; ++iz){
        for(int_t iy=0; iy<ny_roots; ++iy){
            for(int_t ix=0; ix<nx_roots; ++ix){
                delete roots[iz][iy][ix];
            }
        }
    }
    delete[] ixs;
    delete[] iys;
    delete[] izs;
    for(node_it_type it = nodes.begin(); it != nodes.end(); ++it){
        delete it->second;
    }
    for(face_it_type it = faces_x.begin(); it != faces_x.end(); ++it){
        delete it->second;
    }
    for(face_it_type it = faces_y.begin(); it != faces_y.end(); ++it){
        delete it->second;
    }
    for(face_it_type it = faces_z.begin(); it != faces_z.end(); ++it){
        delete it->second;
    }
    for(edge_it_type it = edges_x.begin(); it != edges_x.end(); ++it){
        delete it->second;
    }
    for(edge_it_type it = edges_y.begin(); it != edges_y.end(); ++it){
        delete it->second;
    }
    for(edge_it_type it = edges_z.begin(); it != edges_z.end(); ++it){
        delete it->second;
    }
    roots.clear();
    cells.clear();
    nodes.clear();
    faces_x.clear();
    faces_y.clear();
    faces_z.clear();
    edges_x.clear();
    edges_y.clear();
    edges_z.clear();
};

Cell* Tree::containing_cell(double x, double y, double z){
    // find containing root
    int_t ix = 0;
    int_t iy = 0;
    int_t iz = 0;
    while (x>=xs[ixs[ix+1]] && ix<nx_roots-1){
        ++ix;
    }
    while (y>=ys[iys[iy+1]] && iy<ny_roots-1){
        ++iy;
    }
    if(n_dim == 3){
        while(z>=zs[izs[iz+1]] && iz<nz_roots-1){
            ++iz;
        }
    }
    return roots[iz][iy][ix]->containing_cell(x, y, z);
}

void Tree::shift_cell_centers(double *shift){
    for(int_t iz=0; iz<nz_roots; ++iz)
        for(int_t iy=0; iy<ny_roots; ++iy)
            for(int_t ix=0; ix<nx_roots; ++ix)
                roots[iz][iy][ix]->shift_centers(shift);
}
