#ifndef __TREE_H
#define __TREE_H

#include <vector>
#include <map>
#include <iostream>
#include <algorithm>

typedef std::size_t int_t;

inline int_t key_func(int_t x, int_t y){
//Double Cantor pairing
    return ((x+y)*(x+y+1))/2+y;
}
inline int_t key_func(int_t x, int_t y, int_t z){
    return key_func(key_func(x, y), z);
}
class Node;
class Edge;
class Face;
class Cell;
class Tree;
class PyWrapper;
typedef PyWrapper* function;

typedef std::map<int_t, Node *> node_map_t;
typedef std::map<int_t, Edge *> edge_map_t;
typedef std::map<int_t, Face *> face_map_t;
typedef node_map_t::iterator node_it_type;
typedef edge_map_t::iterator edge_it_type;
typedef face_map_t::iterator face_it_type;
typedef std::vector<Cell *> cell_vec_t;

class PyWrapper{
  public:
    void *py_func;
    int_t (*eval)(void *, Cell*);

  PyWrapper(){
    py_func = NULL;
  };

  void set(void* func, int_t (*wrapper)(void*, Cell*)){
    py_func = func;
    eval = wrapper;
  };

  int operator()(Cell * cell){
    return eval(py_func, cell);
  };
};

class Node{
  public:
    int_t location_ind[3];
    double location[3];
    int_t key;
    int_t reference;
    int_t index;
    bool hanging;
    Node *parents[4];
    Node();
    Node(int_t, int_t, int_t, double*, double*, double*);
    double operator[](int_t index){
      return location[index];
    };
};

class Edge{
  public:
    int_t location_ind[3];
    double location[3];
    int_t key;
    int_t reference;
    int_t index;
    double length;
    bool hanging;
    Node *points[2];
    Edge *parents[2];
    Edge();
    Edge(Node& p1, Node&p2);
};

class Face{
    public:
        int_t location_ind[3];
        double location[3];
        int_t key;
        int_t reference;
        int_t index;
        double area;
        bool hanging;
        Node *points[4];
        Edge *edges[4];
        Face *parent;
        Face();
        Face(Node& p1, Node& p2, Node& p3, Node& p4);
};


class Cell{
  public:
    int_t n_dim;
    Cell *parent, *children[8], *neighbors[6];
    Node *points[8];
    Edge *edges[12];
    Face *faces[6];

    int_t location_ind[3], index, key, level, max_level;
    double location[3];
    double volume;
    function test_func;

    Cell();
    Cell(Node *pts[4], int_t ndim, int_t maxlevel, function func);
    Cell(Node *pts[4], Cell *parent);
    ~Cell();

    bool inline is_leaf(){ return children[0]==NULL;};
    void spawn(node_map_t& nodes, Cell *kids[8], double* xs, double *ys, double *zs);
    void divide(node_map_t& nodes, double* xs, double* ys, double* zs, bool force=false, bool balance=true);
    void set_neighbor(Cell* other, int_t direction);
    void set_test_function(function func);
    void build_cell_vector(cell_vec_t& cells);

    void insert_cell(node_map_t &nodes, double *new_center, int_t p_level, double* xs, double *ys, double *zs);

    Cell* containing_cell(double, double, double);
    void shift_centers(double * shift);
};

class Tree{
  public:
    int_t n_dim;
    std::vector<std::vector<std::vector<Cell *> > > roots;
    int_t max_level, nx, ny, nz;
    int_t *ixs, *iys, *izs;
    int_t nx_roots, ny_roots, nz_roots;
    double *xs;
    double *ys;
    double *zs;

    std::vector<Cell *> cells;
    node_map_t nodes;
    edge_map_t edges_x, edges_y, edges_z;
    face_map_t faces_x, faces_y, faces_z;
    std::vector<Node *> hanging_nodes;
    std::vector<Edge *> hanging_edges_x, hanging_edges_y, hanging_edges_z;
    std::vector<Face *> hanging_faces_x, hanging_faces_y, hanging_faces_z;

    Tree();
    ~Tree();

    void set_dimension(int_t dim);
    void set_levels(int_t l_x, int_t l_y, int_t l_z);
    void set_xs(double *x , double *y, double *z);
    void initialize_roots();
    void build_tree_from_function(function test_func);
    void number();
    void finalize_lists();

    void insert_cell(double *new_center, int_t p_level);

    Cell* containing_cell(double, double, double);

    void shift_cell_centers(double *shift);
};
#endif
