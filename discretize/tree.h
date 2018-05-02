#ifndef __TREE_H
#define __TREE_H

#include <vector>
#include <map>
#include <iostream>

typedef std::size_t int_t;

inline int_t key_func(int_t x, int_t y){
//Double Cantor pairing
    return ((x+y)*(x+y+1))/2+y;
}
inline int_t key_func(int_t x, int_t y, int_t z){
    return key_func(key_func(x,y), z);
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
    void *py_obj;
    void *py_func;
    int_t (*eval)(void*, void *, Cell*);
  PyWrapper(){
    py_func = NULL;
  };

  void set(void* obj, void* func, int_t (*wrapper)(void*, void*, Cell*)){
    py_obj = obj;
    py_func = func;
    eval = wrapper;
  };

  int operator()(Cell * cell){
    return eval(py_obj, py_func, cell);
  };

};

class Node{
  public:
    int_t location[3];
    int_t key;
    int_t reference;
    int_t index;
    bool hanging;
    Node *parents[4];
    Node();
    Node(int_t, int_t, int_t);
    int_t operator[](int_t index){
      return location[index];
    };
};

class Edge{
  public:
    int_t location[3];
    int_t key;
    int_t reference;
    int_t index;
    int_t length;
    bool hanging;
    Node *points[2];
    Edge *parents[2];
    Edge();
    Edge(Node& p1, Node&p2);
};

class Face{
    public:
        int_t location[3];
        int_t key;
        int_t reference;
        int_t index;
        int_t area;
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

    int_t center[3], index, key, level, max_level;
    int_t volume;
    function test_func;

    Cell();
    Cell(Node *pts[4], int_t ndim, int_t maxlevel, function func);
    Cell(Node *pts[4], Cell *parent);
    ~Cell();

    bool inline is_leaf(){ return children[0]==NULL;};
    void spawn(node_map_t& nodes, Cell *kids[8]);
    void divide(node_map_t& nodes, bool force=false, bool balance=true);
    void set_neighbor(Cell* other, int_t direction);
    void build_cell_vector(cell_vec_t& cells);

    void insert_cell(node_map_t &nodes, int_t *new_center, int_t p_level);

    Cell* containing_cell(double, double, double);
};

class Tree{
  public:
    int_t n_dim;
    Cell *root;
    function test_func;
    int_t max_level, nx, ny, nz;

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
    void set_level(int_t max_level);
    void build_tree_from_function(function test_func);
    void number();
    void finalize_lists();

    void insert_cell(int_t *new_center, int_t p_level);

    Cell* containing_cell(double, double, double);
};
#endif
