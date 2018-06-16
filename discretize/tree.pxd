from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.map cimport map

cdef extern from "tree.h":
    ctypedef int int_t

    cdef cppclass Node:
        int_t location_ind[3]
        double location[3]
        int_t key
        int_t reference
        int_t index
        bool hanging
        Node *parents[4]
        Node()
        Node(int_t, int_t, int_t, double, double, double)
        int_t operator[](int_t)

    cdef cppclass Edge:
        int_t location_ind[3]
        double location[3]
        int_t key
        int_t reference
        int_t index
        double length
        bool hanging
        Node *points[2]
        Edge *parents[2]
        Edge()
        Edge(Node& p1, Node& p2)

    cdef cppclass Face:
        int_t location_ind[3]
        double location[3]
        int_t key
        int_t reference
        int_t index
        double area
        bool hanging
        Node *points[4]
        Edge *edges[4]
        Face *parent
        Face()
        Face(Node& p1, Node& p2, Node& p3, Node& p4)

    ctypedef map[int_t, Node *] node_map_t
    ctypedef map[int_t, Edge *] edge_map_t
    ctypedef map[int_t, Face *] face_map_t

    cdef cppclass Cell:
        int_t n_dim
        Cell *parent
        Cell *children[8]
        Cell *neighbors[6]
        Node *points[8]
        Edge *edges[12]
        Face *faces[6]
        int_t location_ind[3]
        double location[3]
        int_t index, key, level, max_level
        double volume
        inline bool is_leaf()

    cdef cppclass PyWrapper:
        PyWrapper()
        void set(void*, int_t(*)(void*, Cell*))

    cdef cppclass Tree:
        int_t n_dim
        int_t max_level, nx, ny, nz

        vector[Cell *] cells
        node_map_t nodes
        edge_map_t edges_x, edges_y, edges_z
        face_map_t faces_x, faces_y, faces_z
        vector[Node *] hanging_nodes
        vector[Edge *] hanging_edges_x, hanging_edges_y, hanging_edges_z
        vector[Face *] hanging_faces_x, hanging_faces_y, hanging_faces_z

        Tree()

        void set_dimension(int_t)
        void set_levels(int_t, int_t, int_t)
        void set_xs(double*, double*, double*)
        void build_tree_from_function(PyWrapper *)
        void number()
        void initialize_roots()
        void insert_cell(double *new_center, int_t p_level);
        void finalize_lists()
        Cell * containing_cell(double, double, double)
        void shift_cell_centers(double*)
