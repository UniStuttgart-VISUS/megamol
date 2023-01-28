/*
 * Types.h
 * Copyright (C) 2006-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MMMOLMAPPLG_TYPES_H_INCLUDED
#define MMMOLMAPPLG_TYPES_H_INCLUDED
#pragma once

#include "helper_includes/helper_cuda.h"
#include "helper_includes/helper_math.h"

#include "vislib/math/Matrix.h"
#include "vislib/math/Vector.h"

#include <map>
#include <vector>

// Typedefs used by MolecularMaps.
typedef vislib::math::Vector<float, 2> vec2f;
typedef vislib::math::Vector<float, 3> vec3f;
typedef vislib::math::Vector<float, 4> vec4f;
typedef vislib::math::Vector<double, 2> vec2d;
typedef vislib::math::Vector<double, 3> vec3d;
typedef vislib::math::Vector<double, 4> vec4d;
typedef vislib::math::Vector<uint, 3> vec3ui;
typedef vislib::math::Vector<int, 3> vec3i;
typedef vislib::math::Vector<uint, 4> vec4ui;
typedef vislib::math::Matrix<float, 3, vislib::math::MatrixLayout::COLUMN_MAJOR> mat3f;
typedef vislib::math::Matrix<float, 4, vislib::math::MatrixLayout::COLUMN_MAJOR> mat4f;
typedef std::pair<double, uint16_t> closestPair;

// Structs used by MolecularMaps.
/**
 * The definition of a cell.
 */

namespace megamol {
namespace molecularmaps {
struct Cell {
    /**
     * Initialises an empty instance.
     */
    Cell(void) {
        this->atom_ids.reserve(10);
    }

    /**
     * Initialises an empty instance.
     */
    Cell(const std::vector<uint>& p_atom_ids) : atom_ids(p_atom_ids) {}

    /** The atoms that are contained in the cell. */
    std::vector<uint> atom_ids;
};
} // namespace molecularmaps
} // namespace megamol

/**
 * The definition of an ray for CUDA.
 */
struct CudaRay {
    /**
     * Initialises an empty instance.
     */
    __host__ __device__ CudaRay(void)
            : dir(make_float3(0.0f, 0.0f, 0.0f))
            , origin(make_float3(0.0f, 0.0f, 0.0f))
            , inv_dir(make_float3(0.0f, 0.0f, 0.0f)) {}


    /**
     * Initialises an empty instance.
     */
    __host__ __device__ CudaRay(const float3& p_dir, const float3& p_origin) : origin(p_origin) {
        // Set the direction.
        this->dir = p_dir;
        this->dir *= -1.0f;
        auto len = sqrt(this->dir.x * this->dir.x + this->dir.y * this->dir.y + this->dir.z * this->dir.z);
        this->dir.x /= len;
        this->dir.y /= len;
        this->dir.z /= len;
        // Set the inverse direction.
        this->inv_dir.x = 1.0f / this->dir.x;
        this->inv_dir.y = 1.0f / this->dir.y;
        this->inv_dir.z = 1.0f / this->dir.z;
    }

    /** The direction of the ray. */
    float3 dir;

    /** The inverse direction of the ray. */
    float3 inv_dir;

    /** The origin of the ray. */
    float3 origin;
};

/**
 * The definition of a BoundingBox for CUDA.
 */
struct CudaBoundingBox {
    /**
     * Initialises an empty instance.
     */
    __host__ __device__ CudaBoundingBox(void) : max(make_float3(0.0f)), min(make_float3(0.0f)) {}

    /**
     * Initialises an empty instance.
     */
    __host__ __device__ CudaBoundingBox(const float3& p_max, const float3& p_min) : max(p_max), min(p_min) {}

    /**
     * Initialises an empty instance.
     */
    __host__ __device__ CudaBoundingBox(const float3* p_vertices, const uint p_vertices_cnt) {
        // Initialise the bounding box.
        max = make_float3(-FLT_MAX);
        min = make_float3(FLT_MAX);

        // Find the dimensions.
        for (uint i = 0; i < p_vertices_cnt; i++) {
            // Update the left, front, bottom point.
            if (p_vertices[i].x < min.x)
                min.x = p_vertices[i].x;
            if (p_vertices[i].y < min.y)
                min.y = p_vertices[i].y;
            if (p_vertices[i].z < min.z)
                min.z = p_vertices[i].z;

            // Update the right, back, top point.
            if (p_vertices[i].x > max.x)
                max.x = p_vertices[i].x;
            if (p_vertices[i].y > max.y)
                max.y = p_vertices[i].y;
            if (p_vertices[i].z > max.z)
                max.z = p_vertices[i].z;
        }
    }

    /**
     * Checks if the bounding box is contained in the given bounding box.
     *
     * @param p_bbox the bounding box that might contain this bounding box
     *
     * @return true if this bounding box is contained, false otherwise
     */
    __host__ __device__ bool IsContained(const CudaBoundingBox& p_bbox) {
        return (((p_bbox.min.x < this->min.x) && (p_bbox.min.y < this->min.y) && (p_bbox.min.z < this->min.z)) &&
                ((p_bbox.max.x > this->max.x) && (p_bbox.max.y > this->max.y) && (p_bbox.max.z > this->max.z)));
    }

    /**
     * Test if the given ray intersects the bounding box.
     *
     * @param p_ray the ray with origin and direction
     *
     * @return true if the ray intersects, false if it does not
     */
    __host__ __device__ bool RayIntersection(const CudaRay& p_ray) const {
        float t1 = (this->min.x - p_ray.origin.x) * p_ray.inv_dir.x;
        float t2 = (this->max.x - p_ray.origin.x) * p_ray.inv_dir.x;
        float t3 = (this->min.y - p_ray.origin.y) * p_ray.inv_dir.y;
        float t4 = (this->max.y - p_ray.origin.y) * p_ray.inv_dir.y;
        float t5 = (this->min.z - p_ray.origin.z) * p_ray.inv_dir.z;
        float t6 = (this->max.z - p_ray.origin.z) * p_ray.inv_dir.z;

        float tmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
        float tmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));

        return tmax > fmax(tmin, 0.0f);
    }

    /** The right, front, top point. */
    float3 max;

    /** The left, back, bottom point. */
    float3 min;
};

/**
 * The definition of an Octree node.
 */
struct CudaOctreeNode {
    /**
     * Initialises an empty instance.
     */
    __host__ __device__ CudaOctreeNode(void) : bounding_box(CudaBoundingBox()) {
        this->child_cnt = 0;
        this->face_cnt = 0;
    }

    /**
     * Initialises an empty instance.
     */
    __host__ __device__ CudaOctreeNode(
        const CudaBoundingBox& p_bounding_box, const uint p_child_cnt, const uint p_face_cnt)
            : bounding_box(p_bounding_box) {
        this->child_cnt = p_child_cnt;
        this->face_cnt = p_face_cnt;
    }

    /**
     * Destroys the instance.
     */
    __host__ __device__ ~CudaOctreeNode() {}

    /** The bounding box of the node. */
    CudaBoundingBox bounding_box;

    /** The number of child nodes. */
    uint child_cnt;

    /** The number of faces inside the bounding box. */
    uint face_cnt;
};

/**
 * The definition of a cut.
 */
struct Cut {
    /**
     * Initialises an empty instance.
     */
    Cut(void) : tunnel_id(0) {
        this->colours = std::vector<float>(0);
        this->faces = std::vector<uint>(0);
        this->normals = std::vector<float>(0);
        this->vertices = std::vector<float>(0);
    }

    /**
     * Initialises an empty instance.
     */
    Cut(const std::vector<float>& p_colours, const std::vector<uint>& p_faces, const std::vector<float>& p_normals,
        const uint p_tunnel_id, const std::vector<float>& p_vertices)
            : tunnel_id(p_tunnel_id) {
        this->colours = p_colours;
        this->faces = p_faces;
        this->normals = p_normals;
        this->vertices = p_vertices;
    }

    /** The colours of the cut vertices. */
    std::vector<float> colours;

    /** The faces of the cut. */
    std::vector<uint> faces;

    /** The normals of the cut vertices. */
    std::vector<float> normals;

    /** The ID of the tunnel the cut belongs to. */
    uint tunnel_id;

    /** The vertices of the cut. */
    std::vector<float> vertices;
};

/**
 * The definition of an ray.
 */
struct Ray {
    /**
     * Initialises an empty instance.
     */
    Ray(void) : dir(vec3f(0.0f, 0.0f, 0.0f)), origin(vec3f(0.0f, 0.0f, 0.0f)), inv_dir(vec3f(0.0f, 0.0f, 0.0f)) {}


    /**
     * Initialises an empty instance.
     */
    Ray(const vec3f& p_dir, const vec3f& p_origin) : origin(p_origin) {
        // Set the direction.
        this->dir = p_dir;
        this->dir *= -1.0f;
        this->dir.Normalise();
        // Set the inverse direction.
        this->inv_dir.SetX(1.0f / this->dir.GetX());
        this->inv_dir.SetY(1.0f / this->dir.GetY());
        this->inv_dir.SetZ(1.0f / this->dir.GetZ());
    }

    /** The direction of the ray. */
    vec3f dir;

    /** The inverse direction of the ray. */
    vec3f inv_dir;

    /** The origin of the ray. */
    vec3f origin;
};

/**
 * The definition of a BoundingBox.
 */
struct BoundingBox {
    /**
     * Initialises an empty instance.
     */
    BoundingBox(void) : max(make_float3(0.0f)), min(make_float3(0.0f)) {}

    /**
     * Initialises an empty instance.
     */
    BoundingBox(const float3& p_max, const float3& p_min) : max(p_max), min(p_min) {}

    /**
     * Initialises an empty instance.
     */
    BoundingBox(const std::vector<float3>& p_vertices) {
        // Initialise the bounding box.
        max = make_float3(-std::numeric_limits<float>::max());
        min = make_float3(std::numeric_limits<float>::max());

        // Find the dimensions.
        for (const auto& vertex : p_vertices) {
            // Update the left, front, bottom point.
            if (vertex.x < min.x)
                min.x = vertex.x;
            if (vertex.y < min.y)
                min.y = vertex.y;
            if (vertex.z < min.z)
                min.z = vertex.z;

            // Update the right, back, top point.
            if (vertex.x > max.x)
                max.x = vertex.x;
            if (vertex.y > max.y)
                max.y = vertex.y;
            if (vertex.z > max.z)
                max.z = vertex.z;
        }
    }

    /**
     * Checks if the bounding box is contained in the given bounding box.
     *
     * @param p_bbox the bounding box that might contain this bounding box
     *
     * @return true if this bounding box is contained, false otherwise
     */
    bool IsContained(const BoundingBox& p_bbox) {
        return (((p_bbox.min.x < this->min.x) && (p_bbox.min.y < this->min.y) && (p_bbox.min.z < this->min.z)) &&
                ((p_bbox.max.x > this->max.x) && (p_bbox.max.y > this->max.y) && (p_bbox.max.z > this->max.z)));
    }

    /**
     * Test if the given ray intersects the bounding box.
     *
     * @param p_ray the ray with origin and direction
     *
     * @return true if the ray intersects, false if it does not
     */
    bool RayIntersection(const Ray& p_ray) const {
        float t1 = (this->min.x - p_ray.origin.GetX()) * p_ray.inv_dir.GetX();
        float t2 = (this->max.x - p_ray.origin.GetX()) * p_ray.inv_dir.GetX();
        float t3 = (this->min.y - p_ray.origin.GetY()) * p_ray.inv_dir.GetY();
        float t4 = (this->max.y - p_ray.origin.GetY()) * p_ray.inv_dir.GetY();
        float t5 = (this->min.z - p_ray.origin.GetZ()) * p_ray.inv_dir.GetZ();
        float t6 = (this->max.z - p_ray.origin.GetZ()) * p_ray.inv_dir.GetZ();

        float tmin = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
        float tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));

        return tmax > std::max(tmin, 0.0f);
    }

    /**
     * Test if the given sphere intersects the bounding box.
     * Based on the paper :
     * Larsson et al.: On Faster Sphere-Box Overlap Testing
     * http://www.idt.mdh.se/personal/tla/publ/sb.pdf
     *
     * @param p_sphere The sphere with position and radius
     *
     * @return True if the sphere intersects, false if it does not.
     */
    bool SphereIntersection(const vec4d& p_sphere) const {
        double x = std::max(this->min.x, std::min(static_cast<float>(p_sphere.X()), this->max.x));
        double y = std::max(this->min.y, std::min(static_cast<float>(p_sphere.Y()), this->max.y));
        double z = std::max(this->min.z, std::min(static_cast<float>(p_sphere.Z()), this->max.z));

        double distsquared = (x - p_sphere.X()) * (x - p_sphere.X()) + (y - p_sphere.Y()) * (y - p_sphere.Y()) +
                             (z - p_sphere.Z()) * (z - p_sphere.Z());
        return (distsquared < (p_sphere.W() * p_sphere.W()));
    }

    /** The right, front, top point. */
    float3 max;

    /** The left, back, bottom point. */
    float3 min;
};

/**
 * The definition of an Edge.
 */
struct Edge {
    /**
     * Initialises an empty instance.
     */
    __host__ __device__ Edge(void)
            : edge_id(0)
            , face_id_0(-1)
            , face_id_1(-1)
            , opposite_edge_id(-1)
            , vertex_id_0(0)
            , vertex_id_1(0) {}

    /**
     * Initialises an empty instance.
     */
    Edge(const uint p_edge_id, const int p_face_id_0, const int p_face_id_1, const int p_opposite_edge_id,
        const uint p_vertex_id_0, const uint p_vertex_id_1)
            : edge_id(p_edge_id)
            , face_id_0(p_face_id_0)
            , face_id_1(p_face_id_1)
            , opposite_edge_id(p_opposite_edge_id)
            , vertex_id_0(p_vertex_id_0)
            , vertex_id_1(p_vertex_id_1) {}

    /**
     * Compare two Edges and return true if they are equal.
     *
     * @param rhs The edge to compare with.
     *
     * @return true if the edges are equal, false otherwise.
     */
    inline bool operator==(const Edge& rhs) {
        return ((this->face_id_0 == rhs.face_id_0) && (this->face_id_1 == rhs.face_id_1) &&
                (this->vertex_id_0 == rhs.vertex_id_0) && (this->vertex_id_1 == rhs.vertex_id_1));
    }

    /**
     * Compare two Edges and return true if they are not equal.
     *
     * @param rhs The edge to compare with.
     *
     * @return true if the edges are not equal, false otherwise.
     */
    inline bool operator!=(const Edge& rhs) {
        return ((this->face_id_0 != rhs.face_id_0) || (this->face_id_1 != rhs.face_id_1) ||
                (this->vertex_id_0 != rhs.vertex_id_0) || (this->vertex_id_1 != rhs.vertex_id_1));
    }

    /** Store the index of the edge. */
    uint edge_id;

    /** Store the first face of the edge. */
    int face_id_0;

    /** Store the second face of the edge. */
    int face_id_1;

    /** Store the index of the opposite edge. */
    int opposite_edge_id;

    /** Store the first vertex of the edge. */
    uint vertex_id_0;

    /** Store the second vertex of the edge. */
    uint vertex_id_1;
};

/**
 * The collection of parameters for the GetEndVertex function.
 */
struct EndVertexParams {
    /**
     * Initialises an empty instance.
     */
    EndVertexParams(const std::pair<vec4d, std::array<uint, 5>>& p_gate, const std::array<vec4d, 2>& p_gate_center,
        std::array<vec4d, 4>& p_gate_vector, const vec3d& p_pivot)
            : gate(p_gate)
            , gate_center(p_gate_center)
            , gate_vector(p_gate_vector)
            , pivot(p_pivot) {}

    /** The current gate that is processed. */
    const std::pair<vec4d, std::array<uint, 5>>& gate;

    /** Contains all two gate centers, only the first one is used. */
    const std::array<vec4d, 2>& gate_center;

    /** Contains all three gate spheres. */
    std::array<vec4d, 4>& gate_vector;

    /**
     * The pivot point that lies on the symmetry axis of the
     * curve between the start vertex and the potential end vertex.
     */
    const vec3d& pivot;
};

/**
 * The definition of a face group.
 */
struct FaceGroup {
    /**
     * Initialises an empty instance.
     */
    FaceGroup(void) : state(false) {
        this->border_edges = std::vector<Edge>(0);
        this->circles = std::vector<std::vector<uint>>(0);
        this->circles.reserve(4);
        this->valid_circles = std::vector<bool>(0);
        this->valid_circles.reserve(4);
    }

    /**
     * Initialises an empty instance.
     */
    FaceGroup(const uint p_face, const bool p_state) : state(p_state) {
        this->border_edges = std::vector<Edge>(0);
        this->circles = std::vector<std::vector<uint>>(0);
        this->circles.reserve(4);
        this->valid_circles = std::vector<bool>(0);
        this->valid_circles.reserve(4);
    }

    /**
     * Add the face to the group.
     *
     * @param p_face the ID of the face
     */
    void AddFace(const uint p_face) {
        this->face_map.insert(std::make_pair(p_face, p_face));
    }

    /**
     * Remove the face from the group.
     *
     * @param p_face the ID of the face
     */
    void RemoveFace(const uint p_face) {
        this->face_map.erase(p_face);
    }

    /** The IDs of edges that border another group of a differnet state. */
    std::vector<Edge> border_edges;

    /** The circles that are found for the group, consists of the vertex IDs that form the circle. */
    std::vector<std::vector<uint>> circles;

    /** The map of faces that belongs to the group. */
    std::map<uint, uint> face_map;

    /** The shadowed state of all faces in the group. */
    bool state;

    /** The circles that are valid. */
    std::vector<bool> valid_circles;
};

/**
 * The definition of an Octree node.
 */
struct OctreeNode {
    /**
     * Initialises an empty instance.
     */
    OctreeNode(void) : bounding_box(BoundingBox()), cuda_idx(0) {
        this->children = std::vector<OctreeNode>(0);
        this->children.reserve(8);
        this->children.shrink_to_fit();
        this->faces = std::vector<uint>(0);
        this->faces.shrink_to_fit();
    }

    /**
     * Initialises an empty instance.
     */
    OctreeNode(const BoundingBox& p_bounding_box, const uint p_cuda_idx, const std::vector<uint>& p_faces)
            : bounding_box(p_bounding_box)
            , cuda_idx(p_cuda_idx) {
        this->children = std::vector<OctreeNode>(0);
        this->children.reserve(8);
        this->children.shrink_to_fit();
        this->faces = std::vector<uint>(0);
        this->faces.reserve(p_faces.size());
        for (const auto face : p_faces) {
            this->faces.push_back(face);
        }
        this->faces.shrink_to_fit();
    }

    /**
     * Initialises an empty instance.
     */
    OctreeNode(const BoundingBox& p_bounding_box, const uint p_cuda_idx, const size_t p_face_cnt)
            : bounding_box(p_bounding_box)
            , cuda_idx(p_cuda_idx) {
        this->children = std::vector<OctreeNode>(0);
        this->children.reserve(8);
        this->children.shrink_to_fit();
        this->faces = std::vector<uint>(0);
        this->faces.reserve(p_face_cnt);
        for (size_t i = 0; i < p_face_cnt; i++) {
            this->faces.push_back(static_cast<uint>(i));
        }
        this->faces.shrink_to_fit();
    }

    /**
     * Destroys the instance.
     */
    ~OctreeNode() {
        this->children.erase(this->children.begin(), this->children.end());
        this->children.clear();
        this->faces.erase(this->faces.begin(), this->faces.end());
        this->faces.clear();
    }

    /**
     * Remove all faces that no longer belong to the node. The nodes are marked true
     * in the given vector.
     *
     * @param p_to_delete a vector of the same length as the faces that belong to
     * the node, is true for every face that will be deleted from the node
     */
    void RemoveFaces(const std::vector<bool>& p_to_delete) {
        size_t last = 0;

        // Loop over the faces and skip the faces that no longer belong to the node.
        for (size_t i = 0; i < this->faces.size(); ++i, ++last) {
            while (p_to_delete[i]) {
                ++i;
            }
            if (i >= this->faces.size())
                break;

            this->faces[last] = this->faces[i];
        }
        this->faces.resize(last);
        this->faces.shrink_to_fit();
    }

    /** The bounding box of the node. */
    BoundingBox bounding_box;

    /** The child nodes. */
    std::vector<OctreeNode> children;

    /** The index of the node in the CUDA vector. */
    uint cuda_idx;

    /** The faces that are inside the bounding box. */
    std::vector<uint> faces;
};

/**
 * The definition of the poles of a sphere.
 */
struct Poles {
    /**
     * Initialises an empty instance.
     */
    Poles(void) : north(0), south(0) {}

    /**
     * Initialises an empty instance.
     */
    Poles(const uint p_north, const uint p_south) : north(p_north), south(p_south) {}

    /** The index of the vertex that represents the north pole. */
    uint north;

    /** The index of the vertex that represents the south pole. */
    uint south;
};

/**
 * The definition of the closest atoms pair for the radix sort.
 */
struct RadixTraitsPair {
    static const int nBytes = 16;
    int kth_byte(const closestPair& x, int k) {
        return static_cast<size_t>(x.first * 100000000000000.0) >> (k * 8) & 0xFF;
    }
    bool compare(const closestPair& x, const closestPair& y) {
        return x.first < y.first;
    }
};

/**
 * The definition of a Voronoi edge.
 */
struct VoronoiEdge {
    /**
     * Initialises an empty instance.
     */
    VoronoiEdge(void) : end_vertex(0), gate_sphere(vec4d()), start_vertex(0) {}

    /**
     * Initialises an empty instance.
     */
    VoronoiEdge(const uint p_end_vertex, const vec4d& p_gate_sphere, const uint p_start_vertex)
            : end_vertex(p_end_vertex)
            , gate_sphere(p_gate_sphere)
            , start_vertex(p_start_vertex) {}

    /**
     * Overloading of the smaller operator for fast sorting
     *
     * @param rhs The other VoronoiEdge we compare with.
     *
     * @return True, if this object is "smaller" than rhs, false otherwise.
     */
    bool operator<(const VoronoiEdge& rhs) const {
        return this->start_vertex < rhs.start_vertex;
    }

    /** The end vertex of the edge. */
    uint end_vertex;

    /** The gate sphere that defines the edge. */
    vec4d gate_sphere;

    /** The start vertex of the edge. */
    uint start_vertex;
};

/**
 * The definition of a Voronoi vertex.
 */
struct VoronoiVertex {
    /**
     * Compute the hash value from the atom IDs of the vertex. The hash function is from:
     * http://stackoverflow.com/questions/1536393/good-hash-function-for-permutations
     *
     * @param p_atom_ids the IDs of the atoms for which the hash is created
     *
     * @return the hash value
     */
    static inline uint64_t ComputeHash(const vec4ui& p_atom_ids) {
        uint64_t r = 11;
        uint64_t hash =
            static_cast<uint64_t>(r + 2 * p_atom_ids.GetX()) * static_cast<uint64_t>(r + 2 * p_atom_ids.GetY()) *
            static_cast<uint64_t>(r + 2 * p_atom_ids.GetZ()) * static_cast<uint64_t>(r + 2 * p_atom_ids.GetW());
        return hash / 2;
    }

    /**
     * Compare two Voronoi vertices and return true if they are equal.
     *
     * @param rhs The VoronoiVertex to compare with.
     *
     * @return true if the vertices are equal, false otherwise.
     */
    inline bool operator==(const VoronoiVertex& rhs) {
        // Compare the hash values of both vertices.
        return this->vertex_hash == rhs.vertex_hash;
    }

    /**
     * Initialises an empty instance.
     */
    VoronoiVertex(void) : atoms(vec4ui()), id(0), infinity_count(0), vertex(vec4d()), vertex_hash(0) {}

    /**
     * Initialises an empty instance.
     */
    VoronoiVertex(
        const vec4ui& p_atoms, const uint p_id, const uint p_infinity_count = 0, const vec4d& p_vertex = vec4d()) {
        // Set the atoms to the given list of gate atom IDs.
        this->atoms = p_atoms;

        // Set the ID of the Voronoi vertex.
        this->id = p_id;

        // Set infinity count of the voronoi vertex.
        this->infinity_count = p_infinity_count;

        // Set the vertex position and radius.
        this->vertex = p_vertex;

        // Compute the hash value from the atom IDs of the vertex.
        this->vertex_hash = VoronoiVertex::ComputeHash(this->atoms);
    }

    /**
     * The copy contructor.
     */
    VoronoiVertex(const VoronoiVertex& rhs) {
        this->atoms = rhs.atoms;
        this->id = rhs.id;
        this->infinity_count = rhs.infinity_count;
        this->vertex = rhs.vertex;
        this->vertex_hash = rhs.vertex_hash;
    }

    /** The atoms that define the Voronoi vertex. */
    vec4ui atoms;

    /** The ID of the Voronoi vertex. */
    uint id;

    /** Number of gates without new voronoi vertex */
    uint infinity_count;

    /** The vertex parameters with position and radius */
    vec4d vertex;

    /** The hash value that is created from the atom IDs. */
    uint64_t vertex_hash;
};

// Enums used by MolecularMaps.
/**
 * The different display modes.
 */
enum DisplayMode {
    PROTEIN = 0,
    SHADOW = PROTEIN + 1,
    GROUPS = SHADOW + 1,
    CUTS = GROUPS + 1,
    VORONOI = CUTS + 1,
    REBUILD = VORONOI + 1,
    TUNNEL = REBUILD + 1,
    SPHERE = TUNNEL + 1,
    MAP = SPHERE + 1
};

/**
 * The different geodesic lines modes.
 */
enum GeodesicMode { NO_LINES = 0, ONE_TO_ALL = NO_LINES + 1, ALL_TO_ALL = ONE_TO_ALL + 1 };

/**
 * The modes for the mesh output
 */
enum MeshMode {
    MESH_ORIGINAL = 0,
    MESH_CUT = MESH_ORIGINAL + 1,
    MESH_SPHERE = MESH_CUT + 1,
    MESH_MAP = MESH_SPHERE + 1
};

#endif /* MMMOLMAPPLG_TYPES_H_INCLUDED */
