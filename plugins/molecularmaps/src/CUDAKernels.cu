/*
 * CUDAKernels.cu
 * Copyright (C) 2009-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "CUDAKernels.cuh"

using namespace megamol;
using namespace megamol::molecularmaps;

__constant__ float C_PI = 3.141592653589f;

/**
 * CUDA symbol for the face grouping will be false if no
 * further changes where made to the face IDs.
 */
__device__ bool changes_d;

/**
 * Smaller operator for edges.
 */
__host__ __device__ bool operator<(const Edge& lhs, const Edge& rhs) {
    return lhs.vertex_id_1 < rhs.vertex_id_1;
}

/**
 * Sort edges ascending to the face_id_0.
 */
struct FaceID0Cmp {
    __host__ __device__ bool operator()(const Edge& lhs, const Edge& rhs) {
        return lhs.face_id_0 < rhs.face_id_0;
    }
};

/**
 * Sort edges ascending to the face_id_1.
 */
struct FaceID1Cmp {
    __host__ __device__ bool operator()(const Edge& lhs, const Edge& rhs) {
        return lhs.face_id_1 < rhs.face_id_1;
    }
};

/**
 * Sort edges ascending to the vertex_id_0.
 */
struct VertexID0Cmp {
    __host__ __device__ bool operator()(const Edge& lhs, const Edge& rhs) {
        return lhs.vertex_id_0 < rhs.vertex_id_0;
    }
};

/**
 * Sort edges ascending to the vertex_id_1.
 */
struct VertexID1Cmp {
    __host__ __device__ bool operator()(const Edge& lhs, const Edge& rhs) {
        return lhs.vertex_id_1 < rhs.vertex_id_1;
    }
};


/**
 * Get the thread index based on the current CUDE grid dimensions.
 *
 * @return Returns the thread index based on the current CUDA grid
 *         dimensions.
 */
__device__ uint GetThreadIndex() {
    return __umul24(__umul24(blockIdx.y, gridDim.x) + blockIdx.x, blockDim.x) + threadIdx.x;
}


/**
 * Check if the ray intersects the triangle based on the vertices.
 *
 * @param p_ray the ray with origin an direction
 * @param p_vertex_0 the first vertex of the triangle
 * @param p_vertex_1 the second vertex of the triangle
 * @param p_vertex_2 the third vertex of the triangle
 *
 * @return true if the ray intersects, false otherwise
 */
__device__ bool RayTriangleIntersection(
    const CudaRay p_ray, const float3 p_vertex_0, const float3 p_vertex_1, const float3 p_vertex_2) {
    float3 e2 = p_vertex_2 - p_vertex_0;
    float3 e1 = p_vertex_1 - p_vertex_0;
    float3 r = cross(p_ray.dir, e2);
    float3 s = p_ray.origin - p_vertex_0;
    float denom = dot(e1, r);
    if (abs(denom) < 1e-5)
        return false;
    float f = 1.0f / denom;
    float3 q = cross(s, e1);
    float u = dot(s, r);

    if (denom > 1e-5) {
        if (u < 0.0f || u > denom)
            return false;
        float v = dot(p_ray.dir, q);
        if (v < 0.0f || (u + v) > denom)
            return false;

    } else {
        if (u > 0.0f || u < denom)
            return false;
        float v = dot(p_ray.dir, q);
        if (v > 0.0f || (u + v) < denom)
            return false;
    }

    float t = f * dot(e2, q);
    if (t > 1e-5)
        return true;
    else
        return false;
}


/**
 * Find the face that is hit by the given ray. Traverse the Octree to
 * find the closest intersection.
 *
 * @param p_octree_nodes the Octree of the surface in CUDA representation
 * @parma p_node_faces the faces that belong to every node in the Octree
 * @param p_node_faces_offset the array that gives the start and end index
 * for the faces of a node in the Octree
 * @param p_faces the faces of the surface
 * @param p_rays the directions for the intersection tests with the faces
 * of the surface
 * @param p_vertices the vertices of the surface
 * @param p_origins the positions of the voronoi vertices
 * @param p_queues the queues for the Octree traversal
 * @param p_intersections will contain the ID of the intersected face in
 * each round or -1 if no face was intersected in the round
 * @param p_ao_values will contain the ao values for every voronoi vertex
 * @param p_max_nodes the number of nodes in the Octree
 * @param p_round the current round
 * @param p_max_rounds the overall number of rounds
 * @param p_vertex_cnt the number of voronoi vertices
 *
 * @return the ID of the face that was hit or -1 if no intersection was
 * found
 */
__global__ void ComputeOctreeIntersection(const CudaOctreeNode* p_octree_nodes, const uint* p_node_faces,
    const uint* p_node_faces_offset, const uint3* p_faces, const float3* p_rays, const float3* p_vertices,
    const float3* p_origins, uint* p_queues, int* p_intersections, float* p_ao_values, uint p_max_nodes, uint p_round,
    uint p_max_rounds, uint p_vertex_cnt) {
    // Get the thread index and return if it is too big.
    const uint idx = GetThreadIndex();
    if (idx >= p_vertex_cnt)
        return;

    // Get current ray and the current voronoi vertex position.
    float3 ray_dir = p_rays[p_round];
    float3 origin = p_origins[idx];
    CudaRay ray = CudaRay(ray_dir, origin);

    // Go into the Octree from the root downwards and look for faces that are intersected
    // by the ray. Pick the closest one as the intersection.
    uint zero = idx * p_max_nodes;
    uint queue_idx = zero;
    float min_dist = FLT_MAX;
    int face_id = -1;

    // Add the root to the queue.
    p_queues[queue_idx++] = 0;

    // Get the current Octree node from the queue and test for intersections.
    while (queue_idx != zero) {
        uint curr = p_queues[--queue_idx];

        // Check all faces that belong to the current node.
        uint begin = p_node_faces_offset[curr];
        uint end = p_node_faces_offset[curr + 1];
        for (uint i = begin; i < end; i++) {
            uint3 face_idxs = p_faces[p_node_faces[i]];
            float3 vertex_0 = p_vertices[face_idxs.x];
            float3 vertex_1 = p_vertices[face_idxs.y];
            float3 vertex_2 = p_vertices[face_idxs.z];
            if (RayTriangleIntersection(ray, vertex_0, vertex_1, vertex_2)) {
                // Found intersection check if the distance is smaller than the closest intersection
                // so far.
                float x = vertex_0.x - ray.origin.x;
                float y = vertex_0.y - ray.origin.y;
                float z = vertex_0.z - ray.origin.z;
                float dist = x * x + y * y + z * z;
                if (dist < min_dist) {
                    min_dist = dist;
                    face_id = p_node_faces[i];
                }
            }
        }

        // Add all children of the current node to the queue if the ray intersects their bounding box.
        for (size_t i = 0; i < p_octree_nodes[curr].child_cnt; i++) {
            uint child_idx = (curr * 8) + (i + 1);
            if (p_octree_nodes[child_idx].bounding_box.RayIntersection(ray)) {
                p_queues[queue_idx++] = child_idx;
            }
        }
    }

    // Set the intersected face or -1 in the p_intersections array.
    if (face_id != -1) {
        p_ao_values[idx]++;
    }
    p_intersections[idx * p_max_rounds + p_round] = face_id;
}

/**
 * Get the IDs of each neighbour for a every vertex.
 *
 * @param p_neighbour_ids Will contain the IDs of neighbouring vertices
 * @param p_valid_z_values Remebers if a vertex is valid.
 * @param p_vertex_edge_offset The edges that contain the vertex.
 * @param p_vertex_edge_offset_depth The number of edges per vertex.
 * @param p_vertex_cnt The number of vertices in the mesh.
 */
__global__ void GetNeighbourIds(uint* p_neighbour_ids, bool* p_valid_z_values, const Edge* p_vertex_edge_offset,
    const uint* p_vertex_edge_offset_depth, uint p_vertex_cnt) {
    const uint idx = GetThreadIndex();
    if (idx >= p_vertex_cnt)
        return;
    if (!p_valid_z_values[idx])
        return;

    uint begin = p_vertex_edge_offset_depth[idx];
    uint end = p_vertex_edge_offset_depth[idx + 1];
    uint neighbour_id;
    // Find the IDs of neighbouring vertices.
    for (uint i = begin; i < end; i++) {
        Edge cur = p_vertex_edge_offset[i];
        if (cur.vertex_id_0 == idx)
            neighbour_id = cur.vertex_id_1;
        else
            neighbour_id = cur.vertex_id_0;
        p_neighbour_ids[i] = neighbour_id;
    }
}

/**
 * Set the ID of each face to the minimum of the neighbouring faces.
 *
 * @param p_face_group_ids the current IDs of all faces
 * @param p_face_shadowed the states of all faces
 * @param p_face_edge_offset the edges that belong to a face
 * @param p_face_edge_offset_depth the number of edges per face in the offset
 * @param p_face_cnt the number of faces
 */
__global__ void GroupFacesKernel(uint* p_face_group_ids, const bool* p_face_shadowed, const Edge* p_face_edge_offset,
    const uint* p_face_edge_offset_depth, const uint p_face_cnt) {
    const uint idx = GetThreadIndex();
    if (idx >= p_face_cnt)
        return;

    bool state = p_face_shadowed[idx];
    uint begin = p_face_edge_offset_depth[idx];
    uint end = p_face_edge_offset_depth[idx + 1];
    uint min = p_face_group_ids[idx];
    for (uint i = begin; i < end; i++) {
        Edge cur = p_face_edge_offset[i];
        if (cur.face_id_0 == idx) {
            if (p_face_group_ids[cur.face_id_1] < min && p_face_shadowed[cur.face_id_1] == state) {
                min = p_face_group_ids[cur.face_id_1];
            }

        } else {
            if (p_face_group_ids[cur.face_id_0] < min && p_face_shadowed[cur.face_id_0] == state) {
                min = p_face_group_ids[cur.face_id_0];
            }
        }
    }

    if (min < p_face_group_ids[idx]) {
        p_face_group_ids[idx] = min;
        changes_d = true;
    }
}

/**
 * Create the edges of the mesh based on the faces. For each
 * face three edges are created. Each edge will exist twice
 * facing opposit directions.
 *
 * @param p_faces The list of faces in the mesh. Each face
 * consists of three vertex IDs.
 * @param p_edge Will contain the edges of the mesh.
 * @param p_face_cnt The number of faces in the mesh.
 */
__global__ void InitEdges(const uint3* p_faces, Edge* p_edges, uint p_face_cnt) {
    const uint idx = GetThreadIndex();
    if (idx >= p_face_cnt)
        return;

    // First edge.
    {
        Edge cur = p_edges[idx * 3];
        cur.face_id_0 = idx;
        cur.face_id_1 = -1;
        cur.vertex_id_0 = p_faces[idx].x;
        cur.vertex_id_1 = p_faces[idx].y;
        cur.edge_id = idx * 3;
        p_edges[idx * 3] = cur;
    }

    // Second edge.
    {
        Edge cur = p_edges[idx * 3 + 1];
        cur.face_id_0 = idx;
        cur.face_id_1 = -1;
        cur.vertex_id_0 = p_faces[idx].y;
        cur.vertex_id_1 = p_faces[idx].z;
        cur.edge_id = idx * 3 + 1;
        p_edges[idx * 3 + 1] = cur;
    }

    // Third edge.
    {
        Edge cur = p_edges[idx * 3 + 2];
        cur.face_id_0 = idx;
        cur.face_id_1 = -1;
        cur.vertex_id_0 = p_faces[idx].z;
        cur.vertex_id_1 = p_faces[idx].x;
        cur.edge_id = idx * 3 + 2;
        p_edges[idx * 3 + 2] = cur;
    }
}

/**
 * Match the edges so that each edge has a full set of faces
 * to which it belongs. Therefore search the sorted edges
 * for the same edge that faces the opposit direction. The
 * sorted edges are sorted ascending to the vertex_id_1.
 *
 * @param p_edges The list of edges of the mesh.
 * @param p_sorted_edges The sorted edges of the mesh.
 * @param p_edge_offset The offset that points to the index
 * in the p_sorted_edges for which vertex_id_1 == vertex_id_0
 * of the current edge.
 * @param p_edge_cnt The number of edges in the mesh (x2).
 */
__global__ void MatchEdges(Edge* p_edges, Edge* p_sorted_edges, uint* p_edge_offset, uint p_edge_cnt) {
    const uint idx = GetThreadIndex();
    if (idx >= p_edge_cnt)
        return;

    // Get current edge and check if it is already matched.
    Edge cur = p_edges[idx];
    if (cur.face_id_1 != -1)
        return;

    // Find the same edge faceing in the opposit direction.
    uint begin = p_edge_offset[cur.vertex_id_0];
    uint end = p_edge_offset[cur.vertex_id_0 + 1];
    for (uint i = begin; i < end; i++) {
        uint id = p_sorted_edges[i].edge_id;

        if (i == idx)
            continue;
        if (p_edges[id].face_id_1 != -1)
            continue;

        if (cur.vertex_id_0 == p_edges[id].vertex_id_1 && cur.vertex_id_1 == p_edges[id].vertex_id_0) {
            // Found the edge.
            cur.face_id_1 = p_edges[id].face_id_0;
            cur.opposite_edge_id = id;
            p_edges[id].face_id_1 = cur.face_id_0;
            p_edges[id].opposite_edge_id = cur.edge_id;
            p_edges[idx] = cur;
            break;
        }
    }
}

/**
 * Find all edges that belong to a certain face by looping over the edges
 * of the mesh.
 *
 * @param p_face_edge_offset Will contain the edges that belong to a face.
 * @param p_face_id_0_offset Contains the edges sorted ascending for face_id_0
 * @param p_face_id_1_offset Contains the edges sorted ascending for face_id_1
 * @param depth The maximum number of edges per face.
 * @param p_face_cnt The number of faces in the mesh.
 */
__global__ void SetFaceEdgeOffset(
    Edge* p_face_edge_offset, Edge* p_face_id_0_offset, Edge* p_face_id_1_offset, uint depth, uint p_face_cnt) {
    const uint idx = GetThreadIndex();
    if (idx >= p_face_cnt)
        return;

    // Find edges that belong to the face.
    uint cur_depth = idx * depth;
    uint begin = idx * 3;
    uint end = (idx + 1) * 3;
    for (uint i = begin; i < end; i++) {
        p_face_edge_offset[cur_depth++] = p_face_id_0_offset[i];
        p_face_edge_offset[cur_depth++] = p_face_id_1_offset[i];
    }
}

/**
 * Find all edges that belong to a certain vertex by looping over the edges
 * of the mesh.
 *
 * @param p_vertex_edge_offset Will contain the edges that belong to a vertex.
 * @param p_vertex_id_0_sorted Contains the edges sorted ascending for vertex_id_0
 * @param p_vertex_id_1_sorted Contains the edges sorted ascending for vertex_id_1
 * @param p_vertex_id_0_offset Points to the first edge in p_vertex_id_0_sorted
 * with the vertex_id_0 == idx
 * @param p_vertex_id_1_offset Points to the first edge in p_vertex_id_1_sorted
 * with the vertex_id_1 == idx
 * @param depth The maximum number of edges per vertex.
 * @param p_vertex_cnt The number of vertices in the mesh.
 */
__global__ void SetVertexEdgeOffset(Edge* p_vertex_edge_offset, Edge* p_vertex_id_0_sorted, Edge* p_vertex_id_1_sorted,
    uint* p_vertex_id_0_offset, uint* p_vertex_id_1_offset, uint* depth, uint p_vertex_cnt) {
    const uint idx = GetThreadIndex();
    if (idx >= p_vertex_cnt)
        return;

    // Find edges that belong to the vertex.
    uint cur_depth = depth[idx];

    // Edges with vertex_id_0 == idx
    uint begin = p_vertex_id_0_offset[idx];
    uint end = p_vertex_id_0_offset[idx + 1];
    for (uint i = begin; i < end; i++) {
        p_vertex_edge_offset[cur_depth++] = p_vertex_id_0_sorted[i];
    }
    // Edges with vertex_id_1 == idx
    begin = p_vertex_id_1_offset[idx];
    end = p_vertex_id_1_offset[idx + 1];
    for (uint i = begin; i < end; i++) {
        p_vertex_edge_offset[cur_depth++] = p_vertex_id_1_sorted[i];
    }
}

/**
 * Compute the average phi value in the neighbourhood and assign it to
 * the current vertex. Take care of special cases arround the boundary
 * meridian by assignig the vertices on the meridian the phi value of
 * 0 if they are approached from the "right" and a value of 2*pi if they
 * are approached from the "left".
 *
 * @param p_phivalues_in The phi values of the last iteration (input)
 * @param p_phivalues_out The phi value of the current iteration (output)
 * @param p_valid_phi_values Remebers if a vertex is valid.
 * @param p_vertex_neighbours The IDs of the neighbouring vertices.
 * @param p_vertex_edge_offset_depth The number of edges per vertex.
 * @param p_vertex_type The type of the vertex: -1: Pole, 0: vertex
 * is not on the meridian or a neighbour, 1: vertex is on the meridian,
 * 2: vertex is on the "right" side of the meridian and 3: vertex is
 * on the "left" side of the meridian.
 * @param p_vertex_neighbours_offset Contains how many neighbours are of
 * type -1.
 * @param p_vertex_cnt The number of vertices in the mesh.
 */
__global__ void SetPhiValues(float* p_phivalues_in, float* p_phivalues_out, bool* p_valid_phi_values,
    const uint* p_vertex_neighbours, const uint* p_vertex_edge_offset_depth, const int* p_vertex_type,
    const uint* p_vertex_neighbours_offset, uint p_vertex_cnt) {
    const uint idx = GetThreadIndex();
    if (idx >= p_vertex_cnt)
        return;
    if (!p_valid_phi_values[idx])
        return;

    uint begin = p_vertex_edge_offset_depth[idx] + p_vertex_neighbours_offset[idx];
    uint end = p_vertex_edge_offset_depth[idx + 1];
    float count = end - begin;
    float tmp = 0.0f;
    // Add up the phivalues of the neighbouring vertices and increase the counter.
    if (p_vertex_type[idx] == 0 || p_vertex_type[idx] == 2) {
        for (uint i = begin; i < end; i++) {
            tmp += p_phivalues_in[p_vertex_neighbours[i]];
        }

    } else {
        /**
         * Since vertices with the types -1 and 1 are remembered in the
         * p_valid_phi_vlaues we can be shure that the vertices here are
         * of type 3.
         */
        for (uint i = begin; i < end; i++) {
            if (p_vertex_type[p_vertex_neighbours[i]] != 1) {
                tmp += p_phivalues_in[p_vertex_neighbours[i]];

            } else {
                tmp += 2.0f * C_PI;
            }
        }
    }

    float tmp_phi = (tmp / count) - p_phivalues_in[idx];
    p_phivalues_out[idx] = p_phivalues_in[idx] + tmp_phi * 1.025f;
}

/**
 * Compute the theta value for each vertex by computing the inverse Mercator
 * projection based on the z value for each vertex.
 *
 * @param p_theta_values The theta value of each vertex
 * @param p_valid_theta_values Remebers if a vertex is valid.
 * @param p_zvalues The z value of each vertex.
 * @param p_theta One step on the angle between the poles.
 * @param p_theta_const Theta constant.
 * @param p_vertex_cnt The number of vertices in the mesh.
 */
__global__ void SetTheataValues(float* p_theta_values, bool* p_valid_theta_values, const float* p_zvalues,
    const float p_theta, const float p_theta_const, uint p_vertex_cnt) {
    const uint idx = GetThreadIndex();
    if (idx >= p_vertex_cnt)
        return;
    if (!p_valid_theta_values[idx])
        return;

    float tup = C_PI - p_theta / 2.0f;
    float tlow = p_theta / 2.0f;
    float dtheta = tup - tlow;
    float theta, tval;
    while (dtheta > (C_PI / 200000.0f)) {
        theta = (tup + tlow) / 2.0f;
        tval = p_theta_const * log(sin(theta) / (1.0f + cos(theta)));
        if (p_zvalues[idx] > tval)
            tlow = theta;
        else
            tup = theta;
        dtheta = tup - tlow;
    }
    p_theta_values[idx] = theta;
}

/**
 * Compute the average z value in the neighbourhood and assign it to the
 * current vertex.
 *
 * @param p_zvalues The z value of each vertex.
 * @param p_valid_z_values Remebers if a vertex is valid.
 * @param p_vertex_neighbours The IDs of the neighbouring vertices.
 * @param p_vertex_edge_offset_depth The number of edges per vertex.
 * @param p_vertex_cnt The number of vertices in the mesh.
 */
__global__ void SetZValues(float* p_zvalues, bool* p_valid_z_values, const uint* p_vertex_neighbours,
    const uint* p_vertex_edge_offset_depth, uint p_vertex_cnt) {
    const uint idx = GetThreadIndex();
    if (idx >= p_vertex_cnt)
        return;
    if (!p_valid_z_values[idx])
        return;

    uint begin = p_vertex_edge_offset_depth[idx];
    uint end = p_vertex_edge_offset_depth[idx + 1];
    float count = end - begin;
    float tmp = 0.0f;
    // Add up the zvalues of the neighbouring vertices and increase the counter.
    for (uint i = begin; i < end; i++) {
        tmp += p_zvalues[p_vertex_neighbours[i]];
    }
    float tmp_z = (tmp / count) - p_zvalues[idx];
    p_zvalues[idx] = p_zvalues[idx] + tmp_z * 1.025f;
}

/**
 * Sort the neighbour IDs ascending to the types of the neighbours.
 *
 * @param p_neighbour_ids The IDs of neighbouring vertices, will be sorted.
 * @param p_neighbour_ids_offset Will contain how many neighbours have an ID of -1.
 * @param p_valid_z_values Remebers if a vertex is valid.
 * @param p_vertex_edge_offset_depth The number of edges per vertex.
 * @param p_vertex_type The type of the vertex: -1: Pole, 0: vertex
 * is not on the meridian or a neighbour, 1: vertex is on the meridian,
 * 2: vertex is on the "right" side of the meridian and 3: vertex is
 * on the "left" side of the meridian.
 * @param p_vertex_cnt The number of vertices in the mesh.
 */
__global__ void SortNeighbourIds(uint* p_neighbour_ids, uint* p_neighbour_ids_offset, bool* p_valid_z_values,
    const uint* p_vertex_edge_offset_depth, const int* p_vertex_type, uint p_vertex_cnt) {
    const uint idx = GetThreadIndex();
    if (idx >= p_vertex_cnt)
        return;
    if (!p_valid_z_values[idx])
        return;

    uint begin = p_vertex_edge_offset_depth[idx];
    uint end = p_vertex_edge_offset_depth[idx + 1];
    uint offset = p_neighbour_ids_offset[idx];
    int type;
    uint buffer;
    // Sort the IDs according to the type.
    for (uint i = begin; i < end; i++) {
        type = p_vertex_type[p_neighbour_ids[i]];
        if (type == -1) {
            buffer = p_neighbour_ids[begin + offset];
            p_neighbour_ids[begin + offset] = p_neighbour_ids[i];
            p_neighbour_ids[i] = buffer;
            offset++;
        }
    }
    p_neighbour_ids_offset[idx] = offset;
}


/*
 * CUDAKernels::~CUDAKernels
 */
CUDAKernels::~CUDAKernels(void) {}


/*
 * CUDAKernels::ComputeVoronoiAO
 */
bool CUDAKernels::ComputeVoronoiAO(const std::vector<CudaOctreeNode>& p_cuda_octree_nodes,
    const std::vector<std::vector<uint>>& p_cuda_node_faces, const std::vector<uint>& p_faces,
    const std::vector<float3>& p_rays, const std::vector<float>& p_vertices,
    const std::vector<VoronoiVertex>& p_voronoi_vertices, const uint p_octree_nodes,
    std::vector<bool>& p_voronoi_tunnel, std::vector<std::vector<uint>>& p_voro_faces,
    std::vector<std::pair<size_t, VoronoiVertex>>& p_potential_vertices) {
    // Create the CUDA representation of the node faces.
    std::vector<uint> cuda_node_faces;
    std::vector<uint> cuda_node_faces_offset;
    cuda_node_faces.reserve(p_faces.size());
    cuda_node_faces_offset.push_back(0);
    for (size_t i = 0; i < p_cuda_node_faces.size(); i++) {
        cuda_node_faces.insert(cuda_node_faces.end(), p_cuda_node_faces[i].begin(), p_cuda_node_faces[i].end());
        cuda_node_faces_offset.push_back(cuda_node_faces_offset[i] + static_cast<uint>(p_cuda_node_faces[i].size()));
    }

    // Convert the faces to a CUDA format
    std::vector<uint3> cuda_faces = std::vector<uint3>(p_faces.size() / 3);
    for (uint i = 0; i < cuda_faces.size(); i++) {
        cuda_faces[i].x = p_faces[i * 3 + 0];
        cuda_faces[i].y = p_faces[i * 3 + 1];
        cuda_faces[i].z = p_faces[i * 3 + 2];
    }

    // Convert the vertices to a CUDA format.
    std::vector<float3> cuda_vertices = std::vector<float3>(p_vertices.size() / 3);
    for (uint i = 0; i < cuda_vertices.size(); i++) {
        cuda_vertices[i].x = p_vertices[i * 3 + 0];
        cuda_vertices[i].y = p_vertices[i * 3 + 1];
        cuda_vertices[i].z = p_vertices[i * 3 + 2];
    }

    // Convert the voronoi vertex position to a CUDA format.
    std::vector<float3> cuda_voronoi_vertices = std::vector<float3>(p_voronoi_vertices.size());
    for (size_t i = 0; i < p_voronoi_vertices.size(); i++) {
        cuda_voronoi_vertices[i] = make_float3(p_voronoi_vertices[i].vertex.GetX(), p_voronoi_vertices[i].vertex.GetY(),
            p_voronoi_vertices[i].vertex.GetZ());
    }

    // Convert the 2D voronoi faces to a 1D representation.
    std::vector<int> cuda_voronoi_faces = std::vector<int>(p_rays.size() * p_voronoi_vertices.size(), -1);

    // Create the AO values for every voronoi vertex.
    std::vector<float> ao_vals = std::vector<float>(p_voronoi_vertices.size(), 0.0f);

    // Create the queues for the kernel.
    uint queue_size = p_octree_nodes / 10;
    std::vector<uint> queues = std::vector<uint>(p_voronoi_vertices.size() * queue_size);

    // Upload the data.
    thrust::device_vector<CudaOctreeNode> cuda_octree_nodes_d = p_cuda_octree_nodes;
    thrust::device_vector<uint> cuda_node_faces_d = cuda_node_faces;
    thrust::device_vector<uint> cuda_node_faces_offset_d = cuda_node_faces_offset;
    thrust::device_vector<uint3> cuda_faces_d = cuda_faces;
    thrust::device_vector<float3> cuda_rays_d = p_rays;
    thrust::device_vector<float3> cuda_vertices_d = cuda_vertices;
    thrust::device_vector<float3> cuda_voronoi_vertices_d = cuda_voronoi_vertices;
    thrust::device_vector<uint> queues_d = queues;
    thrust::device_vector<int> cuda_voronoi_faces_d = cuda_voronoi_faces;
    thrust::device_vector<float> ao_vals_d = ao_vals;
    uint vertex_cnt = static_cast<uint>(p_voronoi_vertices.size());
    uint max_rounds = static_cast<uint>(p_rays.size());

    // Compute the intersection for every ray.
    for (uint i = 0; i < max_rounds; i++) {
        ComputeOctreeIntersection<<<Grid(vertex_cnt, 256), 256>>>(
            thrust::raw_pointer_cast(cuda_octree_nodes_d.data().get()),
            thrust::raw_pointer_cast(cuda_node_faces_d.data().get()),
            thrust::raw_pointer_cast(cuda_node_faces_offset_d.data().get()),
            thrust::raw_pointer_cast(cuda_faces_d.data().get()), thrust::raw_pointer_cast(cuda_rays_d.data().get()),
            thrust::raw_pointer_cast(cuda_vertices_d.data().get()),
            thrust::raw_pointer_cast(cuda_voronoi_vertices_d.data().get()),
            thrust::raw_pointer_cast(queues_d.data().get()),
            thrust::raw_pointer_cast(cuda_voronoi_faces_d.data().get()),
            thrust::raw_pointer_cast(ao_vals_d.data().get()), queue_size, i, max_rounds, vertex_cnt);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // Download the data.
    thrust::copy(ao_vals_d.begin(), ao_vals_d.end(), ao_vals.begin());
    thrust::copy(cuda_voronoi_faces_d.begin(), cuda_voronoi_faces_d.end(), cuda_voronoi_faces.begin());

    // Find the voronoi vertices that are potentially inside of a tunnel.
    for (size_t i = 0; i < ao_vals.size(); i++) {
        // Check if the voronoi vertex is on the surface.
        ao_vals[i] /= static_cast<float>(p_rays.size());
        if (ao_vals[i] > 0.9f) {
            // The AO value is higher than the threshold so remember the voronoi vertex and set the visited flag
            // of the ID to false.
            p_potential_vertices.push_back(std::make_pair(i, p_voronoi_vertices[i]));
            p_voronoi_tunnel[i] = true;
        }

        // Copy the intersected faces into the output 2D vector.
        p_voro_faces[i].reserve(p_rays.size());
        for (size_t j = 0; j < p_rays.size(); j++) {
            int face_id = cuda_voronoi_faces[i * p_rays.size() + j];
            if (face_id != -1) {
                p_voro_faces[i].emplace_back(face_id);
            }
        }
    }

    return true;
}


/*
 * CUDAKernels::CreatemeshTopology
 */
bool CUDAKernels::CreateMeshTopology(const std::vector<uint>& p_faces,
    std::vector<std::vector<Edge>>& p_vertex_edge_offset, std::vector<std::vector<Edge>>& p_face_edge_offset,
    std::vector<uint>& p_vertex_edge_offset_depth, std::vector<uint>& p_face_edge_offset_depth) {
    std::vector<uint3> cuda_faces;
    std::vector<Edge> edges;
    std::vector<Edge> sorted_edges;
    std::vector<uint> sorted_edges_offset;
    std::vector<Edge> sorted_edges_vertex_id_0;
    std::vector<Edge> sorted_edges_vertex_id_1;
    std::vector<uint> sorted_edges_offset_v_0;
    std::vector<uint> sorted_edges_offset_v_1;

    // Initialise counts and depths of offset vectors.
    uint edge_cnt = static_cast<uint>(p_faces.size());
    uint face_cnt = static_cast<uint>(p_faces.size() / 3);
    uint face_offset_depth = static_cast<uint>(p_face_edge_offset[0].size());
    uint vertex_cnt = static_cast<uint>(p_vertex_edge_offset.size());
    uint vertex_offset_depth = static_cast<uint>(p_vertex_edge_offset[0].size());

    // Initialise edges
    edges = std::vector<Edge>(edge_cnt);

    // Convert the faces so that CUDA understands them.
    cuda_faces = std::vector<uint3>(face_cnt);
    for (uint i = 0; i < face_cnt; i++) {
        cuda_faces[i].x = p_faces[i * 3 + 0];
        cuda_faces[i].y = p_faces[i * 3 + 1];
        cuda_faces[i].z = p_faces[i * 3 + 2];
    }

    // Upload to CUDA memory.
    thrust::device_vector<uint3> faces_d = cuda_faces;
    thrust::device_vector<Edge> edges_d = edges;

    // Create edges.
    InitEdges<<<Grid(face_cnt, 256), 256>>>(
        thrust::raw_pointer_cast(faces_d.data().get()), thrust::raw_pointer_cast(edges_d.data().get()), face_cnt);
    checkCudaErrors(cudaDeviceSynchronize());
    thrust::copy(edges_d.begin(), edges_d.end(), edges.begin());
    sorted_edges = edges;

    // Delete faces and the device vector of faces.
    cuda_faces.clear();
    cuda_faces.shrink_to_fit();
    faces_d.clear();
    faces_d.shrink_to_fit();

    // Sort edges ascending to the vertex_id_1.
    thrust::device_vector<Edge> sorted_edges_d = sorted_edges;
    thrust::sort(sorted_edges_d.begin(), sorted_edges_d.end());
    thrust::copy(sorted_edges_d.begin(), sorted_edges_d.end(), sorted_edges.begin());

    // Create edge -> sorted_edges offset
    sorted_edges_offset.reserve(edge_cnt);
    uint cur = sorted_edges[0].vertex_id_1;
    sorted_edges_offset.push_back(0);
    for (uint i = 0; i < edge_cnt; i++) {
        if (sorted_edges[i].vertex_id_1 != cur) {
            auto diff = sorted_edges[i].vertex_id_1 - cur;
            if (diff != 1) {
                for (uint j = 0; j < diff - 1; j++) {
                    sorted_edges_offset.push_back(i);
                }
            }
            sorted_edges_offset.push_back(i);
            cur = sorted_edges[i].vertex_id_1;
        }
    }
    sorted_edges_offset.push_back(edge_cnt);

    // Upload to CUDA memory and delete local copy.
    thrust::device_vector<uint> sorted_edges_offset_d = sorted_edges_offset;
    sorted_edges.clear();
    sorted_edges.shrink_to_fit();

    // Match edges.
    MatchEdges<<<Grid(edge_cnt, 256), 256>>>(thrust::raw_pointer_cast(edges_d.data().get()),
        thrust::raw_pointer_cast(sorted_edges_d.data().get()),
        thrust::raw_pointer_cast(sorted_edges_offset_d.data().get()), edge_cnt);
    checkCudaErrors(cudaDeviceSynchronize());
    thrust::copy(edges_d.begin(), edges_d.end(), edges.begin());

    // Delete device vector of edges.
    edges_d.clear();
    edges_d.shrink_to_fit();

    // Delete the device vector of sorted edges as well as the sorted edges.
    sorted_edges_d.clear();
    sorted_edges_d.shrink_to_fit();

    // Create face edge offset.
    std::vector<Edge> cuda_face_offset;
    cuda_face_offset.resize(face_cnt * face_offset_depth, Edge());
    thrust::device_vector<Edge> cuda_face_offset_d = cuda_face_offset;

    // Sort the edges ascending to face_id_0.
    thrust::device_vector<Edge> sorted_edges_face_id_0_d = edges;
    thrust::sort(sorted_edges_face_id_0_d.begin(), sorted_edges_face_id_0_d.end(), FaceID0Cmp());

    // Sort the edges ascending to face_id_1.
    thrust::device_vector<Edge> sorted_edges_face_id_1_d = edges;
    thrust::sort(sorted_edges_face_id_1_d.begin(), sorted_edges_face_id_1_d.end(), FaceID1Cmp());

    // Set the face edge offset.
    SetFaceEdgeOffset<<<Grid(face_cnt, 256), 256>>>(thrust::raw_pointer_cast(cuda_face_offset_d.data().get()),
        thrust::raw_pointer_cast(sorted_edges_face_id_0_d.data().get()),
        thrust::raw_pointer_cast(sorted_edges_face_id_1_d.data().get()), face_offset_depth, face_cnt);
    checkCudaErrors(cudaDeviceSynchronize());
    thrust::copy(cuda_face_offset_d.begin(), cuda_face_offset_d.end(), cuda_face_offset.begin());

    // Reseize the offsets for each face.
    std::vector<uint> cuda_face_offset_depth = std::vector<uint>(face_cnt + 1, 0);
    for (size_t i = 0; i < cuda_face_offset_depth.size(); i++) {
        cuda_face_offset_depth[i] = static_cast<uint>(i) * face_offset_depth;
    }
    resizeOffsetFaces(cuda_face_offset, p_face_edge_offset, p_face_edge_offset_depth, cuda_face_offset_depth);

    // Delete the cuda face offset vector and the corresponding device vector.
    cuda_face_offset_d.clear();
    cuda_face_offset_d.shrink_to_fit();
    cuda_face_offset.clear();
    cuda_face_offset.shrink_to_fit();

    // Delete the sorted edges on device and host.
    sorted_edges_face_id_0_d.clear();
    sorted_edges_face_id_0_d.shrink_to_fit();
    sorted_edges_face_id_1_d.clear();
    sorted_edges_face_id_1_d.shrink_to_fit();

    // Create vertex edge offset.
    std::vector<Edge> cuda_vertex_offset;
    std::vector<uint> cuda_vertex_offset_depth;
    cuda_vertex_offset.reserve(vertex_cnt * vertex_offset_depth);
    cuda_vertex_offset_depth.reserve(vertex_cnt + 1);
    uint sum = 0;
    for (size_t i = 0; i < p_vertex_edge_offset.size(); i++) {
        cuda_vertex_offset_depth.push_back(sum);
        sum += static_cast<uint>(p_vertex_edge_offset[i].size());
        for (size_t j = 0; j < p_vertex_edge_offset[i].size(); j++) {
            cuda_vertex_offset.push_back(Edge());
        }
    }
    cuda_vertex_offset_depth.push_back(sum);

    // Upload to CUDA memory.
    thrust::device_vector<Edge> cuda_vertex_offset_d = cuda_vertex_offset;
    thrust::device_vector<uint> cuda_vertex_offset_depth_d = cuda_vertex_offset_depth;

    // Sort the edges ascending to vertex_id_0.
    sorted_edges_vertex_id_0 = edges;
    thrust::device_vector<Edge> sorted_edges_vertex_id_0_d = sorted_edges_vertex_id_0;
    thrust::sort(sorted_edges_vertex_id_0_d.begin(), sorted_edges_vertex_id_0_d.end(), VertexID0Cmp());
    thrust::copy(
        sorted_edges_vertex_id_0_d.begin(), sorted_edges_vertex_id_0_d.end(), sorted_edges_vertex_id_0.begin());

    // Create edge -> sorted_edges_vertex_id_0_d offset
    sorted_edges_offset_v_0.reserve(edge_cnt);
    cur = sorted_edges_vertex_id_0[0].vertex_id_0;
    sorted_edges_offset_v_0.push_back(0);
    for (uint i = 0; i < edge_cnt; i++) {
        if (sorted_edges_vertex_id_0[i].vertex_id_0 != cur) {
            auto diff = sorted_edges_vertex_id_0[i].vertex_id_0 - cur;
            if (diff != 1) {
                for (uint j = 0; j < diff - 1; j++) {
                    sorted_edges_offset_v_0.push_back(i);
                }
            }
            sorted_edges_offset_v_0.push_back(i);
            cur = sorted_edges_vertex_id_0[i].vertex_id_0;
        }
    }
    sorted_edges_offset_v_0.push_back(edge_cnt);

    // Upload and delete local copy.
    thrust::device_vector<uint> sorted_edges_offset_v_0_d = sorted_edges_offset_v_0;
    sorted_edges_vertex_id_0.clear();
    sorted_edges_vertex_id_0.shrink_to_fit();
    sorted_edges_offset_v_0.clear();
    sorted_edges_offset_v_0.shrink_to_fit();

    // Sort the edges ascending to vertex_id_1.
    sorted_edges_vertex_id_1 = edges;
    thrust::device_vector<Edge> sorted_edges_vertex_id_1_d = sorted_edges_vertex_id_1;
    thrust::sort(sorted_edges_vertex_id_1_d.begin(), sorted_edges_vertex_id_1_d.end(), VertexID1Cmp());
    thrust::copy(
        sorted_edges_vertex_id_1_d.begin(), sorted_edges_vertex_id_1_d.end(), sorted_edges_vertex_id_1.begin());

    // Delete edges as we don't need them any more.
    edges.clear();
    edges.shrink_to_fit();

    // Create edge -> sorted_edges_vertex_id_1_d offset
    sorted_edges_offset_v_1.reserve(edge_cnt);
    cur = sorted_edges_vertex_id_1[0].vertex_id_1;
    sorted_edges_offset_v_1.push_back(0);
    for (uint i = 0; i < edge_cnt; i++) {
        if (sorted_edges_vertex_id_1[i].vertex_id_1 != cur) {
            auto diff = sorted_edges_vertex_id_1[i].vertex_id_1 - cur;
            if (diff != 1) {
                for (uint j = 0; j < diff - 1; j++) {
                    sorted_edges_offset_v_1.push_back(i);
                }
            }
            sorted_edges_offset_v_1.push_back(i);
            cur = sorted_edges_vertex_id_1[i].vertex_id_1;
        }
    }
    sorted_edges_offset_v_1.push_back(edge_cnt);

    // Upload and delete local copy.
    thrust::device_vector<uint> sorted_edges_offset_v_1_d = sorted_edges_offset_v_1;
    sorted_edges_vertex_id_1.clear();
    sorted_edges_vertex_id_1.shrink_to_fit();
    sorted_edges_offset_v_1.clear();
    sorted_edges_offset_v_1.shrink_to_fit();

    // Set vertex edge offset.
    SetVertexEdgeOffset<<<Grid(face_cnt, 256), 256>>>(thrust::raw_pointer_cast(cuda_vertex_offset_d.data().get()),
        thrust::raw_pointer_cast(sorted_edges_vertex_id_0_d.data().get()),
        thrust::raw_pointer_cast(sorted_edges_vertex_id_1_d.data().get()),
        thrust::raw_pointer_cast(sorted_edges_offset_v_0_d.data().get()),
        thrust::raw_pointer_cast(sorted_edges_offset_v_1_d.data().get()),
        thrust::raw_pointer_cast(cuda_vertex_offset_depth_d.data().get()), vertex_cnt);
    checkCudaErrors(cudaDeviceSynchronize());
    thrust::copy(cuda_vertex_offset_d.begin(), cuda_vertex_offset_d.end(), cuda_vertex_offset.begin());
    // Reseize the offsets for each vertex and delete cuda offset depth.
    resizeOffsetVertices(
        cuda_vertex_offset, p_vertex_edge_offset, p_vertex_edge_offset_depth, cuda_vertex_offset_depth, Edge());
    cuda_vertex_offset_depth.clear();
    cuda_vertex_offset_depth.shrink_to_fit();

    // Delete the cuda vertex offset vector and the corresponding device vector.
    cuda_vertex_offset_d.clear();
    cuda_vertex_offset_d.shrink_to_fit();
    cuda_vertex_offset.clear();
    cuda_vertex_offset.shrink_to_fit();

    // Delete the sorted edges on device and host.
    sorted_edges_vertex_id_0_d.clear();
    sorted_edges_vertex_id_0_d.shrink_to_fit();
    sorted_edges_vertex_id_1_d.clear();
    sorted_edges_vertex_id_1_d.shrink_to_fit();
    sorted_edges_offset_v_0_d.clear();
    sorted_edges_offset_v_0_d.shrink_to_fit();
    sorted_edges_offset_v_1_d.clear();
    sorted_edges_offset_v_1_d.shrink_to_fit();

    return true;
}

/*
 * CUDAKernels::CreatePhiValues
 */
bool CUDAKernels::CreatePhiValues(const float p_threshold, std::vector<float>& p_phi_values,
    std::vector<bool> p_valid_phi_values, const std::vector<std::vector<Edge>>& p_vertex_edge_offset,
    const std::vector<uint>& p_vertex_edge_offset_depth, const std::vector<int>& p_vertex_type) {
    // Convert vertex edge offset to CUDA
    uint vertex_cnt = static_cast<uint>(p_phi_values.size());
    std::vector<Edge> cuda_vertex_offset;
    cuda_vertex_offset.reserve(static_cast<size_t>(p_vertex_edge_offset_depth.back()) * 30);

    for (const auto& offset : p_vertex_edge_offset) {
        for (const auto& edge : offset) {
            cuda_vertex_offset.push_back(edge);
        }
    }

    // Store the vertex IDs of neighbouring vertices.
    std::vector<uint> vertex_neighbours = std::vector<uint>(cuda_vertex_offset.size());

    // Store the offset for neighbours with the type of -1.
    std::vector<uint> vertex_neighbours_offset = std::vector<uint>(vertex_cnt, 0);

    // Upload data and delete local copy.
    thrust::device_vector<float> p_phi_values_one_d = p_phi_values;
    thrust::device_vector<float> p_phi_values_two_d = p_phi_values;
    thrust::device_vector<bool> p_valid_phi_values_d = p_valid_phi_values;
    thrust::device_vector<Edge> cuda_vertex_offset_d = cuda_vertex_offset;
    thrust::device_vector<uint> p_vertex_edge_offset_depth_d = p_vertex_edge_offset_depth;
    thrust::device_vector<int> p_vertex_type_d = p_vertex_type;
    thrust::device_vector<uint> vertex_neighbours_d = vertex_neighbours;
    thrust::device_vector<uint> vertex_neighbours_offset_d = vertex_neighbours_offset;
    cuda_vertex_offset.clear();
    cuda_vertex_offset.shrink_to_fit();
    vertex_neighbours.clear();
    vertex_neighbours.shrink_to_fit();
    vertex_neighbours_offset.clear();
    vertex_neighbours_offset.shrink_to_fit();

    // Get the neighbours of every vertex.
    GetNeighbourIds<<<Grid(vertex_cnt, 256), 256>>>(thrust::raw_pointer_cast(vertex_neighbours_d.data().get()),
        thrust::raw_pointer_cast(p_valid_phi_values_d.data().get()),
        thrust::raw_pointer_cast(cuda_vertex_offset_d.data().get()),
        thrust::raw_pointer_cast(p_vertex_edge_offset_depth_d.data().get()), vertex_cnt);
    checkCudaErrors(cudaDeviceSynchronize());

    // Get the offsets for the neighbours with the type of -1.
    SortNeighbourIds<<<Grid(vertex_cnt, 256), 256>>>(thrust::raw_pointer_cast(vertex_neighbours_d.data().get()),
        thrust::raw_pointer_cast(vertex_neighbours_offset_d.data().get()),
        thrust::raw_pointer_cast(p_valid_phi_values_d.data().get()),
        thrust::raw_pointer_cast(p_vertex_edge_offset_depth_d.data().get()),
        thrust::raw_pointer_cast(p_vertex_type_d.data().get()), vertex_cnt);
    checkCudaErrors(cudaDeviceSynchronize());

    // Perform iterations.
    float diff = 2.0f * p_threshold;
    size_t round = 0;
    while (diff > p_threshold) {
        SetPhiValues<<<Grid(vertex_cnt, 256), 256>>>(thrust::raw_pointer_cast(p_phi_values_one_d.data().get()),
            thrust::raw_pointer_cast(p_phi_values_two_d.data().get()),
            thrust::raw_pointer_cast(p_valid_phi_values_d.data().get()),
            thrust::raw_pointer_cast(vertex_neighbours_d.data().get()),
            thrust::raw_pointer_cast(p_vertex_edge_offset_depth_d.data().get()),
            thrust::raw_pointer_cast(p_vertex_type_d.data().get()),
            thrust::raw_pointer_cast(vertex_neighbours_offset_d.data().get()), vertex_cnt);
        checkCudaErrors(cudaDeviceSynchronize());

        SetPhiValues<<<Grid(vertex_cnt, 256), 256>>>(thrust::raw_pointer_cast(p_phi_values_two_d.data().get()),
            thrust::raw_pointer_cast(p_phi_values_one_d.data().get()),
            thrust::raw_pointer_cast(p_valid_phi_values_d.data().get()),
            thrust::raw_pointer_cast(vertex_neighbours_d.data().get()),
            thrust::raw_pointer_cast(p_vertex_edge_offset_depth_d.data().get()),
            thrust::raw_pointer_cast(p_vertex_type_d.data().get()),
            thrust::raw_pointer_cast(vertex_neighbours_offset_d.data().get()), vertex_cnt);
        checkCudaErrors(cudaDeviceSynchronize());

        // Check the difference between the two.
        if (round % 5000 == 0) {
            float first_sum = thrust::reduce(p_phi_values_two_d.begin(), p_phi_values_two_d.end());
            float second_sum = thrust::reduce(p_phi_values_one_d.begin(), p_phi_values_one_d.end());
            diff = abs(second_sum - first_sum);
        }
        round++;
    }
    thrust::copy(p_phi_values_one_d.begin(), p_phi_values_one_d.end(), p_phi_values.begin());

    return true;
}

/*
 * CUDAKernels::CreateThetaValues
 */
bool CUDAKernels::CreateThetaValues(const float p_theta, std::vector<float>& p_theta_values,
    std::vector<bool> p_valid_theta_vertices, const std::vector<float>& p_zvalues, const float p_min_z) {
    uint vertex_cnt = static_cast<uint>(p_theta_values.size());

    // Upload data.
    thrust::device_vector<float> p_theta_values_d = p_theta_values;
    thrust::device_vector<float> p_zvalues_d = p_zvalues;
    thrust::device_vector<bool> p_valid_theta_vertices_d = p_valid_theta_vertices;

    // Call the SetThetaValues kernel.
    float tconst = -p_min_z / (log(sin(p_theta) / (1.0f + cos(p_theta))));
    SetTheataValues<<<Grid(vertex_cnt, 256), 256>>>(thrust::raw_pointer_cast(p_theta_values_d.data().get()),
        thrust::raw_pointer_cast(p_valid_theta_vertices_d.data().get()),
        thrust::raw_pointer_cast(p_zvalues_d.data().get()), p_theta, tconst, vertex_cnt);
    checkCudaErrors(cudaDeviceSynchronize());
    thrust::copy(p_theta_values_d.begin(), p_theta_values_d.end(), p_theta_values.begin());

    return true;
}

/*
 * CUDAKernels::CreateZValues
 */
bool CUDAKernels::CreateZValues(const uint p_iterations, std::vector<float>& p_zvalues,
    std::vector<bool> p_valid_z_values, const std::vector<std::vector<Edge>>& p_vertex_edge_offset,
    const std::vector<uint>& p_vertex_edge_offset_depth) {
    // Convert vertex edge offset to CUDA
    uint vertex_cnt = static_cast<uint>(p_zvalues.size());
    std::vector<Edge> cuda_vertex_offset;
    cuda_vertex_offset.reserve(static_cast<size_t>(p_vertex_edge_offset_depth.back()) * 30);

    for (const auto& offset : p_vertex_edge_offset) {
        for (const auto& edge : offset) {
            cuda_vertex_offset.push_back(edge);
        }
    }

    // Store the vertex IDs of neighbouring vertices.
    std::vector<uint> vertex_neighbours = std::vector<uint>(cuda_vertex_offset.size());

    // Upload data and delete local copy.
    thrust::device_vector<float> p_zvalues_d = p_zvalues;
    thrust::device_vector<bool> p_valid_z_values_d = p_valid_z_values;
    thrust::device_vector<Edge> cuda_vertex_offset_d = cuda_vertex_offset;
    thrust::device_vector<uint> p_vertex_edge_offset_depth_d = p_vertex_edge_offset_depth;
    thrust::device_vector<uint> vertex_neighbours_d = vertex_neighbours;
    cuda_vertex_offset.clear();
    cuda_vertex_offset.shrink_to_fit();
    vertex_neighbours.clear();
    vertex_neighbours.shrink_to_fit();

    // Get the neighbours of every vertex.
    GetNeighbourIds<<<Grid(vertex_cnt, 256), 256>>>(thrust::raw_pointer_cast(vertex_neighbours_d.data().get()),
        thrust::raw_pointer_cast(p_valid_z_values_d.data().get()),
        thrust::raw_pointer_cast(cuda_vertex_offset_d.data().get()),
        thrust::raw_pointer_cast(p_vertex_edge_offset_depth_d.data().get()), vertex_cnt);
    checkCudaErrors(cudaDeviceSynchronize());

    // Perform iterations.
    for (uint i = 0; i < p_iterations; i++) {
        SetZValues<<<Grid(vertex_cnt, 256), 256>>>(thrust::raw_pointer_cast(p_zvalues_d.data().get()),
            thrust::raw_pointer_cast(p_valid_z_values_d.data().get()),
            thrust::raw_pointer_cast(vertex_neighbours_d.data().get()),
            thrust::raw_pointer_cast(p_vertex_edge_offset_depth_d.data().get()), vertex_cnt);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    thrust::copy(p_zvalues_d.begin(), p_zvalues_d.end(), p_zvalues.begin());

    return true;
}

/*
 * CUDAKernels::CUDAKernels
 */
CUDAKernels::CUDAKernels(void) {}

/*
 * CUDAKernels::GroupFaces
 */
std::vector<FaceGroup> CUDAKernels::GroupFaces(const std::vector<bool>& p_face_shadowed,
    const std::vector<std::vector<Edge>>& p_face_edge_offset, const std::vector<uint>& p_face_edge_offset_depth,
    const vislib::Array<vec3f>& p_group_colour_table, const std::vector<uint>& p_faces,
    std::vector<float>& p_vertexColors_group) {
    size_t face_cnt = p_face_shadowed.size();

    // Initialise the group IDs so that each face has a unique ID.
    std::vector<uint> face_group_ids = std::vector<uint>(face_cnt, 0);
    uint group_id = 1;
    for (size_t i = 0; i < face_cnt; i++) {
        if (p_face_shadowed[i]) {
            face_group_ids[i] = group_id++;
        }
    }

    // Convert face edge offset to CUDA
    std::vector<Edge> cuda_face_offset;
    cuda_face_offset.reserve(face_cnt * 6);

    for (const auto& offset : p_face_edge_offset) {
        for (const auto& edge : offset) {
            cuda_face_offset.push_back(edge);
        }
    }

    // Upload data.
    thrust::device_vector<uint> face_group_ids_d = face_group_ids;
    thrust::device_vector<bool> face_shadowed_d = p_face_shadowed;
    thrust::device_vector<Edge> cuda_face_offset_d = cuda_face_offset;
    thrust::device_vector<uint> face_edge_offset_depth_d = p_face_edge_offset_depth;

    // Perform grouping.
    bool changes = true;
    while (changes) {
        // Reset the changes to false.
        changes = false;
        cudaMemcpyToSymbol(changes_d, &changes, 1 * sizeof(bool));
        checkCudaErrors(cudaDeviceSynchronize());

        // Run the kernel.
        GroupFacesKernel<<<Grid(face_cnt, 256), 256>>>(thrust::raw_pointer_cast(face_group_ids_d.data().get()),
            thrust::raw_pointer_cast(face_shadowed_d.data().get()),
            thrust::raw_pointer_cast(cuda_face_offset_d.data().get()),
            thrust::raw_pointer_cast(face_edge_offset_depth_d.data().get()), face_cnt);
        checkCudaErrors(cudaDeviceSynchronize());

        // Check if we had any changes to face IDs.
        cudaMemcpyFromSymbol(&changes, changes_d, 1 * sizeof(bool));
        checkCudaErrors(cudaDeviceSynchronize());
    }
    // Copy the data back to host memory.
    thrust::copy(face_group_ids_d.begin(), face_group_ids_d.end(), face_group_ids.begin());

    // Create correct IDs for the groups based on the final IDs after the GPU grouping.
    thrust::device_vector<uint> sorted_face_group_ids_d = face_group_ids_d;
    thrust::sort(sorted_face_group_ids_d.begin(), sorted_face_group_ids_d.end());
    std::vector<uint> sorted_face_group_ids = std::vector<uint>(face_cnt, 0);
    thrust::copy(sorted_face_group_ids_d.begin(), sorted_face_group_ids_d.end(), sorted_face_group_ids.begin());

    // Get the highest ID and use it to initialise the offset vector.
    size_t group_cnt = static_cast<size_t>(sorted_face_group_ids.back() + 1);
    std::vector<int> new_group_ids_offset = std::vector<int>(group_cnt, -1);
    std::vector<size_t> group_sizes = std::vector<size_t>(group_cnt, 0);

    // Create the offset vector and remember the size of the groups.
    int cur_id = 0;
    new_group_ids_offset[sorted_face_group_ids[0]] = cur_id++;
    size_t size = 1;
    group_cnt = 1;
    for (size_t i = 1; i < face_cnt; i++) {
        if (sorted_face_group_ids[i - 1] != sorted_face_group_ids[i]) {
            group_sizes[cur_id - 1] = size;
            size = 1;
            new_group_ids_offset[sorted_face_group_ids[i]] = cur_id++;
            group_cnt++;

        } else {
            size++;
        }
    }

    // Create face groups.
    std::vector<FaceGroup> groups = std::vector<FaceGroup>(group_cnt);

    // Add the faces to the groups.
    vec3ui vertex_ids;
    int colour_table_size = static_cast<int>(p_group_colour_table.Count());
    for (size_t i = 0; i < face_cnt; i++) {
        auto idx = new_group_ids_offset[face_group_ids[i]];
        groups[idx].AddFace(static_cast<uint>(i));
        groups[idx].state = p_face_shadowed[i];

        // Set the colour of the vertices.
        vertex_ids.Set(p_faces[i * 3], p_faces[i * 3 + 1], p_faces[i * 3 + 2]);
        p_vertexColors_group[vertex_ids.GetX() * 3] = p_group_colour_table[idx % colour_table_size].GetX();
        p_vertexColors_group[vertex_ids.GetX() * 3 + 1] = p_group_colour_table[idx % colour_table_size].GetY();
        p_vertexColors_group[vertex_ids.GetX() * 3 + 2] = p_group_colour_table[idx % colour_table_size].GetZ();
        p_vertexColors_group[vertex_ids.GetY() * 3] = p_group_colour_table[idx % colour_table_size].GetX();
        p_vertexColors_group[vertex_ids.GetY() * 3 + 1] = p_group_colour_table[idx % colour_table_size].GetY();
        p_vertexColors_group[vertex_ids.GetY() * 3 + 2] = p_group_colour_table[idx % colour_table_size].GetZ();
        p_vertexColors_group[vertex_ids.GetZ() * 3] = p_group_colour_table[idx % colour_table_size].GetX();
        p_vertexColors_group[vertex_ids.GetZ() * 3 + 1] = p_group_colour_table[idx % colour_table_size].GetY();
        p_vertexColors_group[vertex_ids.GetZ() * 3 + 2] = p_group_colour_table[idx % colour_table_size].GetZ();
    }

    return groups;
}

/*
 * CUDAKernels::resizeOffsetVertices
 */
void CUDAKernels::resizeOffsetVertices(std::vector<Edge>& p_offset_in, std::vector<std::vector<Edge>>& p_offset_out,
    std::vector<uint>& p_offset_depth, const std::vector<uint>& p_depth, const Edge p_value) {
    size_t idx = 0;
    size_t cnt = 0;
    uint modulo_cnt = 0;
    uint sum = 0;
    p_offset_depth[0] = 0;
    std::vector<bool> marked_edge = std::vector<bool>(p_offset_in.size(), false);
    for (size_t i = 0; i < p_offset_in.size(); i++) {
        if (p_offset_in[i] != p_value) {
            // Remove the edges in the opposite direction.
            if (!marked_edge[p_offset_in[i].edge_id]) {
                if (p_offset_in[i].opposite_edge_id != -1) {
                    marked_edge[p_offset_in[i].opposite_edge_id] = true;
                    p_offset_in[i].opposite_edge_id = -1;
                }

                p_offset_out[idx][cnt] = p_offset_in[i];
                cnt++;
                sum++;
            }
        }

        modulo_cnt++;
        if (modulo_cnt == p_depth[idx + 1]) {
            p_offset_depth[idx + 1] = sum;
            p_offset_out[idx].resize(cnt);
            idx++;
            cnt = 0;
        }
    }
}

/*
 * CUDAKernels::resizeOffsetFaces
 */
void CUDAKernels::resizeOffsetFaces(std::vector<Edge>& p_offset_in, std::vector<std::vector<Edge>>& p_offset_out,
    std::vector<uint>& p_offset_depth, const std::vector<uint>& p_depth) {
    size_t idx = 0;
    size_t cnt = 0;
    uint modulo_cnt = 0;
    uint sum = 0;
    p_offset_depth[0] = 0;
    for (size_t i = 0; i < p_offset_in.size(); i++) {
        // Remove the edges in the opposite direction.
        if (p_offset_in[i].face_id_0 == idx) {
            p_offset_out[idx][cnt] = p_offset_in[i];
            cnt++;
            sum++;
        }

        modulo_cnt++;
        if (modulo_cnt == p_depth[idx + 1]) {
            p_offset_depth[idx + 1] = sum;
            p_offset_out[idx].resize(cnt);
            idx++;
            cnt = 0;
        }
    }
}

/*
 * CUDAKernels::SortEdges
 */
bool CUDAKernels::SortEdges(std::vector<Edge>& p_edges, const uint p_id) {
    // Upload the data.
    thrust::device_vector<Edge> edges_d = p_edges;

    // Sort the data.
    if (p_id == 0) {
        thrust::sort(edges_d.begin(), edges_d.end(), VertexID0Cmp());

    } else if (p_id == 1) {
        thrust::sort(edges_d.begin(), edges_d.end(), VertexID1Cmp());

    } else {
        return false;
    }

    // Download the data.
    thrust::copy(edges_d.begin(), edges_d.end(), p_edges.begin());

    return true;
}
