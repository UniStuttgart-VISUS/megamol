/*
 * CUDAKernels.cu
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "SombreroKernels.cuh"

using namespace megamol;
using namespace megamol::protein_cuda;

__constant__ float C_PI = 3.141592653589f;

/**
 * CUDA symbol for the face grouping will be false if no
 * further changes where made to the face IDs.
 */
__device__ bool changes_d;

/**
 * Smaller operator for edges.
 */
__host__ __device__ bool operator<(const SombreroKernels::Edge& lhs, const SombreroKernels::Edge& rhs) {
    return lhs.vertex_id_1 < rhs.vertex_id_1;
}

/**
 * Sort edges ascending to the face_id_0.
 */
struct FaceID0Cmp {
    __host__ __device__ bool operator()(const SombreroKernels::Edge& lhs, const SombreroKernels::Edge& rhs) {
        return lhs.face_id_0 < rhs.face_id_0;
    }
};

/**
 * Sort edges ascending to the face_id_1.
 */
struct FaceID1Cmp {
    __host__ __device__ bool operator()(const SombreroKernels::Edge& lhs, const SombreroKernels::Edge& rhs) {
        return lhs.face_id_1 < rhs.face_id_1;
    }
};

/**
 * Sort edges ascending to the vertex_id_0.
 */
struct VertexID0Cmp {
    __host__ __device__ bool operator()(const SombreroKernels::Edge& lhs, const SombreroKernels::Edge& rhs) {
        return lhs.vertex_id_0 < rhs.vertex_id_0;
    }
};

/**
 * Sort edges ascending to the vertex_id_1.
 */
struct VertexID1Cmp {
    __host__ __device__ bool operator()(const SombreroKernels::Edge& lhs, const SombreroKernels::Edge& rhs) {
        return lhs.vertex_id_1 < rhs.vertex_id_1;
    }
};


/**
 * Get the thread index based on the current CUDE grid dimensions.
 *
 * @return Returns the thread index based on the current CUDA grid
 *            dimensions.
 */
__device__ uint GetCurThreadIndex() {
    return __umul24(__umul24(blockIdx.y, gridDim.x) + blockIdx.x, blockDim.x) + threadIdx.x;
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
__global__ void GetNeighbourIds(uint* p_neighbour_ids, bool* p_valid_z_values,
    const SombreroKernels::Edge* p_vertex_edge_offset, const uint* p_vertex_edge_offset_depth, uint p_vertex_cnt,
    uint p_edge_cnt) {
    const uint idx = GetCurThreadIndex();
    if (idx >= p_vertex_cnt) return;
    if (!p_valid_z_values[idx]) return;

    uint begin = p_vertex_edge_offset_depth[idx];
    uint end = p_vertex_edge_offset_depth[idx + 1];
    if (idx == p_vertex_cnt - 1) {
        end = p_edge_cnt; // necessary for the last vertex
    }
    uint neighbour_id;
    // Find the IDs of neighbouring vertices.
    for (uint i = begin; i < end; i++) {
        SombreroKernels::Edge cur = p_vertex_edge_offset[i];
        if (cur.vertex_id_0 == idx)
            neighbour_id = cur.vertex_id_1;
        else
            neighbour_id = cur.vertex_id_0;
        p_neighbour_ids[i] = neighbour_id;
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
__global__ void InitEdges(const uint3* p_faces, SombreroKernels::Edge* p_edges, uint p_face_cnt) {
    const uint idx = GetCurThreadIndex();
    if (idx >= p_face_cnt) return;

    // First edge.
    {
        SombreroKernels::Edge cur = p_edges[idx * 3];
        cur.face_id_0 = idx;
        cur.face_id_1 = -1;
        cur.vertex_id_0 = p_faces[idx].x;
        cur.vertex_id_1 = p_faces[idx].y;
        cur.edge_id = idx * 3;
        p_edges[idx * 3] = cur;
    }

    // Second edge.
    {
        SombreroKernels::Edge cur = p_edges[idx * 3 + 1];
        cur.face_id_0 = idx;
        cur.face_id_1 = -1;
        cur.vertex_id_0 = p_faces[idx].y;
        cur.vertex_id_1 = p_faces[idx].z;
        cur.edge_id = idx * 3 + 1;
        p_edges[idx * 3 + 1] = cur;
    }

    // Third edge.
    {
        SombreroKernels::Edge cur = p_edges[idx * 3 + 2];
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
__global__ void MatchEdges(
    SombreroKernels::Edge* p_edges, SombreroKernels::Edge* p_sorted_edges, uint* p_edge_offset, uint p_edge_cnt) {
    const uint idx = GetCurThreadIndex();
    if (idx >= p_edge_cnt) return;

    // Get current edge and check if it is already matched.
    SombreroKernels::Edge cur = p_edges[idx];
    if (cur.face_id_1 != -1) return;

    // Find the same edge faceing in the opposit direction.
    uint begin = p_edge_offset[cur.vertex_id_0];
    uint end = p_edge_offset[cur.vertex_id_0 + 1];
    for (uint i = begin; i < end; i++) {
        uint id = p_sorted_edges[i].edge_id;

        if (i == idx) continue;
        if (p_edges[id].face_id_1 != -1) continue;

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
__global__ void SetFaceEdgeOffset(SombreroKernels::Edge* p_face_edge_offset, SombreroKernels::Edge* p_face_id_0_offset,
    SombreroKernels::Edge* p_face_id_1_offset, uint depth, uint p_face_cnt) {
    const uint idx = GetCurThreadIndex();
    if (idx >= p_face_cnt) return;

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
__global__ void SetVertexEdgeOffset(SombreroKernels::Edge* p_vertex_edge_offset,
    SombreroKernels::Edge* p_vertex_id_0_sorted, SombreroKernels::Edge* p_vertex_id_1_sorted,
    uint* p_vertex_id_0_offset, uint* p_vertex_id_1_offset, uint* depth, uint p_vertex_cnt, uint p_edge_cnt) {
    const uint idx = GetCurThreadIndex();
    if (idx >= p_vertex_cnt) return;

    // Find edges that belong to the vertex.
    uint cur_depth = depth[idx];

    // Edges with vertex_id_0 == idx
    uint begin = p_vertex_id_0_offset[idx];
    uint end = p_vertex_id_0_offset[idx + 1];
    if (idx == p_vertex_cnt - 1) {
        end = p_edge_cnt; // necessary for the last vertex
    }
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
    const uint* p_vertex_neighbours_offset, uint p_vertex_cnt, uint p_edge_cnt) {
    const uint idx = GetCurThreadIndex();
    if (idx >= p_vertex_cnt) return;
    if (!p_valid_phi_values[idx]) return;

    uint begin = p_vertex_edge_offset_depth[idx] + p_vertex_neighbours_offset[idx];
    uint end = p_vertex_edge_offset_depth[idx + 1];
    if (idx == p_vertex_cnt - 1) {
        end = p_edge_cnt; // necessary for the last vertex
    }
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
    p_phivalues_out[idx] = p_phivalues_in[idx] + tmp_phi * 1.1f;
}


/**
 * Compute the average z value in the neighbourhood and assign it to the
 * current vertex.
 *
 * @param p_zvalues The z value of each vertex.
 * @param p_valid_z_values Remebers if a vertex is valid.
 * @param p_vertex_neighbours The IDs of the neighbouring vertices.
 * @param p_vertex_edge_offset_depth The number of edges per vertex.
 * @param p_vertex_multiplicity The multiplicity factors of the vertex
 * @param p_vertex_cnt The number of vertices in the mesh.
 * @param p_edge_cnt The total number of edges of the mesh.
 */
__global__ void SetZValues(float* p_zvalues, bool* p_valid_z_values, const uint* p_vertex_neighbours,
    const uint* p_vertex_edge_offset_depth, const uint* p_vertex_multiplicity, uint p_vertex_cnt, uint p_edge_cnt) {
    const uint idx = GetCurThreadIndex();
    if (idx >= p_vertex_cnt) return;
    if (!p_valid_z_values[idx]) return;

    uint begin = p_vertex_edge_offset_depth[idx];
    uint end = p_vertex_edge_offset_depth[idx + 1];
    if (idx == p_vertex_cnt - 1) {
        end = p_edge_cnt; // necessary for the last vertex
    }
    float tmp = 0.0f;
    float multCount = 0.0f;
    // Add up the zvalues of the neighbouring vertices and increase the counter.
    for (uint i = begin; i < end; i++) {
        uint mult = p_vertex_multiplicity[p_vertex_neighbours[i]];
        for (uint j = 0; j < mult; j++) {
            tmp += p_zvalues[p_vertex_neighbours[i]];
            multCount += 1.0f;
        }
    }
    float tmp_z = (tmp / multCount) - p_zvalues[idx];
    p_zvalues[idx] = p_zvalues[idx] + tmp_z * 1.1f;
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
    const uint* p_vertex_edge_offset_depth, const int* p_vertex_type, uint p_vertex_cnt, uint p_edge_cnt) {
    const uint idx = GetCurThreadIndex();
    if (idx >= p_vertex_cnt) return;
    if (!p_valid_z_values[idx]) return;

    uint begin = p_vertex_edge_offset_depth[idx];
    uint end = p_vertex_edge_offset_depth[idx + 1];
    if (idx == p_vertex_cnt - 1) {
        end = p_edge_cnt; // necessary for the last vertex
    }
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
 * SombreroKernels::~CUDAKernels
 */
SombreroKernels::~SombreroKernels(void) {}

/*
 * SombreroKernels::CreatePhiValues
 */
bool SombreroKernels::CreatePhiValues(const float p_threshold, std::vector<float>& p_phi_values,
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
    uint edge_cnt = static_cast<uint>(cuda_vertex_offset.size());
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
        thrust::raw_pointer_cast(p_vertex_edge_offset_depth_d.data().get()), vertex_cnt, edge_cnt);
    checkCudaErrors(cudaDeviceSynchronize());

    // Get the offsets for the neighbours with the type of -1.
    SortNeighbourIds<<<Grid(vertex_cnt, 256), 256>>>(thrust::raw_pointer_cast(vertex_neighbours_d.data().get()),
        thrust::raw_pointer_cast(vertex_neighbours_offset_d.data().get()),
        thrust::raw_pointer_cast(p_valid_phi_values_d.data().get()),
        thrust::raw_pointer_cast(p_vertex_edge_offset_depth_d.data().get()),
        thrust::raw_pointer_cast(p_vertex_type_d.data().get()), vertex_cnt, edge_cnt);
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
            thrust::raw_pointer_cast(vertex_neighbours_offset_d.data().get()), vertex_cnt, edge_cnt);
        checkCudaErrors(cudaDeviceSynchronize());

        SetPhiValues<<<Grid(vertex_cnt, 256), 256>>>(thrust::raw_pointer_cast(p_phi_values_two_d.data().get()),
            thrust::raw_pointer_cast(p_phi_values_one_d.data().get()),
            thrust::raw_pointer_cast(p_valid_phi_values_d.data().get()),
            thrust::raw_pointer_cast(vertex_neighbours_d.data().get()),
            thrust::raw_pointer_cast(p_vertex_edge_offset_depth_d.data().get()),
            thrust::raw_pointer_cast(p_vertex_type_d.data().get()),
            thrust::raw_pointer_cast(vertex_neighbours_offset_d.data().get()), vertex_cnt, edge_cnt);
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
 * SombreroKernels::CreateZValues
 */
bool SombreroKernels::CreateZValues(const uint p_iterations, std::vector<float>& p_zvalues,
    std::vector<bool> p_valid_z_values, const std::vector<std::vector<Edge>>& p_vertex_edge_offset,
    const std::vector<uint>& p_vertex_edge_offset_depth, const std::vector<uint>& p_vertex_multiplicity) {
    // Convert vertex edge offset to CUDA
    uint vertex_cnt = static_cast<uint>(p_zvalues.size());
    std::vector<SombreroKernels::Edge> cuda_vertex_offset;
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
    thrust::device_vector<SombreroKernels::Edge> cuda_vertex_offset_d = cuda_vertex_offset;
    thrust::device_vector<uint> p_vertex_edge_offset_depth_d = p_vertex_edge_offset_depth;
    thrust::device_vector<uint> vertex_neighbours_d = vertex_neighbours;
    thrust::device_vector<uint> p_vertex_multiplicity_d = p_vertex_multiplicity;
    uint edge_cnt = static_cast<uint>(cuda_vertex_offset.size());
    cuda_vertex_offset.clear();
    cuda_vertex_offset.shrink_to_fit();
    vertex_neighbours.clear();
    vertex_neighbours.shrink_to_fit();

    // Get the neighbours of every vertex.
    GetNeighbourIds<<<Grid(vertex_cnt, 256), 256>>>(thrust::raw_pointer_cast(vertex_neighbours_d.data().get()),
        thrust::raw_pointer_cast(p_valid_z_values_d.data().get()),
        thrust::raw_pointer_cast(cuda_vertex_offset_d.data().get()),
        thrust::raw_pointer_cast(p_vertex_edge_offset_depth_d.data().get()), vertex_cnt, edge_cnt);
    checkCudaErrors(cudaDeviceSynchronize());

    // Perform iterations.
    for (uint i = 0; i < p_iterations; i++) {
        SetZValues<<<Grid(vertex_cnt, 256), 256>>>(thrust::raw_pointer_cast(p_zvalues_d.data().get()),
            thrust::raw_pointer_cast(p_valid_z_values_d.data().get()),
            thrust::raw_pointer_cast(vertex_neighbours_d.data().get()),
            thrust::raw_pointer_cast(p_vertex_edge_offset_depth_d.data().get()),
            thrust::raw_pointer_cast(p_vertex_multiplicity_d.data().get()), vertex_cnt, edge_cnt);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    thrust::copy(p_zvalues_d.begin(), p_zvalues_d.end(), p_zvalues.begin());

    return true;
}

/*
 * SombreroKernels::SombreroKernels
 */
SombreroKernels::SombreroKernels(void) {}

/*
 * SombreroKernels::SortEdges
 */
bool SombreroKernels::SortEdges(std::vector<Edge>& p_edges, const uint p_id) {
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
