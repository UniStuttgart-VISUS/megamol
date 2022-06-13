/*
 * CUDAKernels.cuh
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MMPROTEINCUDAPLUGIN_SOMBREROKERNELS_CUH_INCLUDED
#define MMPROTEINCUDAPLUGIN_SOMBREROKERNELS_CUH_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "helper_includes/exception.h"
#include "helper_includes/helper_cuda.h"
#include "helper_includes/helper_math.h"


/**
 * The equal operator for the CUDA float3 type.
 */
inline bool operator==(const float3& lhs, const float3& rhs) {
    return ((lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z));
}

namespace megamol {
namespace protein_cuda {

class SombreroKernels {
public:
    /**
     * The definition of an Edge.
     */
    struct Edge {
        /**
         * Initialises an empty instance.
         */
        __host__ __device__ Edge(void)
            : edge_id(0), face_id_0(-1), face_id_1(-1), opposite_edge_id(-1), vertex_id_0(0), vertex_id_1(0) {}

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
     * Destroy the instance.
     */
    ~SombreroKernels(void);

    /**
     * Compute the phi values for the Rahi and Sharp representation of
     * a SES as a sphere. The iterative process is stopped after
     * the given number of iterations and only the phi values of valid
     * vertices are changed. Vertices like the poles or the vertices on
     * the boundary meridian are not considerd valid.
     *
     * @param p_threshold If the difference between two iterations is
     * smaller than this threshold the iteration stopps (convergence).
     * @param p_phivalues The z value of each vertex.
     * @param p_valid_phi_values Remebers if a vertex is valid.
     * @param p_vertex_edge_offset The edges that contain the vertex.
     * @param p_vertex_edge_offset_depth The number of edges per vertex.
     * @param p_vertex_type The type of the vertex: -1: Pole, 0: vertex
     * is not on the meridian or a neighbour, 1: vertex is on the meridian,
     * 2: vertex is on the "right" side of the meridian and 3: vertex is
     * on the "left" side of the meridian.
     *
     * @return True if no error occured, false otherwise.
     */
    bool CreatePhiValues(const float p_threshold, std::vector<float>& p_phi_values,
        std::vector<bool> p_valid_phi_values, const std::vector<std::vector<Edge>>& p_vertex_edge_offset,
        const std::vector<uint>& p_vertex_edge_offset_depth, const std::vector<int>& p_vertex_type);

    /**
     * Compute the z values for the Rahi and Sharp representation of
     * a SES as a sphere. The iterative process is stopped after
     * the given number of iterations and only the z values of valid
     * vertices are changed. Vertices like the poles or the neighbours
     * of the poles are not considerd valid.
     *
     * @param p_iterations The number of iterations.
     * @param p_zvalues The z value of each vertex.
     * @param p_valid_z_vlaues Remebers if a vertex is valid.
     * @param p_vertex_edge_offset The edges that contain the vertex.
     * @param p_vertex_edge_offset_depth The number of edges per vertex.
     * @param p_vertex_multiplicity Weights for the vertices, determining their influence on the final values
     *
     * @return True if no error occured, false otherwise.
     */
    bool CreateZValues(const uint p_iterations, std::vector<float>& p_zvalues, std::vector<bool> p_valid_z_values,
        const std::vector<std::vector<Edge>>& p_vertex_edge_offset, const std::vector<uint>& p_vertex_edge_offset_depth,
        const std::vector<uint>& p_vertex_multiplicity);

    /**
     * Initialise a empty instance.
     */
    SombreroKernels(void);

    /**
     * Sort the given edges according to the first or seoncd vertex
     * ID based on the p_id parameter.
     *
     * @param p_edges the edges to sort
     * @param p_id sort either based on the first or second vertex id
     *
     * @return false if the id is neither 0 nor 1, true otherwise
     */
    bool SortEdges(std::vector<Edge>& p_edges, const uint p_id);
};

} /* namespace protein_cuda */
} /* end namespace megamol */

#endif /* MMPROTEINCUDAPLUGIN_SOMBREROKERNELS_CUH_INCLUDED */
