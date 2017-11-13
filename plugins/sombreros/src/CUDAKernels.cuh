/*
 * CUDAKernels.cuh
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MMSOMBREROSPLUGIN_CUDAKERNELS_CUH_INCLUDED
#define MMSOMBREROSPLUGIN_CUDAKERNELS_CUH_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "helper_includes/helper_cuda.h"
#include "helper_includes/exception.h"
#include "helper_includes/helper_math.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "stdafx.h"

 /**
  * The equal operator for the CUDA float3 type.
  */
inline bool operator==(const float3& lhs, const float3& rhs) {
	return ((lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z));
}

namespace megamol {
namespace sombreros {
	class CUDAKernels {
	public:

		/**
		 * The definition of an Edge.
		 */
		struct Edge {
			/**
			* Initialises an empty instance.
			*/
			__host__ __device__ Edge(void) : edge_id(0), face_id_0(-1), face_id_1(-1),
				opposite_edge_id(-1), vertex_id_0(0), vertex_id_1(0)
			{ }

			/**
			* Initialises an empty instance.
			*/
			Edge(const uint p_edge_id, const int p_face_id_0, const int p_face_id_1,
				const int p_opposite_edge_id, const uint p_vertex_id_0, const uint p_vertex_id_1) :
				edge_id(p_edge_id), face_id_0(p_face_id_0), face_id_1(p_face_id_1),
				opposite_edge_id(p_opposite_edge_id), vertex_id_0(p_vertex_id_0),
				vertex_id_1(p_vertex_id_1)
			{ }

			/**
			* Compare two Edges and return true if they are equal.
			*
			* @param rhs The edge to compare with.
			*
			* @return true if the edges are equal, false otherwise.
			*/
			inline bool operator==(const Edge& rhs) {
				return ((this->face_id_0 == rhs.face_id_0) &&
					(this->face_id_1 == rhs.face_id_1) &&
					(this->vertex_id_0 == rhs.vertex_id_0) &&
					(this->vertex_id_1 == rhs.vertex_id_1));
			}

			/**
			* Compare two Edges and return true if they are not equal.
			*
			* @param rhs The edge to compare with.
			*
			* @return true if the edges are not equal, false otherwise.
			*/
			inline bool operator!=(const Edge& rhs) {
				return ((this->face_id_0 != rhs.face_id_0) ||
					(this->face_id_1 != rhs.face_id_1) ||
					(this->vertex_id_0 != rhs.vertex_id_0) ||
					(this->vertex_id_1 != rhs.vertex_id_1));
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
		~CUDAKernels(void);

		/**
		 * Create the mesh topology by extracting the edges from the
		 * mesh and creating the vertex edge offset and face edge
		 * offset information. The vertex edge offset stores the 
		 * edges to which a vertex blongs and the face edge offset
		 * stores the edges that belong to a face.
		 *
		 * @param p_faces The faces of the mesh.
		 * @param p_vertex_edge_offset Will contain the edge IDs that
		 * a vertex belongs to.
		 * @param p_face_edge_offset Will contain the edge IDs that
		 * belong to a face.
		 * @param p_vertex_edge_offset_depth Will contain the number
		 * of edges that contain the vertex.
		 * @param p_face_edge_offset_depth Will contain the number
		 * of edges that belong to the face.
		 *
		 * @return True if no error occured, false otherwise.
		 */
		bool CreateMeshTopology(const std::vector<uint>& p_faces,
			std::vector<std::vector<Edge>>& p_vertex_edge_offset, 
			std::vector<std::vector<Edge>>& p_face_edge_offset,
			std::vector<uint>& p_vertex_edge_offset_depth,
			std::vector<uint>& p_face_edge_offset_depth);

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
		bool CreatePhiValues(const float p_threshold,
			std::vector<float>& p_phi_values,
			std::vector<bool> p_valid_phi_values,
			const std::vector<std::vector<Edge>>& p_vertex_edge_offset,
			const std::vector<uint>& p_vertex_edge_offset_depth,
			const std::vector<int>& p_vertex_type);

		/**
		 * Compute the theta values for the Rahi and Sharp representation of
		 * a SES as a sphere. Only the phi values of valid vertices are changed. 
		 * The poles are not valid vertices.
		 *
		 * @param p_theta One step on the angle from the north to the south pole.
		 * @param p_theta_values The theta values of each vertex.
		 * @param p_valid_theta_vertices Remebers if a vertex is valid.
		 * @param p_zvalues The z value of each vertex.
		 * @param p_min_z the z value of the south pole.
		 *
		 * @return True if no error occured, false otherwise.
		 */
		bool CreateThetaValues(const float p_theta, std::vector<float>& p_theta_values,
			std::vector<bool> p_valid_theta_vertices, const std::vector<float>& p_zvalues,
			const float p_min_z);
	
		/**
		 * Initialise a empty instance.
		 */
		CUDAKernels(void);

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
	
	private:
	
		/**
		 * Resize each vector inside of the vector to the correct
		 * size. The size is determined by the first element that
		 * is not equal to the given value.
		 *
		 * @param p_offset_in The 1D offset from the kernel.
		 * @param p_offset_out The 2D offset with the correct sizes.
		 * @param p_depth The maximum depth that each offset has.
		 * @param p_depth The initial value that determines the size.
		 * @param p_value The empty edge value that stops the offset loop.
		 */
		void resizeOffsetVertices(std::vector<Edge>& p_offset_in,
			std::vector<std::vector<Edge>>& p_offset_out,
			std::vector<uint>& p_offset_depth,
			const std::vector<uint>& p_depth, const Edge p_value);

		/**
		 * Resize each vector inside of the vector to the correct
		 * size. The size is determined by the first element that
		 * is not equal to the given value.
		 *
		 * @param p_offset_in The 1D offset from the kernel.
		 * @param p_offset_out The 2D offset with the correct sizes.
		 * @param p_depth The maximum depth that each offset has.
		 * @param p_depth The initial value that determines the size.
		 */
		void resizeOffsetFaces(std::vector<Edge>& p_offset_in,
			std::vector<std::vector<Edge>>& p_offset_out,
			std::vector<uint>& p_offset_depth,
			const std::vector<uint>& p_depth);
	};

} /* end namespace molecularmaps */
} /* end namespace megamol */

#endif /* MMMOLMAPPLG_CUDAKERNELS_CUH_INCLUDED */