/*
 * streamlines.cuh
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include <cuda_runtime_api.h>

#include "real_type.h"

#include <array>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Class for computation of stream lines, corresponding labels and distances on the GPU
        */
        class streamlines_cuda_impl
        {
        public:
            /**
            * Initialize constants and textures
            *
            * @param positions              Positions of the vectors
            * @param vectors                Vectors defining the vector field to analyze
            * @param points                 Convergence structures defined as points
            * @param point_ids              IDs (or labels) of the point convergence structures
            * @param lines                  Convergence structures defined as lines
            * @param line_ids               IDs (or labels) of the line convergence structures
            * @param integration_timestep   Time step factor for advection
            * @param max_integration_error  Maximum error for Runge-Kutta 4-5, above which the time step size has to be adapted
            */
            streamlines_cuda_impl(const std::vector<float>& positions, const std::vector<float>& vectors, const std::vector<float>& points,
                const std::vector<int>& point_ids, const std::vector<float>& lines, const std::vector<int>& line_ids,
                float integration_timestep, float max_integration_error);

            /**
            * Destructor
            */
            ~streamlines_cuda_impl();

            /**
            * Update labels for the given seed
            *
            * @param source                 Seed for advecting stream lines
            * @param labels                 In/output labels
            * @param distances              In/output distances
            * @param end_positions          In/output end positions of stream lines
            * @param num_integration_steps  Number of integration steps
            * @param sign                   Sign indicating forward (1) or backward (-1) integration
            * @param integration_steps      Output integration steps as a field
            */
            void update_labels(const std::vector<float>& source, std::vector<float>& labels, std::vector<float>& distances,
                std::vector<float>& end_positions, int num_integration_steps, float sign
#if __streamlines_cuda_detailed_output
                , std::vector<float>& integration_steps
#endif
            );

        private:
            /**
            * Compute stream lines and update the given labels and distances
            *
            * @param d_particles            Initial seed positions for the stream lines
            * @param num_particles          Number of seed particles
            * @param num_critical_points    Number of critical points
            * @param num_segments           Number of line segments
            * @param num_triangles          Number of triangles
            * @param num_steps              Number of integration steps
            * @param sign                   Sign indicating forward (1) or backward (-1) integration
            * @param d_labels               Output labels
            * @param d_dists                Output distances
            * @param d_terminations         Output reasons for stream line termination
            * @param integration_steps      Output integration steps as a field
            */
            void compute_streamlines(float4* d_particles, int num_particles, int num_critical_points, int num_segments, int num_triangles,
                int num_steps, float sign, short* d_labels, float* d_dists, short* d_terminations
#if __streamlines_cuda_detailed_output
                , cudaSurfaceObject_t integration_steps
#endif
            );

            void initialize_texture(void* h_data, int num_components, cudaTextureObject_t* texture, cudaArray** d_data);

            void initialize_texture(void* h_data, int num_elements, int c0, int c1, int c2, int c3, cudaTextureObject_t* texture, void** d_data);

            // Vector field resolution
            std::array<int, 2> resolution;

            // Number of convergence structures
            int num_critical_points;
            int num_line_segments;

            // Constant textures
            cudaTextureObject_t velocity_texture;
            cudaArray *d_velocity;

            cudaTextureObject_t rk4_step_texture;
            cudaArray *d_rk4_step;

            cudaTextureObject_t critical_point_ids_texture;
            float* d_critical_point_ids;
            cudaTextureObject_t critical_points_texture;
            float4* d_critical_points;

            cudaTextureObject_t segment_ids_texture;
            float *d_segment_ids;
            cudaTextureObject_t segments_texture;
            float4 *d_segments;

            cudaTextureObject_t triangleIdsTex;
            float *d_triangleIds;
            cudaTextureObject_t trianglesTex;
            float4 *d_triangles;
        };
    }
}
