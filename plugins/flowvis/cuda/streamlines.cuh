/*
 * streamlines.cuh
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include <cuda_runtime_api.h>

#include "streamlines.h"

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
            * @param resolution                 Domain resolution (number of vectors per direction)
            * @param domain                     Domain size (minimum and maximum coordinates)
            * @param vectors                    Vectors defining the vector field to analyze
            * @param points                     Convergence structures defined as points
            * @param point_ids                  IDs (or labels) of the point convergence structures
            * @param lines                      Convergence structures defined as lines
            * @param line_ids                   IDs (or labels) of the line convergence structures
            * @param integration_timestep       Time step factor for advection
            * @param max_integration_error      Maximum error for Runge-Kutta 4-5, above which the time step size has to be adapted
            */
            streamlines_cuda_impl(const std::array<int, 2>& resolution, const std::array<float, 4>& domain,
                const std::vector<float>& vectors, const std::vector<float>& points, const std::vector<int>& point_ids,
                const std::vector<float>& lines, const std::vector<int>& line_ids, float integration_timestep,
                float max_integration_error);

            /**
            * Destructor
            */
            ~streamlines_cuda_impl();

            /**
            * Update labels for the given seed
            *
            * @param source                     In/output seed for advecting stream lines
            * @param labels                     In/output labels
            * @param distances                  In/output distances
            * @param terminations               In/output termination reasons
            * @param num_integration_steps      Number of integration steps
            * @param sign                       Sign indicating forward (1) or backward (-1) integration
            * @param integration_steps          Output integration steps as a field
            * @param num_particles_per_batch    Number of particles processed and uploaded to the GPU per batch
            */
            void update_labels(std::vector<float>& source, std::vector<float>& labels, std::vector<float>& distances,
                std::vector<float>& terminations, int num_integration_steps, float sign, unsigned int num_particles_per_batch
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
            * @param num_convergence_points Number of convergence structures represented by points
            * @param num_convergence_lines  Number of convergence structures represented by lines
            * @param num_steps              Number of integration steps
            * @param sign                   Sign indicating forward (1) or backward (-1) integration
            * @param d_labels               Output labels
            * @param d_dists                Output distances
            * @param d_terminations         Output reasons for stream line termination
            * @param integration_steps      Output integration steps as a field
            */
            void compute_streamlines(float2* d_particles, int num_particles, int num_convergence_points, int num_convergence_lines,
                int num_steps, float sign, short* d_labels, float* d_dists, short* d_terminations
#if __streamlines_cuda_detailed_output
                , cudaSurfaceObject_t integration_steps
#endif
            );

            /**
            * Initialize a higher-dimensional texture
            *
            * @param h_data         Input data to generate the texture from
            * @param num_components Number of components (=1 scaler, >1 vector)
            * @param texture        Output CUDA texture object
            * @param d_data         Output CUDA array
            */
            void initialize_texture(const void* h_data, int num_components, cudaTextureObject_t* texture, cudaArray** d_data);

            /**
            * Initialize a 1D texture
            *
            * @param h_data         Input data to generate the texture from
            * @param num_elements   Number of texture elements
            * @param c0             Number of bytes for the respective component
            * @param c1             Number of bytes for the respective component
            * @param c2             Number of bytes for the respective component
            * @param c3             Number of bytes for the respective component
            * @param texture        Output CUDA texture object
            * @param d_data         Output CUDA array
            */
            void initialize_texture(const void* h_data, int num_elements, int c0, int c1, int c2, int c3, cudaTextureObject_t* texture, void** d_data);

            // Vector field resolution
            std::array<int, 2> resolution;

            // Number of convergence structures
            int num_convergence_points;
            int num_convergence_lines;

            // Constant textures
            cudaTextureObject_t velocity_texture;
            cudaArray* d_velocity;

            cudaTextureObject_t rk4_step_texture;
            cudaArray* d_rk4_step;

            cudaTextureObject_t convergence_point_ids_texture;
            float* d_convergence_point_ids;
            cudaTextureObject_t convergence_points_texture;
            float2* d_convergence_points;

            cudaTextureObject_t convergence_line_ids_texture;
            float* d_convergence_line_ids;
            cudaTextureObject_t convergence_lines_texture;
            float2* d_convergence_lines;
        };
    }
}
