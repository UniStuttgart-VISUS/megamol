/*
 * streamlines.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

// Used algorithm
#define __streamlines_cuda_shi_et_al 0                  // True: use method by Shi et al., else our method

// Used integration method
#define __streamlines_cuda_runge_kutta_45 0             // True: use Runge-Kutta 4,5 with adaptive step size; else: use Runge-Kutta 4 with fixed step size

#include <array>
#include <memory>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Class for computation of stream lines, corresponding labels and distances on the GPU
        */
        class streamlines_cuda
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
            streamlines_cuda(const std::array<unsigned int, 2>& resolution, const std::array<float, 4>& domain,
                const std::vector<float>& vectors, const std::vector<float>& points, const std::vector<int>& point_ids,
                const std::vector<float>& lines, const std::vector<int>& line_ids, float integration_timestep,
                float max_integration_error);

            /**
            * Update labels for the given seed
            *
            * @param source                     In/output seed for advecting stream lines
            * @param labels                     In/output labels
            * @param distances                  In/output distances
            * @param terminations               In/output termination reasons
            * @param num_integration_steps      Number of integration steps
            * @param sign                       Sign indicating forward (1) or backward (-1) integration
            * @param num_particles_per_batch    Number of particles processed and uploaded to the GPU per batch
            */
            void update_labels(std::vector<float>& source, std::vector<float>& labels, std::vector<float>& distances,
                std::vector<float>& terminations, int num_integration_steps, float sign, unsigned int num_particles_per_batch);
        };
    }
}
