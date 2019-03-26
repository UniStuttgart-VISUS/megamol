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
#define __streamlines_cuda_runge_kutta_45 1             // True: use Runge-Kutta 4,5 with adaptive step size; else: use Runge-Kutta 4 with fixed step size
#define __streamlines_cuda_runge_kutta_45_fixed 0       // True: use Runge-Kutta 4,5 only for error estimation, but keep fixed step size

// Options for detailed output (! performance hit !)
#define __streamlines_cuda_detailed_output 0            // True: output integration step size as field and stream line end points
#define __streamlines_cuda_integration_steps_max 1      // True: output integration step size field with maximum values
#define __streamlines_cuda_integration_steps_min 0      // True: output integration step size field with minimum values
#define __streamlines_cuda_integration_steps_avg 0      // True: output integration step size field with average values
#define __streamlines_cuda_integration_error_max 1      // True: output integration error field with maximum values
#define __streamlines_cuda_integration_error_min 0      // True: output integration error field with minimum values
#define __streamlines_cuda_integration_error_avg 0      // True: output integration error field with average values

#include <memory>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /** Forward definition for the actual implementation */
        class streamlines_cuda_impl;

        /**
        * Class for computation of stream lines, corresponding labels and distances on the GPU
        */
        class streamlines_cuda
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
            streamlines_cuda(const std::vector<float>& positions, const std::vector<float>& vectors, const std::vector<float>& points,
                const std::vector<int>& point_ids, const std::vector<float>& lines, const std::vector<int>& line_ids,
                float integration_timestep, float max_integration_error);

            /**
            * Destructor
            */
            ~streamlines_cuda();

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
            std::unique_ptr<streamlines_cuda_impl> impl;
        };
    }
}
