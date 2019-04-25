/*
 * implicit_topology_results.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "../cuda/streamlines.h"

#include <memory>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Struct for storing results from implicit topology computation
        *
        * @author Alexander Straub
        */
        struct implicit_topology_results
        {
            /** End positions of stream lines */
            std::shared_ptr<std::vector<float>> positions_forward;
            std::shared_ptr<std::vector<float>> positions_backward;

            /** Label and distance fields, and reasons for termination */
            std::shared_ptr<std::vector<float>> labels_forward;
            std::shared_ptr<std::vector<float>> labels_backward;

            std::shared_ptr<std::vector<float>> distances_forward;
            std::shared_ptr<std::vector<float>> distances_backward;

            std::shared_ptr<std::vector<float>> terminations_forward;
            std::shared_ptr<std::vector<float>> terminations_backward;

            /** Triangle mesh */
            std::shared_ptr<std::vector<float>> vertices;
            std::shared_ptr<std::vector<unsigned int>> indices;

            /** Computation state */
            struct state
            {
                /** Integration method */
                streamlines_cuda::integration_method method;

                /** Time step information */
                float integration_timestep;
                float max_integration_error;

                /** Number of time steps computed */
                unsigned int num_integration_steps;

                /** Indicate that the computation is finished */
                bool finished;

            } computation_state;
        };
    }
}