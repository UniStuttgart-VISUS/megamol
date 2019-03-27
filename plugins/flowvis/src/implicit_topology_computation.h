/*
 * implicit_topology_computation.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "triangulation.h"

#include <array>
#include <future>
#include <memory>
#include <thread>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Class for computing the implicit topology of a vector field.
        * This computation is performed concurrently, while allowing access to
        * intermediate results from previous computation steps.
        *
        * @author Alexander Straub
        * @author Grzegorz K. Karch
        */
        class implicit_topology_computation
        {
        public:
            /**
            * Struct storing the (intermediate) results of the computation.
            */
            struct result
            {
                /** Was this the final result? */
                bool finished;

                /** End positions of stream lines */
                std::shared_ptr<std::vector<float>> positions_forward;
                std::shared_ptr<std::vector<float>> positions_backward;

                /** Label and distance fields, and reasons for termination */
                std::shared_ptr<std::vector<float>> labels_forward;
                std::shared_ptr<std::vector<float>> distances_forward;
                std::shared_ptr<std::vector<float>> terminations_forward;

                std::shared_ptr<std::vector<float>> labels_backward;
                std::shared_ptr<std::vector<float>> distances_backward;
                std::shared_ptr<std::vector<float>> terminations_backward;

                /** Triangle mesh */
                std::shared_ptr<std::vector<float>> vertices;
                std::shared_ptr<std::vector<unsigned int>> indices;
            };

            /**
            * Initialize computation by providing seed positions and corresponding vectors, convergence structures,
            * and the initial delaunay triangulation of the domain.
            *
            * @param resolution                         Domain resolution (number of vectors per direction)
            * @param domain                             Domain size (minimum and maximum coordinates)
            * @param positions                          Positions of the vectors, also used as initial seed
            * @param vectors                            Vectors of the vector field
            * @param points                             Convergence structure points (e.g., critical points, periodic orbits, ...)
            * @param point_ids                          Unique IDs (or labels) of the given points
            * @param lines                              Convergence structure lines (e.g., domain boundaries, obstacles, ...)
            * @param line_ids                           (Unique) IDs (or labels) of the given lines
            * @param integration_timestep               (Initial) integration time step
            * @param max_integration_error              Maximum integration error
            * @param num_particles_per_batch            Number of particles processed and uploaded to the GPU per batch
            * @param num_integration_steps_per_batch    Number of integration steps per batch, after which a new (intermediate) result can be extracted
            */
            implicit_topology_computation(std::array<int, 2> resolution, std::array<float, 4> domain,
                std::vector<float> positions, std::vector<float> vectors, std::vector<float> points,
                std::vector<int> point_ids, std::vector<float> lines, std::vector<int> line_ids,
                float integration_timestep, float max_integration_error, unsigned int num_particles_per_batch,
                unsigned int num_integration_steps_per_batch);

            /**
            * Destructor
            */
            ~implicit_topology_computation();

            /**
            * Start the computation process.
            *
            * @param num_integration_steps          Number of total integration steps to perform
            * @param refinement_threshold           Threshold for refinement to prevent from refining infinitly
            * @param refine_at_labels               Refine where different labels meet?
            * @param distance_difference_threshold  Refine when distance difference between neighboring nodes exceed the threshold 
            */
            void start(unsigned int num_integration_steps, float refinement_threshold, bool refine_at_labels, float distance_difference_threshold);

            /**
            * Terminate current computation as soon as possible.
            */
            void terminate();

            /**
            * Get last (intermediate) results.
            *
            * @return Future object on (intermediate) results
            */
            std::shared_future<result> get_results();

        private:
            /**
            * Main algorithm.
            *
            * @param promise Promise containing future results
            */
            void run(std::promise<result>&& promise, unsigned int num_integration_steps, float refinement_threshold,
                bool refine_at_labels, float distance_difference_threshold);

            /** Input domain information */
            const std::array<int, 2> resolution;
            const std::array<float, 4> domain;

            /** Input seed positions and respective vectors */
            const std::vector<float> positions;
            const std::vector<float> vectors;

            /** Input convergence structures with ids (labels) */
            const std::vector<float> points;
            const std::vector<float> lines;

            const std::vector<int> point_ids;
            const std::vector<int> line_ids;

            /** Input timestep information */
            const float integration_timestep;
            const float max_integration_error;

            /** Number of particles per GPU batch */
            const unsigned int num_particles_per_batch;
            const unsigned int num_integration_steps_per_batch;

            /** Output positions */
            std::vector<float> positions_forward;
            std::vector<float> positions_backward;

            /** Output labels, distances, and reasons for termination for forward, and backward integration */
            std::vector<float> labels_forward;
            std::vector<float> distances_forward;
            std::vector<float> terminations_forward;

            std::vector<float> labels_backward;
            std::vector<float> distances_backward;
            std::vector<float> terminations_backward;

            /** Number of integration steps performed */
            unsigned int num_integration_steps_performed;

            /** Delaunay triangulation for computing a triangle mesh for refinement */
            triangulation delaunay;

            /** Computation thread */
            std::thread computation;

            /** Current results */
            std::shared_future<result> current_result;
        };
    }
}
