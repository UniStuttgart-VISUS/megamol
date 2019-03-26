/*
 * implicit_topology_computation.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "triangulation.h"

#include <future>
#include <memory>
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

                /** Label and distance fields */
                std::shared_ptr<std::vector<float>> labels;
                std::shared_ptr<std::vector<float>> distances;

                /** Triangle mesh */
                std::shared_ptr<std::vector<float>> vertices;
                std::shared_ptr<std::vector<unsigned int>> indices;
            };

            /**
            * Initialize computation by providing seed positions and corresponding vectors, convergence structures,
            * and the initial delaunay triangulation of the domain.
            *
            * @param positions  Positions of the vectors, also used as initial seed
            * @param vectors    Vectors of the vector field
            * @param points     Convergence structure points (e.g., critical points, periodic orbits, ...)
            * @param point_ids  Unique IDs (or labels) of the given points
            * @param lines      Convergence structure lines (e.g., domain boundaries, obstacles, ...)
            * @param line_ids   (Unique) IDs (or labels) of the given lines
            */
            implicit_topology_computation(std::vector<float> positions, std::vector<float> vectors, std::vector<float> points,
                std::vector<int> point_ids, std::vector<float> lines, std::vector<int> line_ids);

            /**
            * Start the computation process.
            */
            void start();

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
            /** Input seed positions and respective vectors */
            std::vector<float> positions;
            std::vector<float> vectors;

            /** Input convergence structures with ids (labels) */
            std::vector<float> points;
            std::vector<float> lines;

            std::vector<int> point_ids;
            std::vector<int> line_ids;

            /** Output labels and distances */
            std::vector<float> labels;
            std::vector<float> distances;

            /** Delaunay triangulation for computing a triangle mesh for refinement */
            triangulation delaunay;

            /** Current results */
            std::shared_future<result> current_result;

            /** State for terminating the computation */
            bool terminate_computation;
        };
    }
}
