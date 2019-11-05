/*
 * periodic_orbits_theisel.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mesh_data_call.h"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/math/Cuboid.h"
#include "vislib/math/Rectangle.h"

#include "glad/glad.h"

#include "Eigen/Dense"

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

#include "data/tpf_grid.h"

#include <array>
#include <functional>
#include <memory>
#include <iostream>
#include <utility>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Module for computing and visualizing the the periodic orbits of a vector field.
        *
        * @author Alexander Straub
        */
        class periodic_orbits_theisel : public core::Module {
        public:
            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static inline const char* ClassName() { return "periodic_orbits_theisel"; }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static inline const char* Description() { return "Compute and visualize periodic orbits of a 2D vector field using the method by Theisel et al."; }

            /**
             * Answers whether this module is available on the current system.
             *
             * @return 'true' if the module is available, 'false' otherwise.
             */
            static inline bool IsAvailable() { return true; }

            /**
             * Initialises a new instance.
             */
            periodic_orbits_theisel();

            /**
             * Finalises an instance.
             */
            virtual ~periodic_orbits_theisel();

        protected:
            /**
             * Implementation of 'Create'.
             *
             * @return 'true' on success, 'false' otherwise.
             */
            virtual bool create() override;

            /**
             * Implementation of 'Release'.
             */
            virtual void release() override;

        private:
            using kernel_t = CGAL::Exact_predicates_exact_constructions_kernel;

            /** Get input data and extent from called modules */
            bool get_input_data();
            bool get_input_extent();

            /**
             * Computer periodic orbits as proposed by Theisel et al.
             * in their paper "Grid-Independent Detection of Closed
             * Stream Lines in 2D Vector Fields" from 2004.
             */
            bool compute_periodic_orbits();

            /**
             * Advect given point with the selected integration method
             *
             * @param grid Vector field
             * @param point Point, which will be advected
             * @param delta Time step size, which can be adjusted by the integration method
             * @param forward True: forward integration, false: reverse integration
             *
             * @returns Advected point
             */
            Eigen::Vector3f advect_point(const tpf::data::grid<float, float, 2, 2>& grid, const Eigen::Vector3f& point, float& delta, bool forward) const;
            void advect_point_rk4(const tpf::data::grid<float, float, 2, 2>& grid, Eigen::Vector2f& point, float& delta, bool forward) const;
            void advect_point_rk45(const tpf::data::grid<float, float, 2, 2>& grid, Eigen::Vector2f& point, float& delta, bool forward) const;

            /** Callbacks for the computed periodic orbits */
            bool get_periodic_orbits_data(core::Call& call);
            bool get_periodic_orbits_extent(core::Call& call);

            /** Callbacks for the computed stream surfaces */
            bool get_stream_surfaces_data(core::Call& call);
            bool get_stream_surfaces_extent(core::Call& call);

            bool get_stream_surface_values_data(core::Call& call);
            bool get_stream_surface_values_extent(core::Call& call);

            /** Callbacks for the computed seed lines */
            bool get_seed_lines_data(core::Call& call);
            bool get_seed_lines_extent(core::Call& call);

            /** Callbacks for writing results to file */
            bool get_writer_callback(core::Call& call);
            std::function<std::ostream&()> get_writer;

            /** Output slot for found periodic orbits */
            core::CalleeSlot periodic_orbits_slot;

            /** Output slot for computed stream surfaces */
            core::CalleeSlot stream_surface_slot;
            core::CalleeSlot stream_surface_values_slot;

            /** Output slot for computed seed lines */
            core::CalleeSlot seed_line_slot;

            /** Output slot for writing results to file */
            core::CalleeSlot result_writer_slot;

            /** Input slot for getting the vector field */
            core::CallerSlot vector_field_slot;

            /** Input slot for getting the critical points */
            core::CallerSlot critical_points_slot;

            /** Transfer function for coloring stream surfaces */
            core::param::ParamSlot transfer_function;

            /** Parameters for stream surface computation */
            core::param::ParamSlot integration_method;
            core::param::ParamSlot num_integration_steps;
            core::param::ParamSlot integration_timestep;
            core::param::ParamSlot max_integration_error;

            core::param::ParamSlot num_subdivisions;
            core::param::ParamSlot critical_point_offset;

            core::param::ParamSlot direction;
            core::param::ParamSlot compute_intersections;
            core::param::ParamSlot filter_seed_lines;

            /** Bounding rectangle and box */
            vislib::math::Rectangle<float> bounding_rectangle;
            vislib::math::Cuboid<float> bounding_box;

            /** Input vector field */
            SIZE_T vector_field_hash;
            bool vector_field_changed;

            std::array<unsigned int, 2> resolution;
            std::shared_ptr<std::vector<float>> grid_positions;
            std::shared_ptr<std::vector<float>> vectors;

            /** Input critical points */
            SIZE_T critical_points_hash;
            bool critical_points_changed;

            std::shared_ptr<std::vector<float>> critical_points;

            /** Output stream surfaces */
            SIZE_T stream_surface_hash;

            std::shared_ptr<std::vector<float>> vertices;
            std::shared_ptr<std::vector<unsigned int>> triangles;
            std::shared_ptr<mesh_data_call::data_set> seed_line_ids;
            std::shared_ptr<mesh_data_call::data_set> seed_point_ids;
            std::shared_ptr<mesh_data_call::data_set> integration_ids;

            /** Output periodic orbits */
            SIZE_T periodic_orbits_hash;

            std::vector<std::pair<float, Eigen::Vector2f>> periodic_orbits;

            /** Output seed lines */
            SIZE_T seed_line_hash;

            std::vector<std::pair<float, std::vector<Eigen::Vector2f>>> seed_lines;
        };
    }
}
