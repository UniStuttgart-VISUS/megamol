/*
 * periodic_orbits.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "critical_points.h"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "tpf/data/tpf_grid.h"
#include "tpf/utility/tpf_optional.h"

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

#include "Eigen/Dense"

#include <list>
#include <mutex>
#include <utility>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Module for computing periodic orbits of a vector field.
        *
        * @author Alexander Straub
        */
        class periodic_orbits : public core::Module
        {
        public:
            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static inline const char* ClassName() { return "periodic_orbits"; }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static inline const char* Description() { return "Compute periodic orbits of a 2D vector field"; }

            /**
             * Answers whether this module is available on the current system.
             *
             * @return 'true' if the module is available, 'false' otherwise.
             */
            static inline bool IsAvailable() { return true; }

            /**
             * Initialises a new instance.
             */
            periodic_orbits();

            /**
             * Finalises an instance.
             */
            virtual ~periodic_orbits();

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
            using coords_t = typename tpf::data::grid<float, float, 2, 2>::coords_t;
            using kernel = CGAL::Exact_predicates_exact_constructions_kernel;

            /**
            * Extract periodic orbits.
            *
            * @param grid Vector field
            * @param critical_points Critical points
            * @param seed Seed of the stream line used to find the periodic orbit
            * @param sign Direction of integration
            */
            void extract_periodic_orbit(const tpf::data::grid<float, float, 2, 2>& grid,
                const std::vector<std::pair<critical_points::type, Eigen::Vector2f>>& critical_points, const Eigen::Vector2f& seed, float sign);

            /**
            * Advect using Runge-Kutta with dynamic step size
            *
            * @param grid Vector field
            * @param position Original position
            * @param delta Previous step size
            * @param sign Direction of integration
            * @param max_error Maximum error for step size computation
            * @param max_delta Maximum step size
            *
            * @return The advected position and adjusted step size
            */
            std::pair<Eigen::Vector2f, float> advect_RK45(const tpf::data::grid<float, float, 2, 2>& grid,
                const Eigen::Vector2f& position, float delta, float sign, float max_error, float max_delta) const;

            /**
            * Find a turn, i.e., return a closed sequence of cell coordinates
            *
            * @param grid Vector field
            * @param critical_points Critical points
            * @param position Original/Output position
            * @param delta Previous/Adjusted step size
            * @param sign Direction of integration
            * @param max_error Maximum error for step size computation
            * @param max_delta Maximum step size
            *
            * @return List of coordinates, defining a turn
            */
            tpf::utility::optional<std::list<coords_t>> find_turn(const tpf::data::grid<float, float, 2, 2>& grid,
                const std::vector<std::pair<critical_points::type, Eigen::Vector2f>>& critical_points, Eigen::Vector2f& position,
                float& delta, float sign, float max_error, float max_delta) const;

            /**
            * Validate a previous turn
            *
            * @param grid Vector field
            * @param position Original/Output position
            * @param delta Previous/Adjusted step size
            * @param sign Direction of integration
            * @param max_error Maximum error for step size computation
            * @param max_delta Maximum step size
            * @param comparison List of cells to compare with
            *
            * @return True: valid, false otherwise
            */
            std::pair<bool, std::vector<Eigen::Vector2f>> validate_turn_strict(const tpf::data::grid<float, float, 2, 2>& grid, Eigen::Vector2f& position,
                float& delta, float sign, float max_error, float max_delta, std::list<coords_t> comparison) const;

            /**
            * Validate a previous turn
            *
            * @param grid Vector field
            * @param position Original/Output position
            * @param delta Previous/Adjusted step size
            * @param sign Direction of integration
            * @param max_error Maximum error for step size computation
            * @param max_delta Maximum step size
            * @param comparison List of cells to compare with
            *
            * @return True: valid, false otherwise
            */
            std::pair<bool, std::vector<Eigen::Vector2f>> validate_turn_relaxed(const tpf::data::grid<float, float, 2, 2>& grid, Eigen::Vector2f& position,
                float& delta, float sign, float max_error, float max_delta, const std::list<coords_t>& comparison) const;

            /**
            * Get intermediate cells
            *
            * @param grid Vector field
            * @param source Source cell
            * @param target Target cell
            * @param source_position Source position
            * @param target_position Target position
            *
            * @return Intermediate cells
            */
            std::vector<coords_t> get_cells(const tpf::data::grid<float, float, 2, 2>& grid, coords_t source, const coords_t& target,
                const Eigen::Vector2f& source_position, const Eigen::Vector2f& target_position) const;

            /**
            * Use Poincaré map for generating the orbit
            *
            * @param grid Vector field
            * @param position Original position
            * @param delta Previous/Adjusted step size
            * @param sign Direction of integration
            * @param max_error Maximum error for step size computation
            * @param max_delta Maximum step size
            *
            * @return Line, defined as set of points, representing the orbit
            */
            std::vector<Eigen::Vector2f> integrate_orbit(const tpf::data::grid<float, float, 2, 2>& grid, Eigen::Vector2f position,
                float delta, float sign, float max_error, float max_delta) const;

            /**
            * Linear interpolate position based on value
            *
            * @param left "Left" position
            * @param right "Right" position
            * @param value_left Value at "left" position
            * @param value_right Value at "right" position
            *
            * @return Position at which the value is zero
            */
            Eigen::Vector2f linear_interpolate_position(const Eigen::Vector2f& left, const Eigen::Vector2f& right, float value_left, float value_right) const;

            /** Callbacks for the triangle mesh */
            bool get_glyph_data_callback(core::Call& call);
            bool get_glyph_extent_callback(core::Call& call);

            /** Callback for the mouse event */
            bool get_mouse_coordinates_callback(core::Call& call);

            /** Output slot for the glyphs */
            core::CalleeSlot glyph_slot;
            SIZE_T glyph_hash;
            std::vector<std::pair<float, std::vector<Eigen::Vector2f>>> glyph_output;

            /** Output slot for receiving mouse clicks */
            core::CalleeSlot mouse_slot;

            /** Input slot for getting an input vector field */
            core::CallerSlot vector_field_slot;
            SIZE_T vector_field_hash;

            /** Input slot for getting critical points, needed as seed */
            core::CallerSlot critical_points_slot;
            SIZE_T critical_points_hash;

            /** Parameter for stream line integration */
            core::param::ParamSlot initial_timestep;
            core::param::ParamSlot maximum_timestep;
            core::param::ParamSlot maximum_error;

            /** Parameter for accuracy of the Poincaré map */
            core::param::ParamSlot poincare_iterations;

            /** Parameter for debug output */
            core::param::ParamSlot output_exit_streamlines;

            /** Stored vector field */
            tpf::data::grid<float, float, 2, 2> grid;

            /** Stored critical points */
            std::vector<std::pair<critical_points::type, Eigen::Vector2f>> critical_points;

            /** Mutex for synchronization */
            std::mutex lock;
            std::size_t num_threads;
        };
    }
}