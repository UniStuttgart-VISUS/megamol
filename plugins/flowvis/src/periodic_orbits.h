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
#include "tpf/stdext/tpf_comparator.h"
#include "tpf/utility/tpf_optional.h"

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

#include "Eigen/Dense"

#include <functional>
#include <list>
#include <mutex>
#include <ostream>
#include <set>
#include <tuple>
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
            using coords_t = typename tpf::data::grid<double, double, 2, 2>::coords_t;
            using kernel = CGAL::Exact_predicates_exact_constructions_kernel;

            /** Struct transporting the parameter sets for integration */
            struct integration_parameter_t
            {
                float sign;

                enum class method_t
                {
                    RUNGE_KUTTA_4,
                    RUNGE_KUTTA_45
                } method;

                union integration_t
                {
                    struct runge_kutta_4_t
                    {
                        unsigned int min_steps_per_cell;
                    } rk_4;

                    struct runge_kutta_45_t
                    {
                        float timestep;
                        float maximum_timestep;
                        float maximum_error;
                    } rk_45;
                } param;
            };

            /**
            * Extract periodic orbits.
            *
            * @param grid Vector field
            * @param critical_points Critical points
            * @param seed Seed of the stream line used to find the periodic orbit
            * @param sign Direction of integration
            */
            void extract_periodic_orbit(const tpf::data::grid<double, double, 2, 2>& grid,
                const std::vector<std::pair<critical_points::type, Eigen::Vector2d>>& critical_points, const Eigen::Vector2d& seed, float sign);

            /**
            * Advect using the predefined method
            *
            * @param grid Vector field
            * @param position Original position
            * @param integration_parameter Parameter for time step control
            *
            * @return The advected position
            */
            Eigen::Vector2d advect(const tpf::data::grid<double, double, 2, 2>& grid,
                const Eigen::Vector2d& position, integration_parameter_t& integration_parameter) const;

            /**
            * Advect using Runge-Kutta with fixed step size
            *
            * @param grid Vector field
            * @param position Original position
            * @param integration_parameter Parameter for time step control
            *
            * @return The advected position
            */
            Eigen::Vector2d advect_RK4(const tpf::data::grid<double, double, 2, 2>& grid,
                const Eigen::Vector2d& position, const integration_parameter_t& integration_parameter) const;

            /**
            * Advect using Runge-Kutta with dynamic step size
            *
            * @param grid Vector field
            * @param position Original position
            * @param integration_parameter Parameter for time step control
            *
            * @return The advected position
            */
            Eigen::Vector2d advect_RK45(const tpf::data::grid<double, double, 2, 2>& grid,
                const Eigen::Vector2d& position, integration_parameter_t& integration_parameter) const;

            /**
            * Find a turn, i.e., return a closed sequence of cell coordinates
            *
            * @param grid Vector field
            * @param critical_points Critical points
            * @param position Original/Output position
            * @param integration_parameter Parameter for time step control
            *
            * @return List of coordinates, defining a turn
            */
            tpf::utility::optional<std::pair<std::list<coords_t>, std::list<kernel::Point_2>>> find_turn(const tpf::data::grid<double, double, 2, 2>& grid,
                const std::vector<std::pair<critical_points::type, Eigen::Vector2d>>& critical_points, Eigen::Vector2d& position,
                integration_parameter_t& integration_parameter) const;

            /**
            * Validate a previous turn
            *
            * @param grid Vector field
            * @param position Original/Output position
            * @param integration_parameter Parameter for time step control
            * @param comparison List of cells to compare with
            *
            * @return True: valid, false otherwise
            */
            std::tuple<bool, std::vector<Eigen::Vector2d>, std::vector<kernel::Point_2>> validate_turn(const tpf::data::grid<double, double, 2, 2>& grid,
                Eigen::Vector2d& position, integration_parameter_t& integration_parameter, std::list<coords_t> comparison) const;

            /**
            * Look for exits
            *
            * @param grid Vector field
            * @param position Original/Output position
            * @param integration_parameter Parameter for time step control
            * @param comparison List of cells to compare with
            *
            * @return True: valid, false otherwise
            */
            std::pair<bool, std::vector<Eigen::Vector2d>> find_exits(const tpf::data::grid<double, double, 2, 2>& grid, Eigen::Vector2d& position,
                integration_parameter_t& integration_parameter, const std::list<coords_t>& comparison) const;

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
            std::vector<std::pair<coords_t, kernel::Point_2>> get_cells(const tpf::data::grid<double, double, 2, 2>& grid, coords_t source, const coords_t& target,
                const Eigen::Vector2d& source_position, const Eigen::Vector2d& target_position) const;

            /**
            * Check if the position is on the correct side of the stream line
            *
            * @param position Position to check
            * @param outer Outer stream line
            * @param inner Inner stream line
            *
            * @return True if inside, false otherwise
            */
            bool correct_side(const Eigen::Vector2d& position, const std::list<kernel::Point_2>& outer, const std::vector<kernel::Point_2>& inner) const;

            /**
            * Use Poincaré map for generating the orbit
            *
            * @param grid Vector field
            * @param position Original position
            * @param integration_parameter Parameter for time step control
            * @param max_poincare_error Maximum error while determining the representative orbital point
            *
            * @return Line, defined as set of points, representing the orbit
            */
            std::vector<Eigen::Vector2d> integrate_orbit(const tpf::data::grid<double, double, 2, 2>& grid, Eigen::Vector2d position,
                integration_parameter_t integration_parameter, float max_poincare_error) const;

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
            Eigen::Vector2d linear_interpolate_position(const Eigen::Vector2d& left, const Eigen::Vector2d& right, double value_left, double value_right) const;

            /** Callbacks for the triangle mesh */
            bool get_glyph_data_callback(core::Call& call);
            bool get_glyph_extent_callback(core::Call& call);

            /** Callback for the mouse event */
            bool get_mouse_coordinates_callback(core::Call& call);

            /** Callback for setting the output callback */
            bool get_output_callback(core::Call& call);
            std::function<std::ostream&()> get_output;

            /** Callbacks for user input */
            bool stop_callback(core::param::ParamSlot&);
            bool reset_callback(core::param::ParamSlot&);

            /** Output slot for the glyphs */
            core::CalleeSlot glyph_slot;
            SIZE_T glyph_hash;

            std::vector<std::pair<float, std::vector<Eigen::Vector2f>>> glyph_output;
            std::vector<std::set<coords_t, std::less<coords_t>>> orbit_cells;

            /** Output slot for receiving mouse clicks */
            core::CalleeSlot mouse_slot;

            /** Output slots for writing found periodic orbits to file */
            core::CalleeSlot file_output_slot;

            /** Input slot for getting an input vector field */
            core::CallerSlot vector_field_slot;
            SIZE_T vector_field_hash;

            /** Input slot for getting critical points, needed as seed */
            core::CallerSlot critical_points_slot;
            SIZE_T critical_points_hash;

            /** Parameter for stream line integration */
            core::param::ParamSlot integration_method;
            core::param::ParamSlot integration_direction;

            /** Parameter for integration time step control */
            core::param::ParamSlot min_steps_per_cell;
            core::param::ParamSlot initial_timestep;
            core::param::ParamSlot maximum_timestep;
            core::param::ParamSlot maximum_error;

            /** Parameter for accuracy of the Poincaré map */
            core::param::ParamSlot poincare_error;

            /** Parameter for preventing double detection */
            core::param::ParamSlot unique_detection;

            /** Parameter for debug output */
            core::param::ParamSlot output_exit_streamlines;

            /** Parameter to additionally writing input critical points to file */
            core::param::ParamSlot output_critical_points;
            bool output_critical_points_finished;

            /** Parameter for stopping and resetting the computation */
            core::param::ParamSlot stop;
            core::param::ParamSlot reset;

            /** Stored vector field */
            tpf::data::grid<double, double, 2, 2> grid;

            /** Stored critical points */
            std::vector<std::pair<critical_points::type, Eigen::Vector2d>> critical_points;

            /** Mutex for synchronization */
            std::mutex lock;
            std::size_t num_threads;

            bool terminate;
        };
    }
}