#include "stdafx.h"
#include "periodic_orbits.h"

#include "critical_points.h"
#include "glyph_data_call.h"
#include "mouse_click_call.h"
#include "vector_field_call.h"

#include "mmcore/Call.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/DataHash.h"

#include "vislib/math/Rectangle.h"
#include "vislib/sys/Log.h"

#include "tpf/data/tpf_grid.h"
#include "tpf/math/tpf_vector.h"
#include "tpf/stdext/tpf_hash.h"
#include "tpf/utility/tpf_optional.h"

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/intersection_2.h>

#include "Eigen/Dense"

#include <algorithm>
#include <array>
#include <cmath>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        periodic_orbits::periodic_orbits() :
            glyph_slot("set_glyphs", "Glyph output"),
            glyph_hash(-1),
            mouse_slot("mouse_slot", "Receive mouse coordinates on click"),
            vector_field_slot("get_vector_field", "Vector field input"),
            vector_field_hash(-1),
            critical_points_slot("get_critical_points", "Critical points input"),
            critical_points_hash(-1),
            integration_direction("integration_direction", "Direction of integration"),
            initial_timestep("initial_timestep", "Initial time step for stream line integration"),
            maximum_timestep("maximum_timestep", "Maximum time step for stream line integration"),
            maximum_error("maximum_error", "Maximum error of the time step for stream line integration"),
            poincare_iterations("poincare_iterations", "Number of iterations for the final periodic orbit"),
            output_exit_streamlines("output_exit_streamlines", "Output stream lines from exit search"),
            stop("stop", "Stop the currently running integration processes"),
            reset("reset", "Reset and clear all previous results"),
            num_threads(0), terminate(false)
        {
            // Connect output
            this->glyph_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(0), &periodic_orbits::get_glyph_data_callback);
            this->glyph_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(1), &periodic_orbits::get_glyph_extent_callback);
            this->MakeSlotAvailable(&this->glyph_slot);

            this->mouse_slot.SetCallback(mouse_click_call::ClassName(), mouse_click_call::FunctionName(0), &periodic_orbits::get_mouse_coordinates_callback);
            this->MakeSlotAvailable(&this->mouse_slot);

            // Connect input
            this->vector_field_slot.SetCompatibleCall<vector_field_call::vector_field_description>();
            this->MakeSlotAvailable(&this->vector_field_slot);

            this->critical_points_slot.SetCompatibleCall<glyph_data_call::glyph_data_description>();
            this->MakeSlotAvailable(&this->critical_points_slot);

            // Set parameters
            this->integration_direction << new core::param::EnumParam(0);
            this->integration_direction.Param<core::param::EnumParam>()->SetTypePair(0, "Forward and backward");
            this->integration_direction.Param<core::param::EnumParam>()->SetTypePair(1, "Forward");
            this->integration_direction.Param<core::param::EnumParam>()->SetTypePair(2, "Backward");
            this->MakeSlotAvailable(&this->integration_direction);

            this->initial_timestep << new core::param::FloatParam(0.01f);
            this->MakeSlotAvailable(&this->initial_timestep);

            this->maximum_timestep << new core::param::FloatParam(0.01f);
            this->MakeSlotAvailable(&this->maximum_timestep);

            this->maximum_error << new core::param::FloatParam(0.00001f);
            this->MakeSlotAvailable(&this->maximum_error);

            this->poincare_iterations << new core::param::IntParam(50);
            this->MakeSlotAvailable(&this->poincare_iterations);

            this->output_exit_streamlines << new core::param::BoolParam(false);
            this->MakeSlotAvailable(&output_exit_streamlines);

            this->stop << new core::param::ButtonParam();
            this->stop.SetUpdateCallback(&periodic_orbits::stop_callback);
            this->MakeSlotAvailable(&this->stop);

            this->reset << new core::param::ButtonParam();
            this->reset.SetUpdateCallback(&periodic_orbits::reset_callback);
            this->MakeSlotAvailable(&this->reset);
        }

        periodic_orbits::~periodic_orbits()
        {
            this->Release();
        }

        bool periodic_orbits::create()
        {
            return true;
        }

        void periodic_orbits::release()
        {
        }

        bool periodic_orbits::get_glyph_data_callback(core::Call& call)
        {
            auto* glyph_call = dynamic_cast<glyph_data_call*>(&call);

            if (glyph_call != nullptr)
            {
                // Get vector field and critical points
                auto* get_vector_field = this->vector_field_slot.CallAs<vector_field_call>();
                auto* get_critical_points = this->critical_points_slot.CallAs<glyph_data_call>();

                const bool has_vector_field = get_vector_field != nullptr && (*get_vector_field)(0);
                const bool has_critical_points = get_critical_points != nullptr && (*get_critical_points)(0);
                const bool has_different_input = (get_vector_field->DataHash() != this->vector_field_hash ||
                    get_critical_points->DataHash() != this->critical_points_hash);

                if (has_vector_field && has_critical_points && has_different_input)
                {
                    this->vector_field_hash = get_vector_field->DataHash();
                    this->critical_points_hash = get_critical_points->DataHash();

                    const auto& vectors = *get_vector_field->get_vectors();

                    const auto& critical_points_vertices = *get_critical_points->get_point_vertices();
                    const auto& critical_points_indices = *get_critical_points->get_point_indices();
                    const auto& critical_points_values = *get_critical_points->get_point_values();

                    // Create grid from input vector field
                    tpf::data::extent_t extent(2);
                    extent[0].first = extent[1].first = 0;
                    extent[0].second = get_vector_field->get_resolution()[0] - 1;
                    extent[1].second = get_vector_field->get_resolution()[1] - 1;

                    tpf::data::grid_information<float>::array_type cell_sizes(2);
                    cell_sizes[0].resize(get_vector_field->get_resolution()[0]);
                    cell_sizes[1].resize(get_vector_field->get_resolution()[1]);

                    std::fill(cell_sizes[0].begin(), cell_sizes[0].end(),
                        (get_vector_field->get_bounding_rectangle().Right() - get_vector_field->get_bounding_rectangle().Left()) / (get_vector_field->get_resolution()[0] - 1));
                    std::fill(cell_sizes[1].begin(), cell_sizes[1].end(),
                        (get_vector_field->get_bounding_rectangle().Top() - get_vector_field->get_bounding_rectangle().Bottom()) / (get_vector_field->get_resolution()[1] - 1));

                    tpf::data::grid_information<float>::array_type cell_coordinates(2);
                    tpf::data::grid_information<float>::array_type node_coordinates(2);

                    const std::array<float, 2> offset{ get_vector_field->get_bounding_rectangle().Left(), get_vector_field->get_bounding_rectangle().Bottom() };

                    for (std::size_t d = 0; d < 2; ++d)
                    {
                        cell_coordinates[d].resize(cell_sizes[d].size());
                        node_coordinates[d].resize(cell_sizes[d].size() + 1);

                        cell_coordinates[d][0] = offset[d];
                        node_coordinates[d][0] = offset[d] - 0.5f * cell_sizes[d][0];

                        for (std::size_t i = 1; i < cell_sizes[d].size(); ++i)
                        {
                            cell_coordinates[d][i] = cell_coordinates[d][i - 1] + 0.5f * (cell_sizes[d][i - 1] + cell_sizes[d][i]);
                            node_coordinates[d][i] = node_coordinates[d][i - 1] + cell_sizes[d][i - 1];
                        }

                        node_coordinates[d][cell_sizes[d].size()] = node_coordinates[d][cell_sizes[d].size() - 1] + cell_sizes[d][cell_sizes[d].size() - 1];
                    }

                    this->grid = tpf::data::grid<float, float, 2, 2>("vector_field", extent, vectors,
                        std::move(cell_coordinates), std::move(node_coordinates), std::move(cell_sizes));

                    // Reset output
                    this->glyph_output.clear();
                    this->glyph_hash = -1;

                    // Store critical points
                    this->critical_points.reserve(critical_points_indices.size());

                    for (auto index : critical_points_indices)
                    {
                        this->critical_points.push_back(std::make_pair(static_cast<flowvis::critical_points::type>(static_cast<int>(critical_points_values[index])),
                            Eigen::Vector2f(critical_points_vertices[index * 2 + 0], critical_points_vertices[index * 2 + 1])));
                    }
                }

                // Fill glyph call
                if (glyph_call->DataHash() != this->glyph_hash)
                {
                    glyph_call->clear();

                    for (std::size_t i = 0; i < this->glyph_output.size(); ++i)
                    {
                        glyph_call->add_line(this->glyph_output[i].second, this->glyph_output[i].first);
                    }

                    glyph_call->SetDataHash(this->glyph_hash);
                }
            }

            return true;
        }

        bool periodic_orbits::get_glyph_extent_callback(core::Call& call)
        {
            auto* get_vector_field = this->vector_field_slot.CallAs<vector_field_call>();
            auto* get_critical_points = this->critical_points_slot.CallAs<glyph_data_call>();

            return get_vector_field != nullptr && get_critical_points != nullptr && (*get_vector_field)(1) && (*get_critical_points)(1);
        }

        bool periodic_orbits::get_mouse_coordinates_callback(core::Call& call)
        {
            auto* get_mouse_coordinates = dynamic_cast<mouse_click_call*>(&call);

            if (get_mouse_coordinates != nullptr && this->grid.get_num_elements() > 0)
            {
                const Eigen::Vector2f seed(get_mouse_coordinates->get_coordinates().first, get_mouse_coordinates->get_coordinates().second);

                // Seed a stream lines
                const auto direction = this->integration_direction.Param<core::param::EnumParam>()->Value();

                if (direction == 0 || direction == 1)
                {
                    std::thread(&periodic_orbits::extract_periodic_orbit, this, this->grid, this->critical_points, seed, 1.0f).detach();
                }

                if (direction == 0 || direction == 2)
                {
                    std::thread(&periodic_orbits::extract_periodic_orbit, this, this->grid, this->critical_points, seed, -1.0f).detach();
                }
            }

            return true;
        }

        bool periodic_orbits::stop_callback(core::param::ParamSlot&)
        {
            std::lock_guard<std::mutex> locker(this->lock);

            this->terminate = true;

            return true;
        }

        bool periodic_orbits::reset_callback(core::param::ParamSlot&)
        {
            std::lock_guard<std::mutex> locker(this->lock);

            this->glyph_output.clear();
            ++this->glyph_hash;

            return true;
        }

        void periodic_orbits::extract_periodic_orbit(const tpf::data::grid<float, float, 2, 2>& grid,
            const std::vector<std::pair<flowvis::critical_points::type, Eigen::Vector2f>>& critical_points, const Eigen::Vector2f& seed, const float sign)
        {
            // Get parameters for stream line integration
            float delta;
            float max_delta;
            float max_error;

            bool output_exit_streamline;

            {
                std::lock_guard<std::mutex> locker(this->lock);

                this->terminate = false;

                vislib::sys::Log::DefaultLog.WriteInfo("Number of processes running: %d", ++this->num_threads);

                vislib::sys::Log::DefaultLog.WriteInfo("Starting %s stream line at [%.5f, %.5f]", (sign < 0.0f) ? "backward" : "forward", seed[0], seed[1]);

                delta = this->initial_timestep.Param<core::param::FloatParam>()->Value();
                max_delta = this->maximum_timestep.Param<core::param::FloatParam>()->Value();
                max_error = this->maximum_error.Param<core::param::FloatParam>()->Value();

                output_exit_streamline = this->output_exit_streamlines.Param<core::param::BoolParam>()->Value();
            }

            // Start
            Eigen::Vector2f position = seed;

            bool has_exit = true;

            while (has_exit && !this->terminate)
            {
                // Find turn
                const auto visited_cells = find_turn(grid, critical_points, position, delta, sign, max_error, max_delta);

                if (visited_cells)
                {
                    if (!visited_cells->empty())
                    {
                        // Do a second turn and compare results
                        if (validate_turn_strict(grid, position, delta, sign, max_error, max_delta, *visited_cells).first)
                        {
                            // Look for possible exits
                            auto vector_hash = [](const Eigen::Vector2f& coords) -> std::size_t
                            {
                                return core::utility::DataHash(coords[0], coords[1]);
                            };

                            std::unordered_set<Eigen::Vector2f, decltype(vector_hash)> possible_exits(23, vector_hash);

                            for (const auto& cell : *visited_cells)
                            {
                                const auto corner_bl = grid.get_cell_coordinates(cell);
                                const auto corner_br = grid.get_cell_coordinates(cell + coords_t(1, 0));
                                const auto corner_tl = grid.get_cell_coordinates(cell + coords_t(0, 1));
                                const auto corner_tr = grid.get_cell_coordinates(cell + coords_t(1, 1));

                                const auto value_bl = grid(cell);
                                const auto value_br = grid(cell + coords_t(1, 0));
                                const auto value_tl = grid(cell + coords_t(0, 1));
                                const auto value_tr = grid(cell + coords_t(1, 1));

                                if (std::signbit(value_bl[1]) != std::signbit(value_br[1]))
                                {
                                    possible_exits.insert(linear_interpolate_position(corner_bl, corner_br, value_bl[1], value_br[1]));
                                }
                                if (std::signbit(value_tl[1]) != std::signbit(value_tr[1]))
                                {
                                    possible_exits.insert(linear_interpolate_position(corner_tl, corner_tr, value_tl[1], value_tr[1]));
                                }
                                if (std::signbit(value_bl[0]) != std::signbit(value_tl[0]))
                                {
                                    possible_exits.insert(linear_interpolate_position(corner_bl, corner_tl, value_bl[0], value_tl[0]));
                                }
                                if (std::signbit(value_br[0]) != std::signbit(value_tr[0]))
                                {
                                    possible_exits.insert(linear_interpolate_position(corner_br, corner_tr, value_br[0], value_tr[0]));
                                }

                                possible_exits.insert(corner_bl);
                                possible_exits.insert(corner_br);
                                possible_exits.insert(corner_tl);
                                possible_exits.insert(corner_tr);
                            }

                            // Backward integration from possible exits
                            has_exit = false;

                            std::vector<std::vector<Eigen::Vector2f>> exit_tries;

                            for (auto possible_exit_it = possible_exits.cbegin(); possible_exit_it != possible_exits.cend() && !has_exit; ++possible_exit_it)
                            {
                                Eigen::Vector2f possible_exit = *possible_exit_it;

                                const auto exit = validate_turn_relaxed(grid, possible_exit, delta, -sign, max_error, max_delta, *visited_cells);
                                has_exit = exit.first;

                                if (output_exit_streamline)
                                {
                                    exit_tries.push_back(exit.second);
                                }
                            }

                            if (output_exit_streamline && !has_exit)
                            {
                                std::lock_guard<std::mutex> locker(this->lock);

                                for (const auto& exit_try : exit_tries)
                                {
                                    this->glyph_output.push_back(std::make_pair(0.5f * sign, exit_try));
                                }

                                for (const auto& cell : *visited_cells)
                                {
                                    const auto corner_bl = grid.get_cell_coordinates(cell);
                                    const auto corner_br = grid.get_cell_coordinates(cell + coords_t(1, 0));
                                    const auto corner_tl = grid.get_cell_coordinates(cell + coords_t(0, 1));
                                    const auto corner_tr = grid.get_cell_coordinates(cell + coords_t(1, 1));

                                    std::vector<Eigen::Vector2f> box { corner_bl, corner_br, corner_tr, corner_tl, corner_bl };

                                    this->glyph_output.push_back(std::make_pair(0.0f, box));
                                }

                                ++this->glyph_hash;
                            }
                        }
                    }
                }
                else
                {
                    std::lock_guard<std::mutex> locker(this->lock);

                    vislib::sys::Log::DefaultLog.WriteWarn("Could not find a periodic orbit from %s stream line at [%.5f, %.5f]",
                        (sign < 0.0f) ? "backward" : "forward", seed[0], seed[1]);

                    vislib::sys::Log::DefaultLog.WriteInfo("Number of processes running: %d", --this->num_threads);

                    return;
                }
            }

            if (!this->terminate)
            {
                // Use Poincaré map for finding the closed stream line
                const std::vector<Eigen::Vector2f> periodic_orbit = integrate_orbit(grid, position, delta, sign, max_error, max_delta);

                // Output results
                std::lock_guard<std::mutex> locker(this->lock);

                this->glyph_output.push_back(std::make_pair(sign, periodic_orbit));

                for (const auto& point : periodic_orbit)
                {
                    this->glyph_hash = static_cast<SIZE_T>(core::utility::DataHash(this->glyph_hash, point[0], point[1]));
                }

                vislib::sys::Log::DefaultLog.WriteInfo("Periodic orbit found at [%.5f, %.5f] from %s stream line at [%.5f, %.5f]",
                    periodic_orbit[0][0], periodic_orbit[0][1], (sign < 0.0f) ? "backward" : "forward", seed[0], seed[1]);

                vislib::sys::Log::DefaultLog.WriteInfo("Number of processes running: %d", --this->num_threads);
            }
            else
            {
                std::lock_guard<std::mutex> locker(this->lock);

                vislib::sys::Log::DefaultLog.WriteInfo("Search for periodic orbit from %s stream line at [%.5f, %.5f] terminated",
                    (sign < 0.0f) ? "backward" : "forward", seed[0], seed[1]);

                vislib::sys::Log::DefaultLog.WriteInfo("Number of processes running: %d", --this->num_threads);
            }
        }

        std::pair<Eigen::Vector2f, float> periodic_orbits::advect_RK45(const tpf::data::grid<float, float, 2, 2>& grid,
            const Eigen::Vector2f& position, float delta, const float sign, const float max_error, const float max_delta) const
        {
            // Cash-Karp parameters
            constexpr float b_21 = 0.2f;
            constexpr float b_31 = 0.075f;
            constexpr float b_41 = 0.3f;
            constexpr float b_51 = -11.0f / 54.0f;
            constexpr float b_61 = 1631.0f / 55296.0f;
            constexpr float b_32 = 0.225f;
            constexpr float b_42 = -0.9f;
            constexpr float b_52 = 2.5f;
            constexpr float b_62 = 175.0f / 512.0f;
            constexpr float b_43 = 1.2f;
            constexpr float b_53 = -70.0f / 27.0f;
            constexpr float b_63 = 575.0f / 13824.0f;
            constexpr float b_54 = 35.0f / 27.0f;
            constexpr float b_64 = 44275.0f / 110592.0f;
            constexpr float b_65 = 253.0f / 4096.0f;

            constexpr float c_1 = 37.0f / 378.0f;
            constexpr float c_2 = 0.0f;
            constexpr float c_3 = 250.0f / 621.0f;
            constexpr float c_4 = 125.0f / 594.0f;
            constexpr float c_5 = 0.0f;
            constexpr float c_6 = 512.0f / 1771.0f;

            constexpr float c_1s = 2825.0f / 27648.0f;
            constexpr float c_2s = 0.0f;
            constexpr float c_3s = 18575.0f / 48384.0f;
            constexpr float c_4s = 13525.0f / 55296.0f;
            constexpr float c_5s = 277.0f / 14336.0f;
            constexpr float c_6s = 0.25f;

            // Constants
            constexpr float grow_exponent = -0.2f;
            constexpr float shrink_exponent = -0.25f;
            constexpr float max_growth = 5.0f;
            constexpr float max_shrink = 0.1f;
            constexpr float safety = 0.9f;

            // Calculate Runge-Kutta coefficients
            Eigen::Vector2f output_position;

            bool decreased = false;

            do
            {
                const Eigen::Vector2f k1 = delta * sign * grid.interpolate(position);
                const Eigen::Vector2f k2 = delta * sign * grid.interpolate(position + b_21 * k1);
                const Eigen::Vector2f k3 = delta * sign * grid.interpolate(position + b_31 * k1 + b_32 * k2);
                const Eigen::Vector2f k4 = delta * sign * grid.interpolate(position + b_41 * k1 + b_42 * k2 + b_43 * k3);
                const Eigen::Vector2f k5 = delta * sign * grid.interpolate(position + b_51 * k1 + b_52 * k2 + b_53 * k3 + b_54 * k4);
                const Eigen::Vector2f k6 = delta * sign * grid.interpolate(position + b_61 * k1 + b_62 * k2 + b_63 * k3 + b_64 * k4 + b_65 * k5);

                // Calculate error estimate
                const Eigen::Vector2f fifth_order = position + c_1 * k1 + c_2 * k2 + c_3 * k3 + c_4 * k4 + c_5 * k5 + c_6 * k6;
                const Eigen::Vector2f fourth_order = position + c_1s * k1 + c_2s * k2 + c_3s * k3 + c_4s * k4 + c_5s * k5 + c_6s * k6;

                const Eigen::Vector2f difference = fifth_order - fourth_order;

                const Eigen::Vector2f scale = grid.interpolate(position);

                const float error = std::max(0.0f, std::max(std::abs(difference[0] / scale[0]), std::abs(difference[1] / scale[1]))) / max_error;

                // Set new, adapted time step
                if (error > 1.0f)
                {
                    // Error too large, reduce time step
                    delta *= std::max(max_shrink, safety * std::pow(error, shrink_exponent));
                    decreased = true;
                }
                else
                {
                    // Error (too) small, increase time step
                    delta = std::min(max_delta, delta * std::min(max_growth, safety * std::pow(error, grow_exponent)));
                    decreased = false;
                }

                // Set output
                output_position = fifth_order;
            } while (decreased);

            return std::make_pair(output_position, delta);
        }

        tpf::utility::optional<std::list<typename tpf::data::grid<float, float, 2, 2>::coords_t>> periodic_orbits::find_turn(const tpf::data::grid<float, float, 2, 2>& grid,
            const std::vector<std::pair<flowvis::critical_points::type, Eigen::Vector2f>>& critical_points, Eigen::Vector2f& position,
            float& delta, const float sign, const float max_error, const float max_delta) const
        {
            using coords_t = typename tpf::data::grid<float, float, 2, 2>::coords_t;

            // Initialize cell list
            std::list<coords_t> visited_cells;

            visited_cells.push_back(*grid.find_staggered_cell(position));

            // Advect stream line until a closed turn is detected
            std::size_t num_critical_point_visits = 0;
            std::size_t num_cells_visited = 0;

            bool found_turn = false;

            while (!found_turn)
            {
                const auto advected = advect_RK45(grid, position, delta, sign, max_error, max_delta);

                const Eigen::Vector2f old_position = position;

                position = advected.first;
                delta = advected.second;

                if (position == old_position)
                {
                    // Advection had no result; stream line stopped
                    return tpf::utility::nullopt;
                }

                // Check entry into new cell
                const auto old_cell = *grid.find_staggered_cell(old_position);
                const auto new_cell = grid.find_staggered_cell(position);

                if (!new_cell)
                {
                    // Advection went out-of-bounds
                    return tpf::utility::nullopt;
                }

                if (old_cell != *new_cell)
                {
                    // Add cells that were overstepped
                    auto current_cells = get_cells(grid, old_cell, *new_cell, old_position, position);

                    current_cells.push_back(*new_cell);

                    for (auto current_cell_it = current_cells.begin(); current_cell_it != current_cells.end() && !found_turn; ++current_cell_it)
                    {
                        const auto& current_cell = *current_cell_it;

                        ++num_cells_visited;

                        // Check for existance of critical point within the new cell
                        bool has_critical_point = false;

                        for (const auto& point : critical_points)
                        {
                            const bool attracting =
                                (sign < 0.0f && (point.first == flowvis::critical_points::type::REPELLING_FOCUS || point.first == flowvis::critical_points::type::REPELLING_NODE)) ||
                                (sign > 0.0f && (point.first == flowvis::critical_points::type::ATTRACTING_FOCUS || point.first == flowvis::critical_points::type::ATTRACTING_NODE));

                            if (attracting && *grid.find_staggered_cell(point.second) == current_cell)
                            {
                                has_critical_point = true;
                            }
                        }

                        if (!has_critical_point)
                        {
                            // Cell has changed: check for prior visitation
                            if (std::find(visited_cells.begin(), visited_cells.end(), current_cell) != visited_cells.end())
                            {
                                found_turn = true;

                                // Remove all cells which do not belong to this turn
                                for (auto cell_it = visited_cells.begin(); cell_it != visited_cells.end() && *cell_it != current_cell; )
                                {
                                    visited_cells.erase(cell_it++);
                                }
                            }
                            else
                            {
                                visited_cells.push_back(current_cell);
                            }
                        }
                        else
                        {
                            ++num_critical_point_visits;

                            visited_cells.clear();
                        }

                        // Return, if stuck in a loop with a critical point
                        if (num_critical_point_visits == 1000)
                        {
                            return tpf::utility::nullopt;
                        }
                    }
                }
            }

            // Return results
            return visited_cells;
        }

        std::pair<bool, std::vector<Eigen::Vector2f>> periodic_orbits::validate_turn_strict(const tpf::data::grid<float, float, 2, 2>& grid, Eigen::Vector2f& position,
            float& delta, const float sign, const float max_error, const float max_delta, std::list<coords_t> comparison) const
        {
            // Initialize comparison
            if (*grid.find_staggered_cell(position) != comparison.front())
            {
                return std::make_pair(false, std::vector<Eigen::Vector2f>());
            }

            std::vector<Eigen::Vector2f> streamline;
            streamline.push_back(position);

            comparison.pop_front();

            // Advect stream line while it corresponds to the input list of cells
            while (!comparison.empty())
            {
                const auto advected = advect_RK45(grid, position, delta, sign, max_error, max_delta);

                const Eigen::Vector2f old_position = position;

                position = advected.first;
                delta = advected.second;

                if (position == old_position)
                {
                    // Advection had no result; stream line stopped
                    return std::make_pair(false, streamline);
                }

                // Check entry into new cell
                const auto old_cell = *grid.find_staggered_cell(old_position);
                const auto new_cell = grid.find_staggered_cell(position);

                if (!new_cell)
                {
                    // Advection went out-of-bounds
                    return std::make_pair(false, streamline);
                }

                streamline.push_back(position);

                if (old_cell != *new_cell)
                {
                    // Add cells that were overstepped
                    auto current_cells = get_cells(grid, old_cell, *new_cell, old_position, position);

                    current_cells.push_back(*new_cell);

                    // Check cells
                    for (auto cell_it = current_cells.begin(); cell_it != current_cells.end() && !comparison.empty(); ++cell_it)
                    {
                        const auto& current_cell = *cell_it;

                        if (current_cell != comparison.front())
                        {
                            return std::make_pair(false, streamline);
                        }

                        comparison.remove(current_cell);
                    }
                }
            }

            // Return results
            return std::make_pair(true, streamline);
        }

        std::pair<bool, std::vector<Eigen::Vector2f>> periodic_orbits::validate_turn_relaxed(const tpf::data::grid<float, float, 2, 2>& grid, Eigen::Vector2f& position,
            float& delta, const float sign, const float max_error, const float max_delta, const std::list<coords_t>& comparison) const
        {
            // Initialize comparison
            std::vector<Eigen::Vector2f> streamline;
            streamline.push_back(position);

            std::unordered_set<coords_t, std::hash<coords_t>> visited_cells;

            // Advect one step to ensure the start is not considered outside
            const auto advected = advect_RK45(grid, position, delta, sign, max_error, max_delta);

            position = advected.first;
            delta = advected.second;

            streamline.push_back(position);

            // Advect stream line while it corresponds to the input list of cells
            while (true)
            {
                const auto advected = advect_RK45(grid, position, delta, sign, max_error, max_delta);

                const Eigen::Vector2f old_position = position;

                position = advected.first;
                delta = advected.second;

                if (position == old_position)
                {
                    // Advection had no result; stream line stopped
                    return std::make_pair(false, streamline);
                }

                // Check entry into new cell
                const auto old_cell = *grid.find_staggered_cell(old_position);
                const auto new_cell = grid.find_staggered_cell(position);

                if (!new_cell)
                {
                    // Advection went out-of-bounds
                    return std::make_pair(false, streamline);
                }

                streamline.push_back(position);

                if (old_cell != *new_cell)
                {
                    // Add cells that were overstepped
                    auto current_cells = get_cells(grid, old_cell, *new_cell, old_position, position);

                    current_cells.push_back(*new_cell);

                    // Check cells
                    for (auto cell_it = current_cells.begin(); cell_it != current_cells.end(); ++cell_it)
                    {
                        const auto& current_cell = *cell_it;

                        if (std::find(comparison.begin(), comparison.end(), current_cell) == comparison.end())
                        {
                            // Terminate if position is outside of the reference cells
                            return std::make_pair(false, streamline);
                        }

                        if (visited_cells.find(current_cell) != visited_cells.end())
                        {
                            // Return results
                            return std::make_pair(true, streamline);
                        }

                        visited_cells.insert(current_cell);
                    }
                }
            }
        }

        std::vector<periodic_orbits::coords_t> periodic_orbits::get_cells(const tpf::data::grid<float, float, 2, 2>& grid, coords_t source,
            const coords_t& target, const Eigen::Vector2f& source_position, const Eigen::Vector2f& target_position) const
        {
            std::vector<coords_t> intermediate_cells;

            coords_t direction = target - source;

            while (direction[0] != 0 && direction[1] != 0)
            {
                const auto corner_bl = grid.get_cell_coordinates(source);
                const auto corner_br = grid.get_cell_coordinates(source + coords_t(1, 0));
                const auto corner_tl = grid.get_cell_coordinates(source + coords_t(0, 1));
                const auto corner_tr = grid.get_cell_coordinates(source + coords_t(1, 1));

                const kernel::Segment_2 edge_l(kernel::Point_2(corner_bl[0], corner_bl[1]), kernel::Point_2(corner_tl[0], corner_tl[1]));
                const kernel::Segment_2 edge_r(kernel::Point_2(corner_br[0], corner_br[1]), kernel::Point_2(corner_tr[0], corner_tr[1]));
                const kernel::Segment_2 edge_b(kernel::Point_2(corner_bl[0], corner_bl[1]), kernel::Point_2(corner_br[0], corner_br[1]));
                const kernel::Segment_2 edge_t(kernel::Point_2(corner_tl[0], corner_tl[1]), kernel::Point_2(corner_tr[0], corner_tr[1]));

                const kernel::Segment_2 displacement(kernel::Point_2(source_position[0], source_position[1]), kernel::Point_2(target_position[0], target_position[1]));

                const auto intersect_l = CGAL::do_intersect(displacement, edge_l);
                const auto intersect_r = CGAL::do_intersect(displacement, edge_r);
                const auto intersect_b = CGAL::do_intersect(displacement, edge_b);
                const auto intersect_t = CGAL::do_intersect(displacement, edge_t);

                if (intersect_l && source_position[0] > target_position[0])
                {
                    intermediate_cells.push_back(source - coords_t(1, 0));
                }
                else if (intersect_r && source_position[0] < target_position[0])
                {
                    intermediate_cells.push_back(source + coords_t(1, 0));
                }
                else if (intersect_b && source_position[1] > target_position[1])
                {
                    intermediate_cells.push_back(source - coords_t(0, 1));
                }
                else if (intersect_t && source_position[1] < target_position[1])
                {
                    intermediate_cells.push_back(source + coords_t(0, 1));
                }

                // Set new old cell
                source = intermediate_cells.back();
                direction = target - source;
            }

            return intermediate_cells;
        }

        std::vector<Eigen::Vector2f> periodic_orbits::integrate_orbit(const tpf::data::grid<float, float, 2, 2>& grid, Eigen::Vector2f position,
            float delta, const float sign, const float max_error, const float max_delta) const
        {
            // Extract edge
            const auto first_cell = *grid.find_staggered_cell(position);
            auto second_cell = first_cell;

            Eigen::Vector2f last_position = position;

            while (first_cell == second_cell)
            {
                const auto advected = advect_RK45(grid, position, delta, sign, max_error, max_delta);

                last_position = position;

                position = advected.first;
                delta = advected.second;

                second_cell = *grid.find_staggered_cell(position);
            }

            const auto corner_bl = grid.get_cell_coordinates(first_cell);
            const auto corner_br = grid.get_cell_coordinates(first_cell + coords_t(1, 0));
            const auto corner_tl = grid.get_cell_coordinates(first_cell + coords_t(0, 1));
            const auto corner_tr = grid.get_cell_coordinates(first_cell + coords_t(1, 1));

            const kernel::Segment_2 edge_l(kernel::Point_2(corner_bl[0], corner_bl[1]), kernel::Point_2(corner_tl[0], corner_tl[1]));
            const kernel::Segment_2 edge_r(kernel::Point_2(corner_br[0], corner_br[1]), kernel::Point_2(corner_tr[0], corner_tr[1]));
            const kernel::Segment_2 edge_b(kernel::Point_2(corner_bl[0], corner_bl[1]), kernel::Point_2(corner_br[0], corner_br[1]));
            const kernel::Segment_2 edge_t(kernel::Point_2(corner_tl[0], corner_tl[1]), kernel::Point_2(corner_tr[0], corner_tr[1]));

            kernel::Segment_2 edge;
            CGAL::Orientation crossed_orientation;

            {
                const kernel::Segment_2 line(kernel::Point_2(last_position[0], last_position[1]), kernel::Point_2(position[0], position[1]));

                if (CGAL::do_intersect(line, edge_l))
                {
                    edge = edge_l;
                }
                else if (CGAL::do_intersect(line, edge_r))
                {
                    edge = edge_r;
                }
                else if (CGAL::do_intersect(line, edge_b))
                {
                    edge = edge_b;
                }
                else if (CGAL::do_intersect(line, edge_t))
                {
                    edge = edge_t;
                }

                crossed_orientation = CGAL::orientation(edge.source(), edge.target(), kernel::Point_2(position[0], position[1]));
            }

            // Start stream line at the edge
            std::vector<Eigen::Vector2f> periodic_orbit;
            const std::size_t num_iterations = static_cast<std::size_t>(this->poincare_iterations.Param<core::param::IntParam>()->Value());

            for (std::size_t i = 0; i < num_iterations; ++i)
            {
                const kernel::Point_2 start_point = edge.source() + 0.5 * (edge.target() - edge.source());
                position = Eigen::Vector2f(static_cast<float>(CGAL::to_double(start_point[0])), static_cast<float>(CGAL::to_double(start_point[1])));

                bool halfway = false;

                while (!(halfway && CGAL::orientation(edge.source(), edge.target(), kernel::Point_2(position[0], position[1])) == crossed_orientation))
                {
                    const auto advected = advect_RK45(grid, position, delta, sign, max_error, max_delta);

                    const Eigen::Vector2f old_position = position;

                    position = advected.first;
                    delta = advected.second;

                    if (!halfway && CGAL::orientation(edge.source(), edge.target(), kernel::Point_2(position[0], position[1])) != crossed_orientation)
                    {
                        halfway = true;
                    }

                    if (i == num_iterations - 1)
                    {
                        periodic_orbit.push_back(position);
                    }
                }

                const kernel::Point_2 end_point(position[0], position[1]);

                if ((edge.source() - end_point).squared_length() < (edge.target() - end_point).squared_length())
                {
                    edge = kernel::Segment_2(edge.source(), start_point);
                }
                else
                {
                    edge = kernel::Segment_2(start_point, edge.target());
                }
            }

            // Add first position to close the orbit
            periodic_orbit.push_back(periodic_orbit.front());

            return periodic_orbit;
        }

        Eigen::Vector2f periodic_orbits::linear_interpolate_position(const Eigen::Vector2f& left, const Eigen::Vector2f& right, const float value_left, const float value_right) const
        {
            const auto lambda = value_left / (value_left - value_right);

            return left + lambda * (right - left);
        }
    }
}
