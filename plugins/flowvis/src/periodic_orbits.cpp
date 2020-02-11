#include "stdafx.h"
#include "periodic_orbits.h"

#include "critical_points.h"
#include "glyph_data_call.h"
#include "mouse_click_call.h"
#include "vector_field_call.h"

#include "mmcore/Call.h"
#include "mmcore/DirectDataWriterCall.h"
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
#include <functional>
#include <list>
#include <mutex>
#include <ostream>
#include <thread>
#include <tuple>
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
            file_output_slot("file_output_slot", "Slot for writing results to file"),
            vector_field_slot("get_vector_field", "Vector field input"),
            vector_field_hash(-1),
            critical_points_slot("get_critical_points", "Critical points input"),
            critical_points_hash(-1),
            integration_method("integration_method", "Method used for stream line integration"),
            integration_direction("integration_direction", "Direction of integration"),
            min_steps_per_cell("min_steps_per_cell", "Minimum number of stream line integration steps per cell"),
            initial_timestep("initial_timestep", "Initial time step for stream line integration"),
            maximum_timestep("maximum_timestep", "Maximum time step for stream line integration"),
            maximum_error("maximum_error", "Maximum error of the time step for stream line integration"),
            poincare_error("poincare_error", "Maximum error for representive position of periodic orbits"),
            unique_detection("unique_detection", "Ensure unique detection"),
            critical_point_detection("critical_point_detection", "Dismiss possible periodic orbits near attracting critical points"),
            output_representive("output_representive", "Output representive point for periodic orbits and terminated stream lines"),
            output_exit_streamlines("output_exit_streamlines", "Output stream lines from exit search"),
            output_critical_points("output_critical_points", "Also write input critical points to file?"),
            output_critical_points_finished(false),
            stop("stop", "Stop the currently running integration processes"),
            reset("reset", "Reset and clear all previous results"),
            output("output", "Output next valid turn"),
            num_threads(0), terminate(false), output_next(false)
        {
            // Connect output
            this->glyph_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(0), &periodic_orbits::get_glyph_data_callback);
            this->glyph_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(1), &periodic_orbits::get_glyph_extent_callback);
            this->MakeSlotAvailable(&this->glyph_slot);

            this->mouse_slot.SetCallback(mouse_click_call::ClassName(), mouse_click_call::FunctionName(0), &periodic_orbits::get_mouse_coordinates_callback);
            this->MakeSlotAvailable(&this->mouse_slot);

            this->file_output_slot.SetCallback(core::DirectDataWriterCall::ClassName(), core::DirectDataWriterCall::FunctionName(0), &periodic_orbits::get_output_callback);
            this->MakeSlotAvailable(&this->file_output_slot);
            this->get_output = []() -> std::ostream& { static std::ostream dummy(nullptr); return dummy; };

            // Connect input
            this->vector_field_slot.SetCompatibleCall<vector_field_call::vector_field_description>();
            this->MakeSlotAvailable(&this->vector_field_slot);

            this->critical_points_slot.SetCompatibleCall<glyph_data_call::glyph_data_description>();
            this->MakeSlotAvailable(&this->critical_points_slot);

            // Set parameters
            this->integration_method << new core::param::EnumParam(0);
            this->integration_method.Param<core::param::EnumParam>()->SetTypePair(static_cast<int>(integration_parameter_t::method_t::RUNGE_KUTTA_4), "Runge-Kutta 4 (fixed)");
            this->integration_method.Param<core::param::EnumParam>()->SetTypePair(static_cast<int>(integration_parameter_t::method_t::RUNGE_KUTTA_45), "Runge-Kutta 4-5 (dynamic)");
            this->MakeSlotAvailable(&this->integration_method);

            this->integration_direction << new core::param::EnumParam(0);
            this->integration_direction.Param<core::param::EnumParam>()->SetTypePair(0, "Forward and backward");
            this->integration_direction.Param<core::param::EnumParam>()->SetTypePair(1, "Forward");
            this->integration_direction.Param<core::param::EnumParam>()->SetTypePair(2, "Backward");
            this->MakeSlotAvailable(&this->integration_direction);

            this->min_steps_per_cell << new core::param::IntParam(10);
            this->MakeSlotAvailable(&this->min_steps_per_cell);

            this->initial_timestep << new core::param::FloatParam(0.01f);
            this->initial_timestep.Parameter()->SetGUIVisible(false);
            this->MakeSlotAvailable(&this->initial_timestep);

            this->maximum_timestep << new core::param::FloatParam(0.01f);
            this->maximum_timestep.Parameter()->SetGUIVisible(false);
            this->MakeSlotAvailable(&this->maximum_timestep);

            this->maximum_error << new core::param::FloatParam(0.000001f);
            this->maximum_error.Parameter()->SetGUIVisible(false);
            this->MakeSlotAvailable(&this->maximum_error);

            this->poincare_error << new core::param::FloatParam(0.000001f);
            this->MakeSlotAvailable(&this->poincare_error);

            this->unique_detection << new core::param::BoolParam(true);
            this->MakeSlotAvailable(&this->unique_detection);

            this->critical_point_detection << new core::param::BoolParam(false);
            this->MakeSlotAvailable(&this->critical_point_detection);

            this->output_representive << new core::param::BoolParam(false);
            this->MakeSlotAvailable(&this->output_representive);

            this->output_exit_streamlines << new core::param::BoolParam(false);
            this->MakeSlotAvailable(&this->output_exit_streamlines);

            this->output_critical_points << new core::param::BoolParam(false);
            this->MakeSlotAvailable(&this->output_critical_points);

            this->stop << new core::param::ButtonParam();
            this->stop.SetUpdateCallback(&periodic_orbits::stop_callback);
            this->MakeSlotAvailable(&this->stop);

            this->reset << new core::param::ButtonParam();
            this->reset.SetUpdateCallback(&periodic_orbits::reset_callback);
            this->MakeSlotAvailable(&this->reset);

            this->output << new core::param::ButtonParam();
            this->output.SetUpdateCallback(&periodic_orbits::output_callback);
            this->MakeSlotAvailable(&this->output);
        }

        periodic_orbits::~periodic_orbits()
        {
            this->terminate = true;

            while (this->num_threads != 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

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
                // Set visibility of the GUI elements
                const bool rk4 = this->integration_method.Param<core::param::EnumParam>()->Value() == static_cast<int>(integration_parameter_t::method_t::RUNGE_KUTTA_4);
                const bool rk45 = this->integration_method.Param<core::param::EnumParam>()->Value() == static_cast<int>(integration_parameter_t::method_t::RUNGE_KUTTA_45);

                this->min_steps_per_cell.Parameter()->SetGUIVisible(rk4);

                this->initial_timestep.Parameter()->SetGUIVisible(rk45);
                this->maximum_timestep.Parameter()->SetGUIVisible(rk45);
                this->maximum_error.Parameter()->SetGUIVisible(rk45);

                // Get vector field and critical points
                auto* get_vector_field = this->vector_field_slot.CallAs<vector_field_call>();
                auto* get_critical_points = this->critical_points_slot.CallAs<glyph_data_call>();

                const bool has_vector_field = get_vector_field != nullptr && (*get_vector_field)(0);
                const bool has_critical_points = get_critical_points != nullptr && (*get_critical_points)(0);
                const bool has_different_input = (get_vector_field->DataHash() != this->vector_field_hash ||
                    get_critical_points->DataHash() != this->critical_points_hash);

                if (has_vector_field && has_critical_points && has_different_input)
                {
                    std::lock_guard<std::mutex> locker(this->lock);

                    if (this->num_threads != 0)
                    {
                        return true;
                    }

                    this->vector_field_hash = get_vector_field->DataHash();
                    this->critical_points_hash = get_critical_points->DataHash();

                    std::vector<double> vectors(get_vector_field->get_vectors()->begin(), get_vector_field->get_vectors()->end());

                    const auto& critical_points = get_critical_points->get_points();

                    // Create grid from input vector field
                    tpf::data::extent_t extent(2);
                    extent[0].first = extent[1].first = 0;
                    extent[0].second = get_vector_field->get_resolution()[0] - 1;
                    extent[1].second = get_vector_field->get_resolution()[1] - 1;

                    tpf::data::grid_information<double>::array_type cell_sizes(2);
                    cell_sizes[0].resize(get_vector_field->get_resolution()[0]);
                    cell_sizes[1].resize(get_vector_field->get_resolution()[1]);

                    std::fill(cell_sizes[0].begin(), cell_sizes[0].end(),
                        (get_vector_field->get_bounding_rectangle().Right() - get_vector_field->get_bounding_rectangle().Left()) / (get_vector_field->get_resolution()[0] - 1));
                    std::fill(cell_sizes[1].begin(), cell_sizes[1].end(),
                        (get_vector_field->get_bounding_rectangle().Top() - get_vector_field->get_bounding_rectangle().Bottom()) / (get_vector_field->get_resolution()[1] - 1));

                    tpf::data::grid_information<double>::array_type cell_coordinates(2);
                    tpf::data::grid_information<double>::array_type node_coordinates(2);

                    const std::array<double, 2> offset{ get_vector_field->get_bounding_rectangle().Left(), get_vector_field->get_bounding_rectangle().Bottom() };

                    for (std::size_t d = 0; d < 2; ++d)
                    {
                        cell_coordinates[d].resize(cell_sizes[d].size());
                        node_coordinates[d].resize(cell_sizes[d].size() + 1);

                        cell_coordinates[d][0] = offset[d];
                        node_coordinates[d][0] = offset[d] - 0.5 * cell_sizes[d][0];

                        for (std::size_t i = 1; i < cell_sizes[d].size(); ++i)
                        {
                            cell_coordinates[d][i] = cell_coordinates[d][i - 1] + 0.5 * (cell_sizes[d][i - 1] + cell_sizes[d][i]);
                            node_coordinates[d][i] = node_coordinates[d][i - 1] + cell_sizes[d][i - 1];
                        }

                        node_coordinates[d][cell_sizes[d].size()] = node_coordinates[d][cell_sizes[d].size() - 1] + cell_sizes[d][cell_sizes[d].size() - 1];
                    }

                    this->grid = tpf::data::grid<double, double, 2, 2>("vector_field", extent, std::move(vectors),
                        std::move(cell_coordinates), std::move(node_coordinates), std::move(cell_sizes));

                    // Reset output
                    this->line_output.clear();
                    this->point_output.clear();
                    this->glyph_hash = -1;

                    this->orbit_cells.clear();

                    // Store critical points
                    this->input_critical_points.reserve(critical_points.size());

                    for (const auto& critical_point : critical_points)
                    {
                        this->input_critical_points.push_back(
                            std::make_pair(static_cast<critical_points::type>(static_cast<int>(critical_point.second)),
                                Eigen::Vector2d(critical_point.first[0], critical_point.first[1])));
                    }
                }

                // Fill glyph call
                std::lock_guard<std::mutex> locker(this->lock);

                if (glyph_call->DataHash() != this->glyph_hash)
                {
                    glyph_call->clear();

                    for (std::size_t i = 0; i < this->line_output.size(); ++i)
                    {
                        glyph_call->add_line(this->line_output[i].second, this->line_output[i].first);
                    }

                    for (std::size_t i = 0; i < this->point_output.size(); ++i)
                    {
                        glyph_call->add_point(this->point_output[i].second, this->point_output[i].first);
                    }

                    glyph_call->SetDataHash(this->glyph_hash);
                }
            }
            else
            {
                return false;
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

            std::lock_guard<std::mutex> locker(this->lock);

            if (get_mouse_coordinates != nullptr && this->grid.get_num_elements() > 0)
            {
                const Eigen::Vector2d seed(get_mouse_coordinates->get_coordinates().first, get_mouse_coordinates->get_coordinates().second);

                // Seed a stream lines
                const auto direction = this->integration_direction.Param<core::param::EnumParam>()->Value();

                if (direction == 0 || direction == 1)
                {
                    std::thread(&periodic_orbits::extract_periodic_orbit, this, this->grid, this->input_critical_points, seed, 1.0f).detach();
                }

                if (direction == 0 || direction == 2)
                {
                    std::thread(&periodic_orbits::extract_periodic_orbit, this, this->grid, this->input_critical_points, seed, -1.0f).detach();
                }
            }

            return true;
        }

        bool periodic_orbits::get_output_callback(core::Call& call)
        {
            auto* get_output_cb = dynamic_cast<core::DirectDataWriterCall*>(&call);

            if (get_output_cb != nullptr)
            {
                std::lock_guard<std::mutex> locker(this->lock);

                this->get_output = get_output_cb->GetCallback();

                if (!this->output_critical_points_finished && this->output_critical_points.Param<core::param::BoolParam>()->Value())
                {
                    this->get_output() << "# Critical points" << std::endl;

                    for (const auto& critical_point : this->input_critical_points)
                    {
                        this->get_output() << critical_point.second[0] << "," << critical_point.second[1] << std::endl;
                    }

                    this->get_output() << "# Periodic orbits" << std::endl;

                    this->output_critical_points_finished = true;
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

            this->line_output.clear();
            this->point_output.clear();
            ++this->glyph_hash;

            this->orbit_cells.clear();

            this->get_output() << "# Periodic orbits" << std::endl;

            return true;
        }

        bool periodic_orbits::output_callback(core::param::ParamSlot&)
        {
            std::lock_guard<std::mutex> locker(this->lock);

            this->output_next = true;

            return true;
        }

        void periodic_orbits::extract_periodic_orbit(const tpf::data::grid<double, double, 2, 2>& grid,
            const std::vector<std::pair<critical_points::type, Eigen::Vector2d>>& input_critical_points, const Eigen::Vector2d& seed, const float sign)
        {
            try
            {
                auto create_output = [](const std::vector<Eigen::Vector2d>& input) -> std::vector<Eigen::Vector2f>
                {
                    std::vector<Eigen::Vector2f> output;
                    output.reserve(input.size());
                    std::transform(input.begin(), input.end(), std::back_inserter(output), [](const Eigen::Vector2d& value) { return value.cast<float>(); });
                    return output;
                };

                // Get parameters for stream line integration
                integration_parameter_t integration;
                integration.sign = sign;

                float max_poincare_error;

                bool unique_detection;
                bool critical_point_detection;

                bool output_representive;
                bool output_exit_streamline;

                {
                    std::lock_guard<std::mutex> locker(this->lock);

                    this->terminate = false;

                    vislib::sys::Log::DefaultLog.WriteInfo("Number of processes running: %d", ++this->num_threads);

                    vislib::sys::Log::DefaultLog.WriteInfo("Starting %s stream line at [%.5f, %.5f]", (sign < 0.0f) ? "backward" : "forward", seed[0], seed[1]);

                    integration.method = static_cast<integration_parameter_t::method_t>(this->integration_method.Param<core::param::EnumParam>()->Value());

                    if (integration.method == integration_parameter_t::method_t::RUNGE_KUTTA_4)
                    {
                        integration.param.rk_4.min_steps_per_cell = this->min_steps_per_cell.Param<core::param::IntParam>()->Value();
                    }
                    else
                    {
                        integration.param.rk_45.timestep = this->initial_timestep.Param<core::param::FloatParam>()->Value();
                        integration.param.rk_45.maximum_timestep = this->maximum_timestep.Param<core::param::FloatParam>()->Value();
                        integration.param.rk_45.maximum_error = this->maximum_error.Param<core::param::FloatParam>()->Value();
                    }

                    max_poincare_error = this->poincare_error.Param<core::param::FloatParam>()->Value();

                    unique_detection = this->unique_detection.Param<core::param::BoolParam>()->Value();
                    critical_point_detection = this->critical_point_detection.Param<core::param::BoolParam>()->Value();

                    output_representive = this->output_representive.Param<core::param::BoolParam>()->Value();
                    output_exit_streamline = this->output_exit_streamlines.Param<core::param::BoolParam>()->Value();
                }

                // Start
                std::vector<Eigen::Vector2d> periodic_orbit;

                auto position = seed;
                bool has_exit = true;
                bool output_now = false;

                while (has_exit && !this->terminate)
                {
                    // Find turn
                    const auto visited_cells = find_turn(grid, input_critical_points, position, integration, critical_point_detection);

                    if (!this->terminate)
                    {
                        if (visited_cells && !visited_cells->first.empty())
                        {
                            // Check if a periodic orbit with these cells was already extracted
                            const std::set<coords_t, std::less<coords_t>> sorted_visited_cells(visited_cells->first.cbegin(), visited_cells->first.cend());

                            bool already_found = false;

                            {
                                std::lock_guard<std::mutex> locker(this->lock);

                                already_found = std::find(this->orbit_cells.begin(), this->orbit_cells.end(), sorted_visited_cells) != this->orbit_cells.end();
                            }

                            if (!unique_detection || !already_found)
                            {
                                // Do a second turn and compare results
                                auto validation = validate_turn(grid, position, integration, visited_cells->first);

                                std::swap(periodic_orbit, std::get<1>(validation));

                                if (std::get<0>(validation))
                                {
                                    // Look for possible exits
                                    auto vector_hash = [](const Eigen::Vector2d& coords) -> std::size_t
                                    {
                                        return core::utility::DataHash(coords[0], coords[1]);
                                    };

                                    std::unordered_set<Eigen::Vector2d, decltype(vector_hash)> possible_exits(23, vector_hash);

                                    auto add_if_correct_ = [this, &possible_exits](const Eigen::Vector2d& position,
                                        const std::list<kernel::Point_2>& outer, const std::vector<kernel::Point_2>& inner) -> void
                                    {
                                        if (correct_side(position, outer, inner))
                                        {
                                            possible_exits.insert(position);
                                        }
                                    };

                                    auto add_if_correct = std::bind(add_if_correct_, std::placeholders::_1, visited_cells->second, std::get<2>(validation));

                                    for (const auto& cell : visited_cells->first)
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
                                            add_if_correct(linear_interpolate_position(corner_bl, corner_br, value_bl[1], value_br[1]));
                                        }
                                        if (std::signbit(value_tl[1]) != std::signbit(value_tr[1]))
                                        {
                                            add_if_correct(linear_interpolate_position(corner_tl, corner_tr, value_tl[1], value_tr[1]));
                                        }
                                        if (std::signbit(value_bl[0]) != std::signbit(value_tl[0]))
                                        {
                                            add_if_correct(linear_interpolate_position(corner_bl, corner_tl, value_bl[0], value_tl[0]));
                                        }
                                        if (std::signbit(value_br[0]) != std::signbit(value_tr[0]))
                                        {
                                            add_if_correct(linear_interpolate_position(corner_br, corner_tr, value_br[0], value_tr[0]));
                                        }

                                        add_if_correct(corner_bl);
                                        add_if_correct(corner_br);
                                        add_if_correct(corner_tl);
                                        add_if_correct(corner_tr);
                                    }

                                    // Check for output command
                                    {
                                        std::lock_guard<std::mutex> locker(this->lock);

                                        output_now = this->output_next;
                                        this->output_next = false;

                                        if (output_now)
                                        {
                                            vislib::sys::Log::DefaultLog.WriteInfo("Requested output");
                                        }
                                    }

                                    // Backward integration from possible exits
                                    has_exit = false;

                                    std::vector<std::vector<Eigen::Vector2d>> exit_tries;

                                    for (auto possible_exit_it = possible_exits.cbegin(); possible_exit_it != possible_exits.cend() && (!has_exit || output_now); ++possible_exit_it)
                                    {
                                        Eigen::Vector2d possible_exit = *possible_exit_it;

                                        integration.sign *= -1.0f;
                                        const auto exit = find_exits(grid, possible_exit, integration, visited_cells->first);
                                        integration.sign *= -1.0f;

                                        has_exit = exit.first;

                                        if (output_exit_streamline)
                                        {
                                            exit_tries.push_back(exit.second);
                                        }
                                    }

                                    if (output_exit_streamline && (!has_exit || output_now))
                                    {
                                        std::lock_guard<std::mutex> locker(this->lock);

                                        for (const auto& exit_try : exit_tries)
                                        {
                                            this->line_output.push_back(std::make_pair(0.5f * sign, create_output(exit_try)));
                                        }

                                        for (const auto& cell : visited_cells->first)
                                        {
                                            const auto corner_bl = grid.get_cell_coordinates(cell);
                                            const auto corner_br = grid.get_cell_coordinates(cell + coords_t(1, 0));
                                            const auto corner_tl = grid.get_cell_coordinates(cell + coords_t(0, 1));
                                            const auto corner_tr = grid.get_cell_coordinates(cell + coords_t(1, 1));

                                            std::vector<Eigen::Vector2d> box{ corner_bl, corner_br, corner_tr, corner_tl, corner_bl };

                                            this->line_output.push_back(std::make_pair(0.0f, create_output(box)));
                                        }

                                        ++this->glyph_hash;
                                    }

                                    output_now = false;
                                }

                                // Add list of cells to already extracted periodic orbits
                                if (!has_exit && !this->terminate && unique_detection)
                                {
                                    std::lock_guard<std::mutex> locker(this->lock);

                                    this->orbit_cells.push_back(sorted_visited_cells);
                                }
                            }
                            else
                            {
                                std::lock_guard<std::mutex> locker(this->lock);

                                vislib::sys::Log::DefaultLog.WriteWarn("Periodic orbit found from %s stream line at [%.5f, %.5f] was already extracted",
                                    (sign < 0.0f) ? "backward" : "forward", seed[0], seed[1]);

                                vislib::sys::Log::DefaultLog.WriteInfo("Number of processes running: %d", --this->num_threads);

                                return;
                            }
                        }
                        else
                        {
                            std::lock_guard<std::mutex> locker(this->lock);

                            vislib::sys::Log::DefaultLog.WriteWarn("Could not find a periodic orbit from %s stream line at [%.5f, %.5f], which ended at [%.5f, %.5f]",
                                (sign < 0.0f) ? "backward" : "forward", seed[0], seed[1], position[0], position[1]);

                            if (output_representive)
                            {
                                this->point_output.push_back(std::make_pair(0.0f, Eigen::Vector2f(static_cast<float>(position[0]), static_cast<float>(position[1]))));
                                ++this->glyph_hash;
                            }

                            vislib::sys::Log::DefaultLog.WriteInfo("Number of processes running: %d", --this->num_threads);

                            return;
                        }
                    }
                }

                if (!this->terminate)
                {
                    // Use Poincaré map for finding the closed stream line
                    if (!output_exit_streamline)
                    {
                        periodic_orbit = integrate_orbit(grid, position, integration, max_poincare_error);
                    }

                    // Output results
                    std::lock_guard<std::mutex> locker(this->lock);

                    this->line_output.push_back(std::make_pair(sign, create_output(periodic_orbit)));

                    if (output_representive)
                    {
                        this->point_output.push_back(std::make_pair(sign, Eigen::Vector2f(static_cast<float>(periodic_orbit[0][0]), static_cast<float>(periodic_orbit[0][1]))));
                    }

                    for (const auto& point : periodic_orbit)
                    {
                        this->glyph_hash = static_cast<SIZE_T>(core::utility::DataHash(this->glyph_hash, point[0], point[1]));
                    }

                    vislib::sys::Log::DefaultLog.WriteInfo("Periodic orbit found at [%.5f, %.5f] from %s stream line at [%.5f, %.5f]",
                        periodic_orbit[0][0], periodic_orbit[0][1], (sign < 0.0f) ? "backward" : "forward", seed[0], seed[1]);

                    vislib::sys::Log::DefaultLog.WriteInfo("Number of processes running: %d", --this->num_threads);

                    // Output results to file
                    this->get_output() << periodic_orbit[0][0] << "," << periodic_orbit[0][1] << std::endl;
                }
                else
                {
                    // Output that computation was terminated
                    std::lock_guard<std::mutex> locker(this->lock);

                    vislib::sys::Log::DefaultLog.WriteInfo("Search for periodic orbit from %s stream line at [%.5f, %.5f] terminated at [%.5f, %.5f]",
                        (sign < 0.0f) ? "backward" : "forward", seed[0], seed[1], position[0], position[1]);

                    if (output_representive)
                    {
                        this->point_output.push_back(std::make_pair(0.0f, Eigen::Vector2f(static_cast<float>(position[0]), static_cast<float>(position[1]))));
                        ++this->glyph_hash;
                    }

                    vislib::sys::Log::DefaultLog.WriteInfo("Number of processes running: %d", --this->num_threads);
                }
            }
            catch (const std::exception& e)
            {
                vislib::sys::Log::DefaultLog.WriteError("Error while extracting periodic orbits: %s", e.what());
            }
            catch (...)
            {
                vislib::sys::Log::DefaultLog.WriteError("Unknown error while extracting periodic orbits");
            }
        }

        Eigen::Vector2d periodic_orbits::advect(const tpf::data::grid<double, double, 2, 2>& grid,
            const Eigen::Vector2d& position, integration_parameter_t& integration_parameter) const
        {
            try
            {
                switch (integration_parameter.method)
                {
                case integration_parameter_t::method_t::RUNGE_KUTTA_4:
                    return advect_RK4(grid, position, integration_parameter);
                case integration_parameter_t::method_t::RUNGE_KUTTA_45:
                    return advect_RK45(grid, position, integration_parameter);
                default:
                    return position;
                }
            }
            catch (...)
            {
                return position;
            }
        }

        Eigen::Vector2d periodic_orbits::advect_RK4(const tpf::data::grid<double, double, 2, 2>& grid,
            const Eigen::Vector2d& position, const integration_parameter_t& integration) const
        {
            const auto cell = *grid.find_staggered_cell(position);

            const auto cell_size = (grid.get_cell_coordinates(cell + coords_t(1, 1)) - grid.get_cell_coordinates(cell));
            const auto min_cell_size = std::min(cell_size[0], cell_size[1]);

            const auto max_velocity = std::max(std::max(grid(cell).norm(), grid(cell + coords_t(1, 0)).norm()),
                std::max(grid(cell + coords_t(0, 1)).norm(), grid(cell + coords_t(1, 1)).norm()));

            const auto steps_per_cell = min_cell_size / max_velocity;

            const auto delta = steps_per_cell / static_cast<double>(integration.param.rk_4.min_steps_per_cell);

            const auto k1 = delta * integration.sign * grid.interpolate(position);
            const auto k2 = delta * integration.sign * grid.interpolate(position + 0.5 * k1);
            const auto k3 = delta * integration.sign * grid.interpolate(position + 0.5 * k2);
            const auto k4 = delta * integration.sign * grid.interpolate(position + k3);

            return position + (1.0 / 6.0) * (k1 + 2.0 * (k2 + k3) + k4);
        }

        Eigen::Vector2d periodic_orbits::advect_RK45(const tpf::data::grid<double, double, 2, 2>& grid,
            const Eigen::Vector2d& position, integration_parameter_t& integration) const
        {
            // Cash-Karp parameters
            const auto b_21 = 0.2;
            const auto b_31 = 0.075;
            const auto b_41 = 0.3;
            const auto b_51 = -11.0 / 54.0;
            const auto b_61 = 1631.0 / 55296.0;
            const auto b_32 = 0.225;
            const auto b_42 = -0.9;
            const auto b_52 = 2.5;
            const auto b_62 = 175.0 / 512.0;
            const auto b_43 = 1.2;
            const auto b_53 = -70.0 / 27.0;
            const auto b_63 = 575.0 / 13824.0;
            const auto b_54 = 35.0 / 27.0;
            const auto b_64 = 44275.0 / 110592.0;
            const auto b_65 = 253.0 / 4096.0;

            const auto c_1 = 37.0 / 378.0;
            const auto c_2 = 0.0;
            const auto c_3 = 250.0 / 621.0;
            const auto c_4 = 125.0 / 594.0;
            const auto c_5 = 0.0;
            const auto c_6 = 512.0 / 1771.0;

            const auto c_1s = 2825.0 / 27648.0;
            const auto c_2s = 0.0;
            const auto c_3s = 18575.0 / 48384.0;
            const auto c_4s = 13525.0 / 55296.0;
            const auto c_5s = 277.0 / 14336.0;
            const auto c_6s = 0.25;

            // Constants
            const auto grow_exponent = -0.2;
            const auto shrink_exponent = -0.25;
            const auto max_growth = 5.0;
            const auto max_shrink = 0.1;
            const auto safety = 0.9;

            // Calculate Runge-Kutta coefficients
            Eigen::Vector2d output_position;

            bool decreased = false;

            do
            {
                const auto k1 = integration.param.rk_45.timestep * integration.sign * grid.interpolate(position);
                const auto k2 = integration.param.rk_45.timestep * integration.sign * grid.interpolate(position + b_21 * k1);
                const auto k3 = integration.param.rk_45.timestep * integration.sign * grid.interpolate(position + b_31 * k1 + b_32 * k2);
                const auto k4 = integration.param.rk_45.timestep * integration.sign * grid.interpolate(position + b_41 * k1 + b_42 * k2 + b_43 * k3);
                const auto k5 = integration.param.rk_45.timestep * integration.sign * grid.interpolate(position + b_51 * k1 + b_52 * k2 + b_53 * k3 + b_54 * k4);
                const auto k6 = integration.param.rk_45.timestep * integration.sign * grid.interpolate(position + b_61 * k1 + b_62 * k2 + b_63 * k3 + b_64 * k4 + b_65 * k5);

                // Calculate error estimate
                const auto fifth_order = position + c_1 * k1 + c_2 * k2 + c_3 * k3 + c_4 * k4 + c_5 * k5 + c_6 * k6;
                const auto fourth_order = position + c_1s * k1 + c_2s * k2 + c_3s * k3 + c_4s * k4 + c_5s * k5 + c_6s * k6;

                const auto difference = fifth_order - fourth_order;

                const auto scale = grid.interpolate(position);

                const auto error = std::max(0.0, std::max(std::abs(difference[0] / scale[0]), std::abs(difference[1] / scale[1]))) / integration.param.rk_45.maximum_error;

                // Set new, adapted time step
                if (error > 1.0)
                {
                    // Error too large, reduce time step
                    integration.param.rk_45.timestep *= static_cast<float>(std::max(max_shrink, safety * std::pow(error, shrink_exponent)));
                    decreased = true;
                }
                else
                {
                    // Error (too) small, increase time step
                    integration.param.rk_45.timestep = std::min(integration.param.rk_45.maximum_timestep,
                        static_cast<float>(integration.param.rk_45.timestep * std::min(max_growth, safety * std::pow(error, grow_exponent))));
                    decreased = false;
                }

                // Set output
                output_position = fifth_order;
            } while (decreased);

            return output_position;
        }

        tpf::utility::optional<std::pair<std::list<periodic_orbits::coords_t>, std::list<periodic_orbits::kernel::Point_2>>>
            periodic_orbits::find_turn(const tpf::data::grid<double, double, 2, 2>& grid,
                const std::vector<std::pair<critical_points::type, Eigen::Vector2d>>& input_critical_points,
                Eigen::Vector2d& position, integration_parameter_t& integration_param, const bool critical_point_detection) const
        {
            // Initialize cell list
            std::list<coords_t> visited_cells;
            std::list<periodic_orbits::kernel::Point_2> crossed_edges;

            // Advect stream line until a closed turn is detected
            std::size_t num_critical_point_visits = 0;

            bool found_turn = false;

            while (!found_turn && !this->terminate)
            {
                const auto old_position = position;

                position = advect(grid, position, integration_param);

                if (position == old_position)
                {
                    // Advection had no result; stream line stopped
                    vislib::sys::Log::DefaultLog.WriteWarn("Integration stopped");
                    return tpf::utility::nullopt;
                }

                // Check entry into new cell
                const auto old_cell = *grid.find_staggered_cell(old_position);
                const auto new_cell = grid.find_staggered_cell(position);

                if (!new_cell)
                {
                    // Advection went out-of-bounds
                    vislib::sys::Log::DefaultLog.WriteWarn("Integration went out-of-bounds");
                    return tpf::utility::nullopt;
                }

                if (old_cell != *new_cell)
                {
                    // Add cells that were overstepped
                    auto current_cells = get_cells(grid, old_cell, *new_cell, old_position, position);

                    for (auto current_cell_it = current_cells.begin(); current_cell_it != current_cells.end() && !found_turn; ++current_cell_it)
                    {
                        const auto& current_cell = *current_cell_it;

                        // Check for existance of critical point within the new cell
                        bool has_critical_point = false;

                        if (critical_point_detection)
                        {
                            for (const auto& point : input_critical_points)
                            {
                                const bool attracting =
                                    (integration_param.sign < 0.0f && (point.first == critical_points::type::REPELLING_FOCUS
                                        || point.first == critical_points::type::REPELLING_NODE)) ||
                                        (integration_param.sign > 0.0f && (point.first == critical_points::type::ATTRACTING_FOCUS
                                            || point.first == critical_points::type::ATTRACTING_NODE));

                                if (attracting && *grid.find_staggered_cell(point.second) == current_cell.first)
                                {
                                    has_critical_point = true;
                                }
                            }
                        }

                        if (!has_critical_point)
                        {
                            // Cell has changed: check for prior visitation
                            if (std::find(visited_cells.begin(), visited_cells.end(), current_cell.first) != visited_cells.end())
                            {
                                found_turn = true;

                                // Remove all cells which do not belong to this turn
                                for (auto cell_it = visited_cells.begin(); cell_it != visited_cells.end() && *cell_it != current_cell.first; )
                                {
                                    visited_cells.erase(cell_it++);
                                    crossed_edges.pop_front();
                                }
                            }
                            else
                            {
                                visited_cells.push_back(current_cell.first);
                                crossed_edges.push_back(current_cell.second);
                            }
                        }
                        else
                        {
                            ++num_critical_point_visits;

                            visited_cells.clear();
                            crossed_edges.clear();
                        }

                        // Return, if stuck in a loop with a critical point
                        if (num_critical_point_visits == 1000)
                        {
                            vislib::sys::Log::DefaultLog.WriteWarn("Integration was stuck near a critical point");
                            return tpf::utility::nullopt;
                        }
                    }
                }
            }

            // Return results
            return std::make_pair(visited_cells, crossed_edges);
        }

        std::tuple<bool, std::vector<Eigen::Vector2d>, std::vector<periodic_orbits::kernel::Point_2>> periodic_orbits::validate_turn(
            const tpf::data::grid<double, double, 2, 2>& grid, Eigen::Vector2d& position,
            integration_parameter_t& integration_param, std::list<coords_t> comparison) const
        {
            // Initialize comparison
            std::vector<Eigen::Vector2d> streamline;
            streamline.push_back(position);

            std::vector<kernel::Point_2> crossed_edges;

            comparison.push_back(comparison.front());
            comparison.pop_front();

            // Advect stream line while it corresponds to the input list of cells
            while (!comparison.empty())
            {
                const auto old_position = position;

                position = advect(grid, position, integration_param);

                if (position == old_position)
                {
                    // Advection had no result; stream line stopped
                    return std::make_tuple(false, streamline, crossed_edges);
                }

                // Check entry into new cell
                const auto old_cell = *grid.find_staggered_cell(old_position);
                const auto new_cell = grid.find_staggered_cell(position);

                if (!new_cell)
                {
                    // Advection went out-of-bounds
                    return std::make_tuple(false, streamline, crossed_edges);
                }

                streamline.push_back(position);

                if (old_cell != *new_cell)
                {
                    // Add cells that were overstepped
                    auto current_cells = get_cells(grid, old_cell, *new_cell, old_position, position);

                    // Check cells
                    for (auto cell_it = current_cells.begin(); cell_it != current_cells.end() && !comparison.empty(); ++cell_it)
                    {
                        const auto& current_cell = *cell_it;

                        if (current_cell.first != comparison.front())
                        {
                            return std::make_tuple(false, streamline, crossed_edges);
                        }

                        comparison.remove(current_cell.first);

                        crossed_edges.push_back(current_cell.second);
                    }
                }
            }

            // Return results
            return std::make_tuple(true, streamline, crossed_edges);
        }

        std::pair<bool, std::vector<Eigen::Vector2d>> periodic_orbits::find_exits(const tpf::data::grid<double, double, 2, 2>& grid,
            Eigen::Vector2d& position, integration_parameter_t& integration_param, const std::list<coords_t>& comparison) const
        {
            // Initialize comparison
            std::vector<Eigen::Vector2d> streamline;
            streamline.push_back(position);

            std::unordered_set<coords_t, std::hash<coords_t>> visited_cells;

            // Advect one step to ensure the start is not considered outside
            {
                position = advect(grid, position, integration_param);

                streamline.push_back(position);

                bool is_in_cell = false;

                std::for_each(comparison.begin(), comparison.end(), [&grid, &position, &is_in_cell](const coords_t& coords)
                {
                    is_in_cell |= grid.is_in_staggered_cell(coords, position);
                });

                if (!is_in_cell)
                {
                    // Still outside
                    return std::make_pair(false, streamline);
                }
            }

            // Advect stream line while it corresponds to the input list of cells
            bool first_cell = true;

            while (!this->terminate)
            {
                const auto old_position = position;

                position = advect(grid, position, integration_param);

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

                    // Check cells
                    for (auto cell_it = current_cells.begin(); cell_it != current_cells.end(); ++cell_it)
                    {
                        const auto& current_cell = cell_it->first;

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

                        // Do one extra
                        if (first_cell)
                        {
                            first_cell = false;
                        }
                        else
                        {
                            visited_cells.insert(current_cell);
                        }
                    }
                }
            }

            // Terminated, anyway
            return std::make_pair(false, streamline);
        }

        std::vector<std::pair<periodic_orbits::coords_t, periodic_orbits::kernel::Point_2>> periodic_orbits::get_cells(const tpf::data::grid<double, double, 2, 2>& grid,
            coords_t source, const coords_t& target, const Eigen::Vector2d& source_position, const Eigen::Vector2d& target_position) const
        {
            std::vector<std::pair<coords_t, kernel::Point_2>> cells;

            coords_t direction = target - source;

            while (direction[0] != 0 || direction[1] != 0)
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
                    cells.push_back(std::make_pair(static_cast<coords_t>(source - coords_t(1, 0)), boost::get<kernel::Point_2>(*CGAL::intersection(displacement, edge_l))));
                }
                else if (intersect_r && source_position[0] < target_position[0])
                {
                    cells.push_back(std::make_pair(static_cast<coords_t>(source + coords_t(1, 0)), boost::get<kernel::Point_2>(*CGAL::intersection(displacement, edge_r))));
                }
                else if (intersect_b && source_position[1] > target_position[1])
                {
                    cells.push_back(std::make_pair(static_cast<coords_t>(source - coords_t(0, 1)), boost::get<kernel::Point_2>(*CGAL::intersection(displacement, edge_b))));
                }
                else if (intersect_t && source_position[1] < target_position[1])
                {
                    cells.push_back(std::make_pair(static_cast<coords_t>(source + coords_t(0, 1)), boost::get<kernel::Point_2>(*CGAL::intersection(displacement, edge_t))));
                }

                // Set new old cell
                source = cells.back().first;
                direction = target - source;
            }

            return cells;
        }

        bool periodic_orbits::correct_side(const Eigen::Vector2d& position, const std::list<kernel::Point_2>& outer, const std::vector<kernel::Point_2>& inner) const
        {
            const kernel::Point_2 point(position[0], position[1]);

            // Create lambda for distance calculation
            auto calculate_distance = [&point](const auto& list)
            {
                auto distance = CGAL::squared_distance(kernel::Segment_2(list.back(), list.front()), point);

                auto last = list.front();

                for (auto it = ++list.begin(); it != list.end(); ++it)
                {
                    const auto current = *it;

                    distance = CGAL::min(distance, CGAL::squared_distance(kernel::Segment_2(last, current), point));

                    last = current;
                }

                return distance;
            };

            // Calculate and compare distances
            return calculate_distance(inner) < calculate_distance(outer);
        }

        std::vector<Eigen::Vector2d> periodic_orbits::integrate_orbit(const tpf::data::grid<double, double, 2, 2>& grid, Eigen::Vector2d position,
            integration_parameter_t integration_param, const float max_poincare_error) const
        {
            // Extract edge
            const auto first_cell = *grid.find_staggered_cell(position);
            auto second_cell = first_cell;

            auto last_position = position;

            while (first_cell == second_cell)
            {
                last_position = position;

                position = advect(grid, position, integration_param);

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

            // Compute actual periodic orbit, until the error is small enough
            std::vector<Eigen::Vector2d> periodic_orbit;

            auto error = std::numeric_limits<double>::max();

            while (error > max_poincare_error)
            {
                periodic_orbit.clear();

                // Calculate new start point
                const kernel::Point_2 start_point = edge.source() + 0.5 * (edge.target() - edge.source());
                position << CGAL::to_double(start_point[0]), CGAL::to_double(start_point[1]);

                bool halfway = false;

                // Integrate stream line until it crosses the edge again
                while (!(halfway && CGAL::orientation(edge.source(), edge.target(), kernel::Point_2(position[0], position[1])) == crossed_orientation))
                {
                    last_position = position;

                    position = advect(grid, position, integration_param);

                    if (!halfway && CGAL::orientation(edge.source(), edge.target(), kernel::Point_2(position[0], position[1])) != crossed_orientation)
                    {
                        halfway = true;
                    }

                    periodic_orbit.push_back(position);
                }

                // Compute edge intersection and cut edge accordingly
                const auto edge_intersection = CGAL::intersection(edge,
                    kernel::Segment_2(kernel::Point_2(last_position[0], last_position[1]), kernel::Point_2(position[0], position[1])));

                if (edge_intersection)
                {
                    if ((edge.source() - boost::get<kernel::Point_2>(*edge_intersection)).squared_length() <
                        (edge.target() - boost::get<kernel::Point_2>(*edge_intersection)).squared_length())
                    {
                        edge = kernel::Segment_2(edge.source(), start_point);
                    }
                    else
                    {
                        edge = kernel::Segment_2(start_point, edge.target());
                    }

                    error = std::sqrt(CGAL::to_double(edge.squared_length()));
                }
                else
                {
                    error = 0.0;
                }
            }

            // Add first position to close the orbit
            periodic_orbit.push_back(periodic_orbit.front());

            return periodic_orbit;
        }

        Eigen::Vector2d periodic_orbits::linear_interpolate_position(const Eigen::Vector2d& left,
            const Eigen::Vector2d& right, const double value_left, const double value_right) const
        {
            const auto lambda = value_left / (value_left - value_right);

            return left + lambda * (right - left);
        }
    }
}
