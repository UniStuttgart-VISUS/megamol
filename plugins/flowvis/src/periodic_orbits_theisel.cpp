#include "stdafx.h"
#include "periodic_orbits_theisel.h"

#include "glyph_data_call.h"
#include "mesh_data_call.h"
#include "triangle_mesh_call.h"
#include "vector_field_call.h"

#include "flowvis/integrator.h"

#include "mmcore/Call.h"
#include "mmcore/DirectDataWriterCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/utility/DataHash.h"

#include "vislib/sys/Log.h"

#include "Eigen/Dense"

#include <CGAL/intersections.h>

#include "data/tpf_data_information.h"
#include "data/tpf_grid.h"
#include "data/tpf_grid_information.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

namespace megamol {
namespace flowvis {

periodic_orbits_theisel::periodic_orbits_theisel()
    : periodic_orbits_slot("periodic_orbits", "Computed periodic orbits as line glyphs")
    , stream_surface_slot("stream_surface", "Computed stream surfaces")
    , stream_surface_values_slot("stream_surface_values", "Values for coloring the stream surfaces")
    , result_writer_slot("result_writer_slot", "Results writer for storing periodic orbits")
    , vector_field_slot("vector_field_slot", "Vector field input")
    , seed_lines_slot("seed_lines", "Input seed lines for stream surfaces")
    , transfer_function("transfer_function", "Transfer function for coloring the stream surfaces")
    , integration_method("integration_method", "Method for streamline integration")
    , num_integration_steps("num_integration_steps", "Number of streamline integration steps")
    , integration_timestep("integration_timestep", "Initial time step for streamline integration")
    , max_integration_error("max_integration_error", "Maximum integration error for Runge-Kutta 4-5")
    , domain_height("domain_height", "Domain height coefficient for the stream surfaces")
    , num_seed_points("num_seed_points", "Number of seed points along a seed line")
    , num_subdivisions("num_subdivisions", "Number of subdivisions for seed line refinement")
    , critical_point_offset("critical_point_offset", "Offset from critical points for increased numeric stability")
    , direction("direction", "Integration direction for stream surface computation")
    , compute_intersections(
          "compute_intersections", "Compute intersection of stream surfaces for periodic orbit detection")
    , filter_seed_lines("filter_seed_lines", "Filter input seed lines used for computation")
    , vector_field_hash(-1)
    , vector_field_changed(false)
    , seed_lines_hash(-1)
    , seed_lines_changed(false)
    , stream_surface_hash(-1)
    , periodic_orbits_hash(-1) {

    // Connect output
    this->periodic_orbits_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(0),
        &periodic_orbits_theisel::get_periodic_orbits_data);
    this->periodic_orbits_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(1),
        &periodic_orbits_theisel::get_periodic_orbits_extent);
    this->MakeSlotAvailable(&this->periodic_orbits_slot);

    this->stream_surface_slot.SetCallback(triangle_mesh_call::ClassName(), triangle_mesh_call::FunctionName(0),
        &periodic_orbits_theisel::get_stream_surfaces_data);
    this->stream_surface_slot.SetCallback(triangle_mesh_call::ClassName(), triangle_mesh_call::FunctionName(1),
        &periodic_orbits_theisel::get_stream_surfaces_extent);
    this->MakeSlotAvailable(&this->stream_surface_slot);

    this->stream_surface_values_slot.SetCallback(mesh_data_call::ClassName(), mesh_data_call::FunctionName(0),
        &periodic_orbits_theisel::get_stream_surface_values_data);
    this->stream_surface_values_slot.SetCallback(mesh_data_call::ClassName(), mesh_data_call::FunctionName(1),
        &periodic_orbits_theisel::get_stream_surface_values_extent);
    this->MakeSlotAvailable(&this->stream_surface_values_slot);

    this->result_writer_slot.SetCallback(core::DirectDataWriterCall::ClassName(),
        core::DirectDataWriterCall::FunctionName(0), &periodic_orbits_theisel::get_writer_callback);
    this->MakeSlotAvailable(&this->result_writer_slot);
    this->get_writer = []() -> std::ostream& {
        static std::ostream dummy(nullptr);
        return dummy;
    };

    // Connect input
    this->vector_field_slot.SetCompatibleCall<vector_field_call::vector_field_description>();
    this->MakeSlotAvailable(&this->vector_field_slot);

    this->seed_lines_slot.SetCompatibleCall<glyph_data_call::glyph_data_description>();
    this->MakeSlotAvailable(&this->seed_lines_slot);

    // Create transfer function parameters
    this->transfer_function << new core::param::TransferFunctionParam("");
    this->MakeSlotAvailable(&this->transfer_function);

    // Create computation parameters
    this->integration_method << new core::param::EnumParam(0);
    this->integration_method.Param<core::param::EnumParam>()->SetTypePair(0, "Runge-Kutta 4 (fixed)");
    this->integration_method.Param<core::param::EnumParam>()->SetTypePair(1, "Runge-Kutta 4-5 (dynamic)");
    this->MakeSlotAvailable(&this->integration_method);

    this->num_integration_steps << new core::param::IntParam(100);
    this->MakeSlotAvailable(&this->num_integration_steps);

    this->integration_timestep << new core::param::FloatParam(0.01f);
    this->MakeSlotAvailable(&this->integration_timestep);

    this->max_integration_error << new core::param::FloatParam(0.000001f);
    this->MakeSlotAvailable(&this->max_integration_error);

    this->domain_height << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&domain_height);

    this->num_seed_points << new core::param::IntParam(10);
    this->MakeSlotAvailable(&this->num_seed_points);

    this->num_subdivisions << new core::param::IntParam(10);
    this->MakeSlotAvailable(&this->num_subdivisions);

    this->critical_point_offset << new core::param::FloatParam(0.5f);
    this->MakeSlotAvailable(&this->critical_point_offset);

    this->direction << new core::param::EnumParam(0);
    this->direction.Param<core::param::EnumParam>()->SetTypePair(0, "both");
    this->direction.Param<core::param::EnumParam>()->SetTypePair(1, "forward");
    this->direction.Param<core::param::EnumParam>()->SetTypePair(2, "backward");
    this->MakeSlotAvailable(&this->direction);

    this->compute_intersections << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->compute_intersections);

    this->filter_seed_lines << new core::param::IntParam(-1);
    this->MakeSlotAvailable(&this->filter_seed_lines);
}

periodic_orbits_theisel::~periodic_orbits_theisel() { this->Release(); }

bool periodic_orbits_theisel::create() { return true; }

void periodic_orbits_theisel::release() {}

bool periodic_orbits_theisel::get_input_data() {
    auto vfc_ptr = this->vector_field_slot.CallAs<vector_field_call>();
    auto slc_ptr = this->seed_lines_slot.CallAs<glyph_data_call>();

    if (vfc_ptr == nullptr || slc_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "The periodic_orbits_theisel module needs all input connections to be set properly");

        return false;
    }

    auto& vfc = *vfc_ptr;
    auto& slc = *slc_ptr;

    if (!(vfc(0) && slc(0))) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting data from vector field or glyph data source");

        return false;
    }

    if (vfc.DataHash() != this->vector_field_hash) {
        this->resolution = vfc.get_resolution();
        this->grid_positions = vfc.get_positions();
        this->vectors = vfc.get_vectors();

        this->vector_field_hash = vfc.DataHash();
        this->vector_field_changed = true;
    }

    if (slc.DataHash() != this->seed_lines_hash) {
        this->seed_lines = slc.get_line_segments();

        this->seed_lines_hash = slc.DataHash();
        this->seed_lines_changed = true;
    }

    return true;
}

bool periodic_orbits_theisel::get_input_extent() {
    auto vfc_ptr = this->vector_field_slot.CallAs<vector_field_call>();
    auto slc_ptr = this->seed_lines_slot.CallAs<glyph_data_call>();

    if (vfc_ptr == nullptr || slc_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "The periodic_orbits_theisel module needs all input connections to be set properly");

        return false;
    }

    auto& vfc = *vfc_ptr;
    auto& slc = *slc_ptr;

    if (!(vfc(1) && slc(1))) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting extents from vector field or glyph data source");

        return false;
    }

    this->bounding_rectangle = vfc.get_bounding_rectangle();

    const auto average_size = this->domain_height.Param<core::param::FloatParam>()->Value() * 0.5f *
                              (this->bounding_rectangle.Width() + this->bounding_rectangle.Height());

    this->bounding_box.Set(this->bounding_rectangle.GetLeft(), this->bounding_rectangle.GetBottom(), 0.0f,
        this->bounding_rectangle.GetRight(), this->bounding_rectangle.GetTop(), average_size);

    return true;
}

bool periodic_orbits_theisel::compute_periodic_orbits() {
    if (this->vector_field_changed || this->seed_lines_changed || this->direction.IsDirty() ||
        this->integration_method.IsDirty() || this->integration_timestep.IsDirty() ||
        this->max_integration_error.IsDirty() || this->domain_height.IsDirty() ||
        this->num_integration_steps.IsDirty() || this->num_seed_points.IsDirty() ||
        this->num_subdivisions.IsDirty() || this->compute_intersections.IsDirty() ||
        this->filter_seed_lines.IsDirty() || this->critical_point_offset.IsDirty()) {

        this->direction.ResetDirty();
        this->integration_method.ResetDirty();
        this->integration_timestep.ResetDirty();
        this->max_integration_error.ResetDirty();
        this->domain_height.ResetDirty();
        this->num_integration_steps.ResetDirty();
        this->num_seed_points.ResetDirty();
        this->num_subdivisions.ResetDirty();
        this->compute_intersections.ResetDirty();
        this->filter_seed_lines.ResetDirty();
        this->critical_point_offset.ResetDirty();

        // Create grid containing the vector field
        const auto vector_field = create_grid();

        // Get seed lines
        std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> seed_lines;
        Eigen::Vector2f last_critical_point;

        if (!this->seed_lines.empty()) {
            seed_lines.reserve(this->seed_lines.size());

            for (const auto& seed_line : this->seed_lines) {
                seed_lines.push_back(seed_line.first);
            }

            last_critical_point = seed_lines.back().second;
        } else {
            const auto& points = this->seed_lines_slot.CallAs<glyph_data_call>()->get_points();

            last_critical_point = points.front().first;
        }

        // Create seed line between last critical point and the nearest domain boundary
        const auto grid_origin = vector_field.get_node_coordinates(tpf::data::coords2_t(0, 0));
        const auto grid_diagonal = vector_field.get_node_coordinates(
            tpf::data::coords2_t(vector_field.get_extent()[0].second, vector_field.get_extent()[1].second));

        const auto distance_left = last_critical_point.x() - grid_origin.x();
        const auto distance_right = grid_diagonal.x() - last_critical_point.x();
        const auto distance_bottom = last_critical_point.y() - grid_origin.y();
        const auto distance_top = grid_diagonal.y() - last_critical_point.y();

        const auto x_dir = std::min(distance_left, distance_right) < std::min(distance_bottom, distance_top);

        if (x_dir) {
            if (distance_left < distance_right) {
                seed_lines.push_back(
                    std::make_pair(last_critical_point, Eigen::Vector2f(grid_origin.x(), last_critical_point.y())));
            } else {
                seed_lines.push_back(
                    std::make_pair(last_critical_point, Eigen::Vector2f(grid_diagonal.x(), last_critical_point.y())));
            }
        } else {
            if (distance_bottom < distance_top) {
                seed_lines.push_back(
                    std::make_pair(last_critical_point, Eigen::Vector2f(last_critical_point.x(), grid_origin.y())));
            } else {
                seed_lines.push_back(
                    std::make_pair(last_critical_point, Eigen::Vector2f(last_critical_point.x(), grid_diagonal.y())));
            }
        }

        // Offset seed line endpoints from critical points for more numerical stability
        const auto cp_offset = this->critical_point_offset.Param<core::param::FloatParam>()->Value();

        for (auto& seed_line : seed_lines) {
            const auto direction = (seed_line.second - seed_line.first).normalized();
            const auto offset =
                cp_offset * vector_field.get_cell_sizes(tpf::data::coords2_t(0, 0)).cwiseProduct(direction);

            seed_line.first += offset;
            seed_line.second -= offset;
        }

        // Allocate output
        const auto num_seed_points =
            static_cast<size_t>(std::max(2, this->num_seed_points.Param<core::param::IntParam>()->Value()));
        const auto num_triangles = 2 * (num_seed_points - 1);
        const auto num_integration_steps =
            static_cast<size_t>(this->num_integration_steps.Param<core::param::IntParam>()->Value());

        this->triangles = std::make_shared<std::vector<unsigned int>>();
        this->triangles->reserve(2 * seed_lines.size() * 3 * num_triangles * num_integration_steps);

        this->vertices = std::make_shared<std::vector<float>>();
        this->vertices->reserve(2 * seed_lines.size() * 3 * num_seed_points * (num_integration_steps + 1));

        this->seed_line_ids = std::make_shared<mesh_data_call::data_set>();
        this->seed_line_ids->data = std::make_shared<std::vector<float>>();
        this->seed_line_ids->data->reserve(2 * seed_lines.size() * num_seed_points * (num_integration_steps + 1));

        this->seed_point_ids = std::make_shared<mesh_data_call::data_set>();
        this->seed_point_ids->data = std::make_shared<std::vector<float>>();
        this->seed_point_ids->data->reserve(2 * seed_lines.size() * num_seed_points * (num_integration_steps + 1));

        this->integration_ids = std::make_shared<mesh_data_call::data_set>();
        this->integration_ids->data = std::make_shared<std::vector<float>>();
        this->integration_ids->data->reserve(2 * seed_lines.size() * num_seed_points * (num_integration_steps + 1));

        this->periodic_orbits.clear();

        // For each seed line, compute forward and backward stream surfaces
        std::size_t seed_index = 0;
        std::size_t seed_end = seed_lines.size();

        const auto filter_index = this->filter_seed_lines.Param<core::param::IntParam>()->Value();

        if (filter_index != -1) {
            seed_index = filter_index;
            seed_end = seed_index + 1;
        }

        for (; seed_index < seed_end; ++seed_index) {
            // Subdivide line and create a seed point per subdivision
            const auto height = this->bounding_box.Depth();

            const Eigen::Vector3f seed_line_start(
                seed_lines[seed_index].first.x(), seed_lines[seed_index].first.y(), 0.0f);
            const Eigen::Vector3f seed_line_end(
                seed_lines[seed_index].second.x(), seed_lines[seed_index].second.y(), height);

            const auto direction = (seed_line_end - seed_line_start).normalized();
            const auto step = (seed_line_end - seed_line_start).norm() / (num_seed_points - 1);

            std::vector<Eigen::Vector3f> forward_points, backward_points;
            forward_points.reserve(num_seed_points * (num_integration_steps + 1));
            backward_points.reserve(num_seed_points * (num_integration_steps + 1));

            std::vector<float> forward_timesteps, backward_timesteps;
            forward_timesteps.reserve(num_seed_points);
            backward_timesteps.reserve(num_seed_points);

            const float timestep = this->integration_timestep.Param<core::param::FloatParam>()->Value();

            for (int point_index = 0; point_index < num_seed_points; ++point_index) {
                const Eigen::Vector3f seed_point = seed_line_start + (point_index * step) * direction;

                forward_timesteps.push_back(timestep);
                backward_timesteps.push_back(timestep);

                forward_points.push_back(seed_point);
                backward_points.push_back(seed_point);
            }

            // Advect both set of points
            auto previous_forward_points = forward_points;
            auto previous_backward_points = backward_points;

            std::vector<Eigen::Vector3f> advected_forward_points, advected_backward_points;
            advected_forward_points.reserve(num_seed_points);
            advected_backward_points.reserve(num_seed_points);

            for (std::size_t integration = 0; integration < num_integration_steps; ++integration) {
                // Advect forward stream surface
                advected_forward_points.clear();

                for (std::size_t point_index = 0; point_index < previous_forward_points.size(); ++point_index) {
                    advected_forward_points.push_back(advect_point(
                        vector_field, previous_forward_points[point_index], forward_timesteps[point_index], true));
                }

                forward_points.insert(forward_points.end(), advected_forward_points.begin(), advected_forward_points.end());

                // Look for an intersection
                if (this->compute_intersections.Param<core::param::BoolParam>()->Value() && num_integration_steps > 1 &&
                    this->direction.Param<core::param::EnumParam>()->Value() == 0) {

                    const auto intersection_points = find_intersection(previous_forward_points,
                        previous_backward_points, advected_forward_points, advected_backward_points);

                    if (!intersection_points.empty()) {
                        const auto num_subdivisions = this->num_subdivisions.Param<core::param::IntParam>()->Value();

                        if (num_subdivisions > 0) {
                            const auto refined_intersections =
                                refine_intersections(vector_field, seed_line_start, direction, step, timestep,
                                    integration, previous_forward_points, previous_backward_points,
                                    advected_forward_points, advected_backward_points, intersection_points);

                            for (const auto& refined_intersection : refined_intersections) {
                                this->periodic_orbits.push_back(std::make_pair(0.0f, refined_intersection));
                            }
                        } else {
                            for (const auto& intersection_point : intersection_points) {
                                this->periodic_orbits.push_back(std::make_pair(0.0f, std::get<0>(intersection_point)));
                            }
                        }
                    }
                }

                std::swap(previous_forward_points, advected_forward_points);

                // Advect backward stream surface
                advected_backward_points.clear();

                for (std::size_t point_index = 0; point_index < previous_backward_points.size(); ++point_index) {
                    advected_backward_points.push_back(advect_point(
                        vector_field, previous_backward_points[point_index], backward_timesteps[point_index], false));
                }

                backward_points.insert(backward_points.end(), advected_backward_points.begin(), advected_backward_points.end());

                // Look for an intersection
                if (this->compute_intersections.Param<core::param::BoolParam>()->Value() && num_integration_steps > 1 &&
                    this->direction.Param<core::param::EnumParam>()->Value() == 0) {

                    const auto intersection_points = find_intersection(previous_forward_points, previous_backward_points,
                        advected_forward_points, advected_backward_points);

                    if (!intersection_points.empty()) {
                        const auto num_subdivisions = this->num_subdivisions.Param<core::param::IntParam>()->Value();

                        if (num_subdivisions > 0) {
                            const auto refined_intersections =
                                refine_intersections(vector_field, seed_line_start, direction, step, timestep,
                                    integration, previous_forward_points, previous_backward_points,
                                    advected_forward_points, advected_backward_points, intersection_points);

                            for (const auto& refined_intersection : refined_intersections) {
                                this->periodic_orbits.push_back(std::make_pair(0.0f, refined_intersection));
                            }
                        } else {
                            for (const auto& intersection_point : intersection_points) {
                                this->periodic_orbits.push_back(std::make_pair(0.0f, std::get<0>(intersection_point)));
                            }
                        }
                    }
                }

                std::swap(previous_backward_points, advected_backward_points);
            }

            // Store vertices and indices in a GL-friendly manner
            const auto integration_direction = this->direction.Param<core::param::EnumParam>()->Value();

            if (integration_direction == 0 || integration_direction == 1) {
                std::size_t point_id = 0;

                for (const auto& forward_point : forward_points) {
                    this->vertices->push_back(forward_point.x());
                    this->vertices->push_back(forward_point.y());
                    this->vertices->push_back(forward_point.z());

                    this->seed_line_ids->data->push_back(
                        (integration_direction == 0 ? 0.5f : 1.0f) * static_cast<float>(seed_index));
                    this->seed_point_ids->data->push_back(static_cast<float>(point_id % num_seed_points));
                    this->integration_ids->data->push_back(static_cast<float>(point_id / num_seed_points));

                    ++point_id;
                }
            }

            if (integration_direction == 0 || integration_direction == 2) {
                std::size_t point_id = 0;

                for (const auto& backward_point : backward_points) {
                    this->vertices->push_back(backward_point.x());
                    this->vertices->push_back(backward_point.y());
                    this->vertices->push_back(backward_point.z());

                    this->seed_line_ids->data->push_back(
                        (integration_direction == 0 ? 0.5f * seed_lines.size() : 0.0f) +
                        (integration_direction == 0 ? 0.5f : 1.0f) * static_cast<float>(seed_index));
                    this->seed_point_ids->data->push_back(static_cast<float>(point_id % num_seed_points));
                    this->integration_ids->data->push_back(static_cast<float>(point_id / num_seed_points));

                    ++point_id;
                }
            }

            // Create surface mesh
            auto seed_index_filtered = seed_index;

            if (filter_index != -1) {
                seed_index_filtered = 0;
            }

            if (integration_direction == 1 || integration_direction == 2) {
                const auto seed_line_offset = seed_index_filtered * (num_integration_steps + 1) * num_seed_points;

                for (std::size_t integration = 0; integration < num_integration_steps; ++integration) {
                    const auto integration_offset = seed_line_offset + integration * num_seed_points;

                    for (std::size_t point_index = 0; point_index < num_seed_points - 1; ++point_index) {
                        const auto point_offset = integration_offset + point_index;

                        this->triangles->push_back(static_cast<unsigned int>(point_offset));
                        this->triangles->push_back(static_cast<unsigned int>(point_offset + num_seed_points));
                        this->triangles->push_back(static_cast<unsigned int>(point_offset + num_seed_points + 1));

                        this->triangles->push_back(static_cast<unsigned int>(point_offset));
                        this->triangles->push_back(static_cast<unsigned int>(point_offset + num_seed_points + 1));
                        this->triangles->push_back(static_cast<unsigned int>(point_offset + 1));
                    }
                }
            }

            if (integration_direction == 0) {
                const auto seed_line_offset_forward =
                    2 * seed_index_filtered * (num_integration_steps + 1) * num_seed_points;

                for (std::size_t integration = 0; integration < num_integration_steps; ++integration) {
                    const auto integration_offset = seed_line_offset_forward + integration * num_seed_points;

                    for (std::size_t point_index = 0; point_index < num_seed_points - 1; ++point_index) {
                        const auto point_offset = integration_offset + point_index;

                        this->triangles->push_back(static_cast<unsigned int>(point_offset));
                        this->triangles->push_back(static_cast<unsigned int>(point_offset + num_seed_points));
                        this->triangles->push_back(static_cast<unsigned int>(point_offset + num_seed_points + 1));

                        this->triangles->push_back(static_cast<unsigned int>(point_offset));
                        this->triangles->push_back(static_cast<unsigned int>(point_offset + num_seed_points + 1));
                        this->triangles->push_back(static_cast<unsigned int>(point_offset + 1));
                    }
                }

                const auto seed_line_offset_backward =
                    (2 * seed_index_filtered + 1) * (num_integration_steps + 1) * num_seed_points;

                for (std::size_t integration = 0; integration < num_integration_steps; ++integration) {
                    const auto integration_offset = seed_line_offset_backward + integration * num_seed_points;

                    for (std::size_t point_index = 0; point_index < num_seed_points - 1; ++point_index) {
                        const auto point_offset = integration_offset + point_index;

                        this->triangles->push_back(static_cast<unsigned int>(point_offset));
                        this->triangles->push_back(static_cast<unsigned int>(point_offset + num_seed_points));
                        this->triangles->push_back(static_cast<unsigned int>(point_offset + num_seed_points + 1));

                        this->triangles->push_back(static_cast<unsigned int>(point_offset));
                        this->triangles->push_back(static_cast<unsigned int>(point_offset + num_seed_points + 1));
                        this->triangles->push_back(static_cast<unsigned int>(point_offset + 1));
                    }
                }
            }
        }

        if (this->compute_intersections.Param<core::param::BoolParam>()->Value() && num_integration_steps > 1 &&
            this->direction.Param<core::param::EnumParam>()->Value() == 0) {

            vislib::sys::Log::DefaultLog.WriteInfo("Found %u periodic orbits.", this->periodic_orbits.size());
        }

        this->stream_surface_hash = core::utility::DataHash(this->vector_field_hash, this->seed_lines_hash,
            this->direction.Param<core::param::EnumParam>()->Value(),
            this->integration_method.Param<core::param::EnumParam>()->Value(),
            this->integration_timestep.Param<core::param::FloatParam>()->Value(),
            this->max_integration_error.Param<core::param::FloatParam>()->Value(),
            this->num_integration_steps.Param<core::param::IntParam>()->Value(),
            this->num_seed_points.Param<core::param::IntParam>()->Value(),
            this->num_subdivisions.Param<core::param::IntParam>()->Value(),
            this->filter_seed_lines.Param<core::param::IntParam>()->Value(),
            this->critical_point_offset.Param<core::param::FloatParam>()->Value());

        this->periodic_orbits_hash = core::utility::DataHash(
            this->stream_surface_hash, this->compute_intersections.Param<core::param::BoolParam>()->Value());
    }

    this->vector_field_changed = false;
    this->seed_lines_changed = false;

    return true;
}

tpf::data::grid<float, float, 2, 2> periodic_orbits_theisel::create_grid() const {

    // Get extent
    tpf::data::extent_t extent;
    extent.push_back(std::make_pair(0ull, static_cast<std::size_t>(this->resolution[0] - 1)));
    extent.push_back(std::make_pair(0ull, static_cast<std::size_t>(this->resolution[1] - 1)));

    // Compute cell and node coordinates
    const Eigen::Vector2f origin((*this->grid_positions)[0], (*this->grid_positions)[1]);
    const Eigen::Vector2f cell_diagonal(
        (*this->grid_positions)[2], (*this->grid_positions)[2 * static_cast<std::size_t>(this->resolution[0]) + 1]);

    const auto cell_size = cell_diagonal - origin;
    const auto node_origin = origin - 0.5f * cell_size;

    tpf::data::grid_information<float>::array_type cell_coordinates(2), node_coordinates(2), cell_sizes(2);

    for (std::size_t dimension = 0; dimension < 2; ++dimension) {
        cell_coordinates[dimension].resize(this->resolution[dimension]);
        node_coordinates[dimension].resize(static_cast<std::size_t>(this->resolution[dimension]) + 1);
        cell_sizes[dimension].resize(this->resolution[dimension]);

        for (std::size_t element = 0; element < this->resolution[dimension]; ++element) {
            cell_sizes[dimension][element] = cell_size[dimension];
            cell_coordinates[dimension][element] = origin[dimension] + element * cell_size[dimension];
            node_coordinates[dimension][element] = origin[dimension] + (element - 0.5f) * cell_size[dimension];
        }

        node_coordinates[dimension][this->resolution[dimension]] =
            origin[dimension] + (this->resolution[dimension] - 0.5f) * cell_size[dimension];
    }

    // Create grid
    return tpf::data::grid<float, float, 2, 2>("vector_field", extent, *this->vectors, std::move(cell_coordinates),
        std::move(node_coordinates), std::move(cell_sizes));
}

Eigen::Vector3f periodic_orbits_theisel::advect_point(const tpf::data::grid<float, float, 2, 2>& grid,
    const Eigen::Vector3f& point, float& delta, const bool forward) const {

    if (delta == 0.0f) return point;

    // Sanity check
    const auto min_x = 0;
    const auto min_y = 0;
    const auto max_x = this->resolution[0] - 1;
    const auto max_y = this->resolution[1] - 1;

    if (point[0] < grid.get_cell_coordinates(min_x, 0) || point[1] < grid.get_cell_coordinates(min_y, 1) ||
        point[0] > grid.get_cell_coordinates(max_x, 0) || point[1] > grid.get_cell_coordinates(max_y, 1)) {

        delta = 0.0f;
        return point;
    }

    // Advect
    Eigen::Vector2f advected_point = point.head<2>();

    try {
        switch (this->integration_method.Param<core::param::EnumParam>()->Value()) {
        case 0:
            advect_point_rk4<2>(grid, advected_point, delta, forward);
            break;
        case 1:
            advect_point_rk45<2>(grid, advected_point, delta,
                this->max_integration_error.Param<core::param::FloatParam>()->Value(), forward);
            break;
        default:
            vislib::sys::Log::DefaultLog.WriteError("Unknown advection method selected");
        }
    } catch (const std::runtime_error&) {
        vislib::sys::Log::DefaultLog.WriteWarn("Interpolation yielded no movement or overshooting");
        delta = 0.0f;
    }

    return Eigen::Vector3f(advected_point[0], advected_point[1], point[2]);
}

std::vector<std::tuple<Eigen::Vector2f, std::size_t, std::size_t>> periodic_orbits_theisel::find_intersection(
    const std::vector<Eigen::Vector3f>& previous_forward_points,
    const std::vector<Eigen::Vector3f>& previous_backward_points,
    const std::vector<Eigen::Vector3f>& advected_forward_points,
    const std::vector<Eigen::Vector3f>& advected_backward_points) const {

    const auto num_seed_points = previous_forward_points.size();

    std::vector<std::tuple<Eigen::Vector2f, std::size_t, std::size_t>> intersections;

    for (std::size_t fwd_point_index = 0; fwd_point_index < num_seed_points - 1; ++fwd_point_index) {
        const kernel_t::Point_3 fwd_point_1(previous_forward_points[fwd_point_index][0],
            previous_forward_points[fwd_point_index][1], previous_forward_points[fwd_point_index][2]);
        const kernel_t::Point_3 fwd_point_2(previous_forward_points[fwd_point_index + 1][0],
            previous_forward_points[fwd_point_index + 1][1], previous_forward_points[fwd_point_index + 1][2]);
        const kernel_t::Point_3 fwd_point_3(advected_forward_points[fwd_point_index][0],
            advected_forward_points[fwd_point_index][1], advected_forward_points[fwd_point_index][2]);
        const kernel_t::Point_3 fwd_point_4(advected_forward_points[fwd_point_index + 1][0],
            advected_forward_points[fwd_point_index + 1][1], advected_forward_points[fwd_point_index + 1][2]);

        const kernel_t::Triangle_3 fwd_triangle_1(fwd_point_1, fwd_point_3, fwd_point_4);
        const kernel_t::Triangle_3 fwd_triangle_2(fwd_point_1, fwd_point_4, fwd_point_2);

        for (std::size_t bwd_point_index = 0; bwd_point_index < num_seed_points - 1; ++bwd_point_index) {
            const kernel_t::Point_3 bwd_point_1(previous_backward_points[bwd_point_index][0],
                previous_backward_points[bwd_point_index][1], previous_backward_points[bwd_point_index][2]);
            const kernel_t::Point_3 bwd_point_2(previous_backward_points[bwd_point_index + 1][0],
                previous_backward_points[bwd_point_index + 1][1], previous_backward_points[bwd_point_index + 1][2]);
            const kernel_t::Point_3 bwd_point_3(advected_backward_points[bwd_point_index][0],
                advected_backward_points[bwd_point_index][1], advected_backward_points[bwd_point_index][2]);
            const kernel_t::Point_3 bwd_point_4(advected_backward_points[bwd_point_index + 1][0],
                advected_backward_points[bwd_point_index + 1][1], advected_backward_points[bwd_point_index + 1][2]);

            const kernel_t::Triangle_3 bwd_triangle_1(bwd_point_1, bwd_point_3, bwd_point_4);
            const kernel_t::Triangle_3 bwd_triangle_2(bwd_point_1, bwd_point_4, bwd_point_2);

            decltype(CGAL::intersection(fwd_triangle_1, bwd_triangle_1)) intersection;

            if (CGAL::do_intersect(fwd_triangle_1, bwd_triangle_1)) {
                intersection = CGAL::intersection(fwd_triangle_1, bwd_triangle_1);
            } else if (CGAL::do_intersect(fwd_triangle_1, bwd_triangle_2)) {
                intersection = CGAL::intersection(fwd_triangle_1, bwd_triangle_2);
            } else if (CGAL::do_intersect(fwd_triangle_2, bwd_triangle_1)) {
                intersection = CGAL::intersection(fwd_triangle_2, bwd_triangle_1);
            } else if (CGAL::do_intersect(fwd_triangle_2, bwd_triangle_2)) {
                intersection = CGAL::intersection(fwd_triangle_2, bwd_triangle_2);
            }

            if (intersection) {
                const auto intersection_point = boost::get<kernel_t::Point_3>(&*intersection);
                const auto intersection_line = boost::get<kernel_t::Segment_3>(&*intersection);
                const auto intersection_triangle = boost::get<kernel_t::Triangle_3>(&*intersection);
                const auto intersection_points = boost::get<std::vector<kernel_t::Point_3>>(&*intersection);

                if (intersection_point != nullptr) {
                    // Not relevant
                } else if (intersection_line != nullptr) {
                    const auto point = Eigen::Vector2f(CGAL::to_double(intersection_line->vertex(0)[0]),
                        CGAL::to_double(intersection_line->vertex(0)[1]));

                    intersections.push_back(std::make_tuple(point, fwd_point_index, bwd_point_index));
                } else if (intersection_triangle != nullptr) {
                    vislib::sys::Log::DefaultLog.WriteWarn("Triangle result from triangle intersection not supported");
                } else if (intersection_points != nullptr) {
                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "Vector of points result from triangle intersection not supported");
                }
            }
        }
    }

    return intersections;
}

std::vector<Eigen::Vector2f> periodic_orbits_theisel::refine_intersections(
    const tpf::data::grid<float, float, 2, 2>& vector_field, const Eigen::Vector3f& seed_line_start,
    const Eigen::Vector3f& seed_line_direction, const float seed_line_step, const float timestep,
    const std::size_t integration, const std::vector<Eigen::Vector3f>& previous_forward_points,
    const std::vector<Eigen::Vector3f>& previous_backward_points,
    const std::vector<Eigen::Vector3f>& advected_forward_points,
    const std::vector<Eigen::Vector3f>& advected_backward_points,
    const std::vector<std::tuple<Eigen::Vector2f, std::size_t, std::size_t>>& intersections) const {

    const auto num_subdivisions = this->num_subdivisions.Param<core::param::IntParam>()->Value();

    std::vector<Eigen::Vector2f> refined_intersections;

    for (const auto& intersection_point : intersections) {
        // Seed a new point for each subdivision and check intersection again
        float step_fwd_low = static_cast<float>(std::get<1>(intersection_point));
        float step_fwd_hi = static_cast<float>(std::get<1>(intersection_point) + 1);
        float step_bwd_low = static_cast<float>(std::get<2>(intersection_point));
        float step_bwd_hi = static_cast<float>(std::get<2>(intersection_point) + 1);

        Eigen::Vector3f fwd_prev_low = previous_forward_points[std::get<1>(intersection_point)];
        Eigen::Vector3f fwd_prev_hi = previous_forward_points[std::get<1>(intersection_point) + 1];
        Eigen::Vector3f fwd_adv_low = advected_forward_points[std::get<1>(intersection_point)];
        Eigen::Vector3f fwd_adv_hi = advected_forward_points[std::get<1>(intersection_point) + 1];
        Eigen::Vector3f bwd_prev_low = previous_backward_points[std::get<2>(intersection_point)];
        Eigen::Vector3f bwd_prev_hi = previous_backward_points[std::get<2>(intersection_point) + 1];
        Eigen::Vector3f bwd_adv_low = advected_backward_points[std::get<2>(intersection_point)];
        Eigen::Vector3f bwd_adv_hi = advected_backward_points[std::get<2>(intersection_point) + 1];

        for (std::size_t subdivision = 0; subdivision < num_subdivisions; ++subdivision) {
            const float step_fwd_ref = 0.5f * (step_fwd_low + step_fwd_hi);
            const float step_bwd_ref = 0.5f * (step_bwd_low + step_bwd_hi);

            Eigen::Vector3f seed_point_fwd = seed_line_start + (step_fwd_ref * seed_line_step) * seed_line_direction;
            Eigen::Vector3f seed_point_bwd = seed_line_start + (step_bwd_ref * seed_line_step) * seed_line_direction;

            float delta_fwd = timestep;
            float delta_bwd = timestep;

            // Advect new seed points
            for (std::size_t ref_integration = 0; ref_integration < integration; ++ref_integration) {
                seed_point_fwd = advect_point(vector_field, seed_point_fwd, delta_fwd, true);
                seed_point_bwd = advect_point(vector_field, seed_point_bwd, delta_bwd, false);
            }

            const Eigen::Vector3f adv_point_fwd = advect_point(vector_field, seed_point_fwd, delta_fwd, true);
            const Eigen::Vector3f adv_point_bwd = advect_point(vector_field, seed_point_bwd, delta_bwd, true);

            // Check intersection
            const std::vector<Eigen::Vector3f> fwd_prev{fwd_prev_low, seed_point_fwd, fwd_prev_hi};
            const std::vector<Eigen::Vector3f> bwd_prev{bwd_prev_low, seed_point_bwd, bwd_prev_hi};
            const std::vector<Eigen::Vector3f> fwd_adv{fwd_adv_low, adv_point_fwd, fwd_adv_hi};
            const std::vector<Eigen::Vector3f> bwd_adv{bwd_adv_low, adv_point_bwd, bwd_adv_hi};

            const auto ref_intersection_points = find_intersection(fwd_prev, bwd_prev, fwd_adv, bwd_adv);

            if (ref_intersection_points.empty()) {
                break;
            }

            if (subdivision == (static_cast<long long>(num_subdivisions) - 1)) {
                refined_intersections.push_back(std::get<0>(ref_intersection_points[0]));
                break;
            }

            // Prepare for next subdivision
            if (std::get<1>(ref_intersection_points[0]) == 0) {
                step_fwd_hi = step_fwd_ref;

                fwd_prev_hi = seed_point_fwd;
                fwd_adv_hi = adv_point_fwd;
            } else {
                step_fwd_low = step_fwd_ref;

                fwd_prev_low = seed_point_fwd;
                fwd_adv_low = adv_point_fwd;
            }

            if (std::get<2>(ref_intersection_points[0]) == 0) {
                step_bwd_hi = step_bwd_ref;

                bwd_prev_hi = seed_point_bwd;
                bwd_adv_hi = adv_point_bwd;
            } else {
                step_bwd_low = step_bwd_ref;

                bwd_prev_low = seed_point_bwd;
                bwd_adv_low = adv_point_bwd;
            }
        }
    }

    return refined_intersections;
}

bool periodic_orbits_theisel::get_periodic_orbits_data(core::Call& call) {
    auto& gdc = static_cast<glyph_data_call&>(call);

    if (!(get_input_data() && compute_periodic_orbits())) {
        return false;
    }

    if (gdc.DataHash() != this->periodic_orbits_hash) {
        gdc.clear();

        for (const auto& point : this->periodic_orbits) {
            gdc.add_point(point.second, point.first);
        }

        gdc.SetDataHash(this->periodic_orbits_hash);
    }

    return true;
}

bool periodic_orbits_theisel::get_periodic_orbits_extent(core::Call& call) {
    auto& gdc = static_cast<glyph_data_call&>(call);

    if (!get_input_extent()) {
        return false;
    }

    gdc.set_bounding_rectangle(this->bounding_rectangle);

    return true;
}

bool periodic_orbits_theisel::get_stream_surfaces_data(core::Call& call) {
    auto& tmc = static_cast<triangle_mesh_call&>(call);

    if (!(get_input_data() && compute_periodic_orbits())) {
        return false;
    }

    if (tmc.DataHash() != this->stream_surface_hash) {
        tmc.set_vertices(this->vertices);
        tmc.set_indices(this->triangles);

        tmc.SetDataHash(this->stream_surface_hash);
    }

    return true;
}

bool periodic_orbits_theisel::get_stream_surfaces_extent(core::Call& call) {
    auto& tmc = static_cast<triangle_mesh_call&>(call);

    if (!get_input_extent()) {
        return false;
    }

    tmc.set_dimension(triangle_mesh_call::dimension_t::THREE);
    tmc.set_bounding_box(this->bounding_box);

    return true;
}

bool periodic_orbits_theisel::get_stream_surface_values_data(core::Call& call) {
    auto& mdc = static_cast<mesh_data_call&>(call);

    if (!(get_input_data() && compute_periodic_orbits())) {
        return false;
    }

    if (mdc.DataHash() != this->stream_surface_hash || this->transfer_function.IsDirty()) {
        const auto tf_string = this->transfer_function.Param<core::param::TransferFunctionParam>()->Value();

        this->seed_line_ids->min_value = 0.0f;
        this->seed_line_ids->max_value = static_cast<float>(this->seed_lines.size());
        this->seed_line_ids->transfer_function = tf_string;
        this->seed_line_ids->transfer_function_dirty = true;
        mdc.set_data("seed line", this->seed_line_ids);

        this->seed_point_ids->min_value = 0.0f;
        this->seed_point_ids->max_value =
            static_cast<float>(std::max(2, this->num_seed_points.Param<core::param::IntParam>()->Value()));
        this->seed_point_ids->transfer_function = tf_string;
        this->seed_point_ids->transfer_function_dirty = true;
        mdc.set_data("seed point", this->seed_point_ids);

        this->integration_ids->min_value = 0.0f;
        this->integration_ids->max_value =
            static_cast<float>(this->num_integration_steps.Param<core::param::IntParam>()->Value());
        this->integration_ids->transfer_function = tf_string;
        this->integration_ids->transfer_function_dirty = true;
        mdc.set_data("integration", this->integration_ids);

        mdc.SetDataHash(this->stream_surface_hash);

        this->transfer_function.ResetDirty();
    }

    return true;
}

bool periodic_orbits_theisel::get_stream_surface_values_extent(core::Call& call) {
    auto& mdc = static_cast<mesh_data_call&>(call);

    if (!get_input_extent()) {
        return false;
    }

    mdc.set_data("seed line");
    mdc.set_data("seed point");
    mdc.set_data("integration");

    return true;
}

bool periodic_orbits_theisel::get_writer_callback(core::Call& call) {
    auto& ddwc = static_cast<core::DirectDataWriterCall&>(call);

    this->get_writer = ddwc.GetCallback();

    return true;
}

} // namespace flowvis
} // namespace megamol
