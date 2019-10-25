#include "stdafx.h"
#include "periodic_orbits_theisel.h"

#include "glyph_data_call.h"
#include "triangle_mesh_call.h"
#include "vector_field_call.h"

#include "mmcore/Call.h"
#include "mmcore/DirectDataWriterCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/TransferFunctionParam.h"

#include "vislib/sys/Log.h"

#include "Eigen/Dense"

#include "data/tpf_grid.h"
#include "data/tpf_grid_information.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace megamol {
namespace flowvis {

periodic_orbits_theisel::periodic_orbits_theisel()
    : periodic_orbits_slot("periodic_orbits", "Computed periodic orbits as line glyphs")
    , stream_surface_slot("stream_surface", "Computed stream surfaces")
    , seed_line_slot("seed_line", "Computed seed lines")
    , result_writer_slot("result_writer_slot", "Results writer for storing periodic orbits")
    , vector_field_slot("vector_field_slot", "Vector field input")
    , critical_points_slot("critical_points", "Critical points input")
    , transfer_function("transfer_function", "Transfer function for coloring the stream surfaces")
    , integration_method("integration_method", "Method for stream line integration")
    , num_integration_steps("num_integration_steps", "Number of stream line integration steps")
    , integration_timestep("integration_timestep", "Initial time step for stream line integration")
    , max_integration_error("max_integration_error", "Maximum integration error for Runge-Kutta 4-5")
    , num_subdivisions("num_subdivisions", "Number of subdivisions")
    , vector_field_hash(-1)
    , critical_points_hash(-1)
    , stream_surface_hash(-1)
    , periodic_orbits_hash(-1)
    , seed_line_hash(-1) {

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

    this->seed_line_slot.SetCallback(
        glyph_data_call::ClassName(), glyph_data_call::FunctionName(0), &periodic_orbits_theisel::get_seed_lines_data);
    this->seed_line_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(1),
        &periodic_orbits_theisel::get_seed_lines_extent);
    this->MakeSlotAvailable(&this->seed_line_slot);

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

    this->critical_points_slot.SetCompatibleCall<glyph_data_call::glyph_data_description>();
    this->MakeSlotAvailable(&this->critical_points_slot);

    // Create transfer function parameters
    this->transfer_function << new core::param::TransferFunctionParam("");
    this->MakeSlotAvailable(&this->transfer_function);

    // Create computation parameters
    this->integration_method << new core::param::EnumParam(0);
    this->integration_method.Param<core::param::EnumParam>()->SetTypePair(0, "Runge-Kutta 4 (fixed)");
    this->integration_method.Param<core::param::EnumParam>()->SetTypePair(1, "Runge-Kutta 4-5 (dynamic)");
    this->MakeSlotAvailable(&this->integration_method);

    this->num_integration_steps << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->num_integration_steps);

    this->integration_timestep << new core::param::FloatParam(0.01f);
    this->MakeSlotAvailable(&this->integration_timestep);

    this->max_integration_error << new core::param::FloatParam(0.000001f);
    this->MakeSlotAvailable(&this->max_integration_error);

    this->num_subdivisions << new core::param::IntParam(10);
    this->MakeSlotAvailable(&this->num_subdivisions);
}

periodic_orbits_theisel::~periodic_orbits_theisel() { this->Release(); }

bool periodic_orbits_theisel::create() { return true; }

void periodic_orbits_theisel::release() {}

bool periodic_orbits_theisel::get_input_data() {
    auto vfc_ptr = this->vector_field_slot.CallAs<vector_field_call>();
    auto cpc_ptr = this->critical_points_slot.CallAs<glyph_data_call>();

    if (vfc_ptr == nullptr || cpc_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "The periodic_orbits_theisel module needs all input connections to be set properly");

        return false;
    }

    auto& vfc = *vfc_ptr;
    auto& cpc = *cpc_ptr;

    if (!(vfc(0) && cpc(0))) {
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

    if (cpc.DataHash() != this->critical_points_hash) {
        this->vertices = cpc.get_line_vertices();
        this->lines = cpc.get_line_indices();

        this->critical_points_hash = cpc.DataHash();
        this->critical_points_changed = true;
    }

    return true;
}

bool periodic_orbits_theisel::get_input_extent() {
    auto vfc_ptr = this->vector_field_slot.CallAs<vector_field_call>();
    auto cpc_ptr = this->critical_points_slot.CallAs<glyph_data_call>();

    if (vfc_ptr == nullptr || cpc_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "The periodic_orbits_theisel module needs all input connections to be set properly");

        return false;
    }

    auto& vfc = *vfc_ptr;
    auto& cpc = *cpc_ptr;

    if (!(vfc(1) && cpc(1))) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting extents from vector field or glyph data source");

        return false;
    }

    this->bounding_rectangle = vfc.get_bounding_rectangle();

    const auto average_size = 0.5f * (this->bounding_rectangle.Width() + this->bounding_rectangle.Height());

    this->bounding_box.Set(this->bounding_rectangle.GetLeft(), 0.0f, this->bounding_rectangle.GetTop(),
        this->bounding_rectangle.GetRight(), average_size, this->bounding_rectangle.GetBottom());

    return true;
}

bool periodic_orbits_theisel::compute_periodic_orbits() {
    if (this->vector_field_changed || this->critical_points_changed) {
        // Create grid containing the vector field
        this->grid_positions;
        this->vectors;

        tpf::data::extent_t extent;
        extent.push_back(std::make_pair(0ull, static_cast<std::size_t>(this->resolution[0] - 1)));
        extent.push_back(std::make_pair(0ull, static_cast<std::size_t>(this->resolution[1] - 1)));

        const Eigen::Vector2f origin((*this->grid_positions)[0], (*this->grid_positions)[1]);
        const Eigen::Vector2f right((*this->grid_positions)[2], (*this->grid_positions)[3]);
        const Eigen::Vector2f up((*this->grid_positions)[2 * this->resolution[0]], (*this->grid_positions)[2 * this->resolution[0] + 1]);

        const Eigen::Vector2f cell_size(right.x() - origin.x(), up.y() - origin.y());
        const auto node_origin = origin - 0.5f * cell_size;

        tpf::data::grid_information<float>::array_type cell_coordinates(2), node_coordinates(2), cell_sizes(2);

        for (std::size_t dimension = 0; dimension < 2; ++dimension) {
            cell_coordinates[dimension].resize(this->resolution[dimension]);
            node_coordinates[dimension].resize(this->resolution[dimension] + 1);
            cell_sizes[dimension].resize(this->resolution[dimension]);

            for (std::size_t element = 0; element < this->resolution[dimension]; ++element) {
                cell_sizes[dimension][element] = cell_size[dimension];
                cell_coordinates[dimension][element] = origin[dimension] + element * cell_size[dimension];
                node_coordinates[dimension][element] = origin[dimension] + (element - 0.5f) * cell_size[dimension];
            }

            node_coordinates[dimension][this->resolution[dimension]] =
                origin[dimension] + (this->resolution[dimension] - 0.5f) * cell_size[dimension];
        }


        tpf::data::grid<float, float, 2, 2> vector_field("vector_field", extent, *this->vectors,
            std::move(cell_coordinates), std::move(node_coordinates), std::move(cell_sizes));

        // Create seed lines
        std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> seed_lines;

        // TODO
        // DEBUG

        seed_lines.push_back(
            std::make_pair(Eigen::Vector2f(0.0701957, 0.00664742), Eigen::Vector2f(0.0825285, 0.0217574)));

        // DEBUG

        // For each seed line, compute forward and backward stream surfaces
        const auto num_subdivisions =
            static_cast<size_t>(std::max(1, this->num_subdivisions.Param<core::param::IntParam>()->Value()));
        const auto num_integration_steps =
            static_cast<size_t>(this->num_integration_steps.Param<core::param::IntParam>()->Value());

        this->triangles = std::make_shared<std::vector<unsigned int>>();
        this->triangles->reserve(2 * seed_lines.size() * num_subdivisions * 2 * 3 * num_integration_steps);

        this->mesh_vertices = std::make_shared<std::vector<float>>();
        this->mesh_vertices->reserve(2 * seed_lines.size() * 3 * num_subdivisions * (num_integration_steps + 1));

        this->seed_lines.clear();
        this->seed_lines.reserve(seed_lines.size());

        this->periodic_orbits.clear();

        for (const auto& seed_line : seed_lines) {
            this->seed_lines.push_back(std::make_pair(0.0f, std::vector<Eigen::Vector2f>{ seed_line.first, seed_line.second }));

            // Subdivide line and create a seed point per subdivision
            const auto height = this->bounding_box.Height();

            const Eigen::Vector3f seed_line_start(seed_line.first.x(), 0.0f, seed_line.first.y());
            const Eigen::Vector3f seed_line_end(seed_line.second.x(), height, seed_line.second.y());

            const auto direction = (seed_line_end - seed_line_start).normalized();
            const auto step = (seed_line_end - seed_line_start).norm() / num_subdivisions;

            std::vector<Eigen::Vector3f> forward_points, backward_points;
            forward_points.reserve(num_subdivisions * (num_integration_steps + 1));
            backward_points.reserve(num_subdivisions * (num_integration_steps + 1));

            std::vector<float> forward_timesteps, backward_timesteps;
            forward_timesteps.reserve(num_subdivisions);
            backward_timesteps.reserve(num_subdivisions);

            const float timestep = this->integration_timestep.Param<core::param::FloatParam>()->Value();

            for (int subdiv = 0; subdiv < num_subdivisions; ++subdiv) {
                const Eigen::Vector3f seed_point = seed_line_start + (subdiv * step) * direction;

                forward_timesteps.push_back(timestep);
                backward_timesteps.push_back(timestep);

                forward_points.push_back(seed_point);
                backward_points.push_back(seed_point);
            }

            // Advect both set of points
            auto previous_forward_points = forward_points;
            auto previous_backward_points = backward_points;

            std::vector<Eigen::Vector3f> advected_forward_points, advected_backward_points;
            advected_forward_points.reserve(num_subdivisions);
            advected_backward_points.reserve(num_subdivisions);

            for (std::size_t integration = 0; integration < num_integration_steps; ++integration) {
                advected_forward_points.clear();
                advected_backward_points.clear();

                for (std::size_t point_index = 0; point_index < previous_forward_points.size(); ++point_index) {
                    advect_point(vector_field, previous_forward_points[point_index], forward_timesteps[point_index], true);
                    advected_forward_points.push_back(previous_forward_points[point_index]);
                }

                for (std::size_t point_index = 0; point_index < previous_backward_points.size(); ++point_index) {
                    advect_point(vector_field, previous_backward_points[point_index], backward_timesteps[point_index], false);
                    advected_backward_points.push_back(previous_backward_points[point_index]);
                }

                // Create surface mesh for stream surface between previous and advected points
                const auto offset = forward_points.size() - previous_forward_points.size();

                for (std::size_t subdivisions = 0; subdivisions < num_subdivisions; ++subdivisions) {
                    this->triangles->push_back(static_cast<unsigned int>(offset + subdivisions));
                    this->triangles->push_back(static_cast<unsigned int>(offset + subdivisions + num_subdivisions + 1));
                    this->triangles->push_back(static_cast<unsigned int>(offset + subdivisions + num_subdivisions + 2));

                    this->triangles->push_back(static_cast<unsigned int>(offset + subdivisions));
                    this->triangles->push_back(static_cast<unsigned int>(offset + subdivisions + num_subdivisions + 2));
                    this->triangles->push_back(static_cast<unsigned int>(offset + subdivisions + 1));
                }

                forward_points.insert(
                    forward_points.end(), advected_forward_points.begin(), advected_forward_points.end());
                backward_points.insert(
                    backward_points.end(), advected_backward_points.begin(), advected_backward_points.end());

                // Check for intersections


                // TODO: this->periodic_orbits.push_back();

                // Prepare for next execution
                std::swap(previous_forward_points, advected_forward_points);
                std::swap(previous_backward_points, advected_backward_points);
            }

            // Store vertices and indices in a GL-friendly manner
            for (const auto& forward_point : forward_points) {
                this->mesh_vertices->push_back(forward_point.x());
                this->mesh_vertices->push_back(forward_point.y());
                this->mesh_vertices->push_back(forward_point.z());
            }

            for (const auto& backward_point : backward_points) {
                this->mesh_vertices->push_back(backward_point.x());
                this->mesh_vertices->push_back(backward_point.y());
                this->mesh_vertices->push_back(backward_point.z());
            }

            const auto offset = static_cast<unsigned int>(this->triangles->size());

            std::transform(this->triangles->begin(), this->triangles->end(), std::back_inserter(*this->triangles),
                [offset](unsigned int index) { return offset + index; });
        }

        ++this->stream_surface_hash;

        this->periodic_orbits_hash = -1;
        this->seed_line_hash = -1;
    }

    this->vector_field_changed = false;
    this->critical_points_changed = false;

    return true;
}

void periodic_orbits_theisel::advect_point(
    const tpf::data::grid<float, float, 2, 2>& grid, Eigen::Vector3f& point, float& delta, const bool forward) const {

    switch (this->integration_method.Param<core::param::EnumParam>()->Value()) {
    case 0:
        advect_point_rk4(grid, point, delta, forward);
        break;
    case 1:
        advect_point_rk45(grid, point, delta, forward);
        break;
    default:
        vislib::sys::Log::DefaultLog.WriteError("Unknown advection method selected");
    }
}

void periodic_orbits_theisel::advect_point_rk4(
    const tpf::data::grid<float, float, 2, 2>& grid, Eigen::Vector3f& point, float& delta, const bool forward) const {

    if (!grid.find_cell(point.head<2>())) return;

    // Calculate step size
    const auto max_velocity = grid.interpolate(point.head<2>()).norm();
    const auto min_cellsize = grid.get_cell_sizes(*grid.find_cell(point.head<2>())).minCoeff();

    const auto steps_per_cell = max_velocity > 0.0f ? min_cellsize / max_velocity : 0.0f;

    // Integration parameters
    const auto sign = forward ? 1.0f : -1.0f;

    // Calculate Runge-Kutta coefficients
    const auto k1 = steps_per_cell * delta * sign * grid.interpolate(point.head<2>());
    const auto k2 = steps_per_cell * delta * sign * grid.interpolate(point.head<2>() + 0.5f * k1);
    const auto k3 = steps_per_cell * delta * sign * grid.interpolate(point.head<2>() + 0.5f * k2);
    const auto k4 = steps_per_cell * delta * sign * grid.interpolate(point.head<2>() + k3);

    // Advect and store position
    Eigen::Vector2f advection = (1.0f / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);

    point += Eigen::Vector3f(advection[0], advection[1], 0.0f);
}

void periodic_orbits_theisel::advect_point_rk45(
    const tpf::data::grid<float, float, 2, 2>& grid, Eigen::Vector3f& point, float& delta, const bool forward) const {

    if (!grid.find_cell(point.head<2>())) return;

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

    // Integration parameters
    const auto sign = forward ? 1.0f : -1.0f;
    const auto max_error = this->max_integration_error.Param<core::param::FloatParam>()->Value();

    // Calculate Runge-Kutta coefficients
    bool decreased = false;

    do {
        const auto k1 = delta * sign * grid.interpolate(point.head<2>());
        const auto k2 = delta * sign * grid.interpolate(point.head<2>() + b_21 * k1);
        const auto k3 = delta * sign * grid.interpolate(point.head<2>() + b_31 * k1 + b_32 * k2);
        const auto k4 = delta * sign * grid.interpolate(point.head<2>() + b_41 * k1 + b_42 * k2 + b_43 * k3);
        const auto k5 = delta * sign * grid.interpolate(point.head<2>() + b_51 * k1 + b_52 * k2 + b_53 * k3 + b_54 * k4);
        const auto k6 = delta * sign * grid.interpolate(point.head<2>() + b_61 * k1 + b_62 * k2 + b_63 * k3 + b_64 * k4 + b_65 * k5);

        // Calculate error estimate
        const auto fifth_order = point.head<2>() + c_1 * k1 + c_2 * k2 + c_3 * k3 + c_4 * k4 + c_5 * k5 + c_6 * k6;
        const auto fourth_order = point.head<2>() + c_1s * k1 + c_2s * k2 + c_3s * k3 + c_4s * k4 + c_5s * k5 + c_6s * k6;

        const auto difference = (fifth_order - fourth_order).cwiseAbs();
        const auto scale = grid.interpolate(point.head<2>()).cwiseAbs();

        const auto error = std::max(0.0f, std::max(difference.x() / scale.x(), difference.y() / scale.y())) / max_error;

        // Set new, adapted time step
        if (error > 1.0f) {
            // Error too large, reduce time step
            delta *= std::max(max_shrink, safety * std::pow(error, shrink_exponent));
            decreased = true;
        } else {
            // Error (too) small, increase time step
            delta *= std::min(max_growth, safety * std::pow(error, grow_exponent));
            decreased = false;
        }

        // Set output
        point << fifth_order, 0.0f;
    } while (decreased);
}

bool periodic_orbits_theisel::get_periodic_orbits_data(core::Call& call) {
    auto& gdc = static_cast<glyph_data_call&>(call);

    if (!(get_input_data() && compute_periodic_orbits())) {
        return false;
    }

    if (gdc.DataHash() != this->periodic_orbits_hash) {
        gdc.clear();

        for (const auto& line : this->periodic_orbits) {
            gdc.add_line(line.second, line.first);
        }

        this->periodic_orbits_hash = gdc.DataHash();
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
        tmc.set_vertices(this->mesh_vertices);
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

bool periodic_orbits_theisel::get_seed_lines_data(core::Call& call) {
    auto& gdc = static_cast<glyph_data_call&>(call);

    if (!(get_input_data() && compute_periodic_orbits())) {
        return false;
    }

    if (gdc.DataHash() != this->seed_line_hash) {
        gdc.clear();

        for (const auto& line : this->seed_lines) {
            gdc.add_line(line.second, line.first);
        }

        this->seed_line_hash = gdc.DataHash();
    }

    return true;
}

bool periodic_orbits_theisel::get_seed_lines_extent(core::Call& call) {
    auto& gdc = static_cast<glyph_data_call&>(call);

    if (!get_input_extent()) {
        return false;
    }

    gdc.set_bounding_rectangle(this->bounding_rectangle);

    return true;
}

bool periodic_orbits_theisel::get_writer_callback(core::Call& call) {
    auto& ddwc = static_cast<core::DirectDataWriterCall&>(call);

    this->get_writer = ddwc.GetCallback();

    return true;
}

} // namespace flowvis
} // namespace megamol
