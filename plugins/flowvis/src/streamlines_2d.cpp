#include "stdafx.h"
#include "streamlines_2d.h"

#include "glyph_data_call.h"
#include "vector_field_call.h"

#include "flowvis/integrator.h"

#include "mmcore/Call.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/DataHash.h"

#include "vislib/sys/Log.h"

#include "data/tpf_data_information.h"
#include "data/tpf_grid.h"
#include "data/tpf_grid_information.h"

namespace megamol {
namespace flowvis {

streamlines_2d::streamlines_2d()
    : streamlines_slot("streamlines", "Computed streamlines")
    , vector_field_slot("vector_field", "Vector field")
    , seed_points_slot("seed_points", "Seed points")
    , integration_method("integration_method", "Method for streamline integration")
    , num_integration_steps("num_integration_steps", "Number of streamline integration steps")
    , integration_timestep("integration_timestep", "Initial time step for streamline integration")
    , max_integration_error("max_integration_error", "Maximum integration error for Runge-Kutta 4-5")
    , direction("direction", "Integration direction for streamline computation")
    , vector_field_hash(-1)
    , vector_field_changed(false)
    , seed_points_hash(-1)
    , seed_points_changed(false)
    , streamlines_hash(-1) {

    // Connect output
    this->streamlines_slot.SetCallback(
        glyph_data_call::ClassName(), glyph_data_call::FunctionName(0), &streamlines_2d::get_streamlines_data);
    this->streamlines_slot.SetCallback(
        glyph_data_call::ClassName(), glyph_data_call::FunctionName(1), &streamlines_2d::get_streamlines_extent);
    this->MakeSlotAvailable(&this->streamlines_slot);

    // Connect input
    this->vector_field_slot.SetCompatibleCall<vector_field_call::vector_field_description>();
    this->MakeSlotAvailable(&this->vector_field_slot);

    this->seed_points_slot.SetCompatibleCall<glyph_data_call::glyph_data_description>();
    this->MakeSlotAvailable(&this->seed_points_slot);

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

    this->direction << new core::param::EnumParam(0);
    this->direction.Param<core::param::EnumParam>()->SetTypePair(0, "both");
    this->direction.Param<core::param::EnumParam>()->SetTypePair(1, "forward");
    this->direction.Param<core::param::EnumParam>()->SetTypePair(2, "backward");
    this->MakeSlotAvailable(&this->direction);
}

streamlines_2d::~streamlines_2d() { this->Release(); }

bool streamlines_2d::create() { return true; }

void streamlines_2d::release() {}

bool streamlines_2d::get_input_data() {
    auto vfc_ptr = this->vector_field_slot.CallAs<vector_field_call>();
    auto spc_ptr = this->seed_points_slot.CallAs<glyph_data_call>();

    if (vfc_ptr == nullptr || spc_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("Both input slots must be connected");
        return false;
    }

    auto& vfc = *vfc_ptr;
    auto& spc = *spc_ptr;

    if (!vfc(0)) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting input vector field");
        return false;
    }

    if (!spc(0)) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting input seed points");
        return false;
    }

    if (vfc.DataHash() != this->vector_field_hash) {
        this->resolution = vfc.get_resolution();
        this->grid_positions = vfc.get_positions();
        this->vectors = vfc.get_vectors();

        this->vector_field_hash = vfc.DataHash();
        this->vector_field_changed = true;
    }

    if (spc.DataHash() != this->seed_points_hash) {
        this->seed_points = spc.get_points();

        this->seed_points_hash = spc.DataHash();
        this->seed_points_changed = true;
    }

    return true;
}

bool streamlines_2d::get_input_extent() {
    auto vfc_ptr = this->vector_field_slot.CallAs<vector_field_call>();
    auto spc_ptr = this->seed_points_slot.CallAs<glyph_data_call>();

    if (vfc_ptr == nullptr || spc_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("Both input slots must be connected");
        return false;
    }

    auto& vfc = *vfc_ptr;
    auto& spc = *spc_ptr;

    if (!vfc(1)) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting input vector field extents");
        return false;
    }

    if (!spc(1)) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting input seed points extents");
        return false;
    }

    this->bounding_rectangle = vfc.get_bounding_rectangle();

    return true;
}

bool streamlines_2d::compute_streamlines() {
    if (this->vector_field_changed || this->seed_points_changed || this->direction.IsDirty() ||
        this->integration_method.IsDirty() || this->integration_timestep.IsDirty() ||
        this->max_integration_error.IsDirty() || this->num_integration_steps.IsDirty()) {

        this->direction.ResetDirty();
        this->integration_method.ResetDirty();
        this->integration_timestep.ResetDirty();
        this->max_integration_error.ResetDirty();
        this->num_integration_steps.ResetDirty();

        // Create grid containing the vector field
        tpf::data::extent_t extent;
        extent.push_back(std::make_pair(0ull, static_cast<std::size_t>(this->resolution[0] - 1)));
        extent.push_back(std::make_pair(0ull, static_cast<std::size_t>(this->resolution[1] - 1)));

        const Eigen::Vector2f origin((*this->grid_positions)[0], (*this->grid_positions)[1]);
        const Eigen::Vector2f right((*this->grid_positions)[2], (*this->grid_positions)[3]);
        const Eigen::Vector2f up(
            (*this->grid_positions)[2 * this->resolution[0]], (*this->grid_positions)[2 * this->resolution[0] + 1]);

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

        const tpf::data::grid<float, float, 2, 2> vector_field("vector_field", extent, *this->vectors,
            std::move(cell_coordinates), std::move(node_coordinates), std::move(cell_sizes));

        // Get parameters
        const auto integration_method = this->integration_method.Param<core::param::EnumParam>()->Value();
        const auto num_integration_steps = this->num_integration_steps.Param<core::param::IntParam>()->Value();
        const auto max_integration_error = this->max_integration_error.Param<core::param::FloatParam>()->Value();
        const auto direction = this->direction.Param<core::param::EnumParam>()->Value();

        // Advect all seed points
        this->streamlines.clear();
        this->streamlines.resize(direction == 0 ? 2 * this->seed_points.size() : this->seed_points.size());

        for (std::size_t direction_run = 0; direction_run < (direction == 0 ? 2 : 1); ++direction_run) {
            #pragma omp parallel for
            for (long long point_index = 0; point_index < static_cast<long long>(this->seed_points.size());
                 ++point_index) {

                Eigen::Vector2f point = this->seed_points[point_index].first;

                auto integration_timestep = this->integration_timestep.Param<core::param::FloatParam>()->Value();

                std::vector<Eigen::Vector2f> line_points;
                line_points.reserve(num_integration_steps + 1);

                line_points.push_back(point);

                try {
                    for (std::size_t integration = 0; integration < num_integration_steps; ++integration) {
                        switch (integration_method) {
                        case 0:
                            advect_point_rk4<2>(vector_field, point, integration_timestep,
                                direction == 1 || (direction == 0 && direction_run == 0));
                            line_points.push_back(point);
                            break;
                        case 1:
                            advect_point_rk45<2>(vector_field, point, integration_timestep, max_integration_error,
                                direction == 1 || (direction == 0 && direction_run == 0));
                            line_points.push_back(point);
                            break;
                        }
                    }
                } catch (std::exception&) {
                    if (line_points.size() == 1) {
                        line_points.push_back(point);
                    }
                }

                this->streamlines[direction_run * this->seed_points.size() + point_index] =
                   std::make_pair(static_cast<float>(point_index), line_points);
            }
        }

        this->streamlines_hash = core::utility::DataHash(this->vector_field_hash, this->seed_points_hash,
            this->direction.Param<core::param::EnumParam>()->Value(),
            this->integration_method.Param<core::param::EnumParam>()->Value(),
            this->integration_timestep.Param<core::param::FloatParam>()->Value(),
            this->max_integration_error.Param<core::param::FloatParam>()->Value(),
            this->num_integration_steps.Param<core::param::IntParam>()->Value());
    }

    this->vector_field_changed = false;
    this->seed_points_changed = false;

    return true;
}

bool streamlines_2d::get_streamlines_data(core::Call& call) {
    auto& gdc = static_cast<glyph_data_call&>(call);

    if (!(get_input_data() && compute_streamlines())) {
        return false;
    }

    if (gdc.DataHash() != this->streamlines_hash) {
        gdc.clear();

        for (const auto& line : this->streamlines) {
            gdc.add_line(line.second, line.first);
        }

        gdc.SetDataHash(this->streamlines_hash);
    }

    return true;
}

bool streamlines_2d::get_streamlines_extent(core::Call& call) {
    auto& gdc = static_cast<glyph_data_call&>(call);

    if (!get_input_extent()) {
        return false;
    }

    gdc.set_bounding_rectangle(this->bounding_rectangle);

    return true;
}

} // namespace flowvis
} // namespace megamol
