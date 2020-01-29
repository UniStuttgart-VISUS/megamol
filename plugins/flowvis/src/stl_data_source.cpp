#include "stdafx.h"
#include "stl_data_source.h"

#include "mmcore/Call.h"

#include "triangle_mesh_call.h"

#include "vislib/sys/Log.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace megamol {
namespace flowvis {

stl_data_source::stl_data_source() : ngmesh_output_slot("mesh_output", "Triangle mesh output") {
    // Remove parent output slot
    Module::SetSlotUnavailable(&(STLDataSource::mesh_output_slot));

    // Create output slot for triangle mesh data
    this->ngmesh_output_slot.SetCallback(
        triangle_mesh_call::ClassName(), "get_extent", &stl_data_source::get_extent_callback);
    this->ngmesh_output_slot.SetCallback(
        triangle_mesh_call::ClassName(), "get_data", &stl_data_source::get_mesh_data_callback);

    Module::MakeSlotAvailable(&this->ngmesh_output_slot);
}

stl_data_source::~stl_data_source() { }

bool stl_data_source::get_extent_callback(core::Call& caller) {
    // Get mesh call
    auto& call = dynamic_cast<triangle_mesh_call&>(caller);

    if (!get_input()) {
        return false;
    }

    call.set_dimension(triangle_mesh_call::dimension_t::THREE);
    call.set_bounding_box(
        vislib::math::Cuboid<float>(this->min_x, this->min_y, this->min_z, this->max_x, this->max_y, this->max_z));

    return true;
}

bool stl_data_source::get_mesh_data_callback(core::Call& caller) {
    // Get mesh call
    auto& call = dynamic_cast<triangle_mesh_call&>(caller);

    if (call.DataHash() != static_cast<SIZE_T>(STLDataSource::hash())) {
        call.SetDataHash(static_cast<SIZE_T>(STLDataSource::hash()));

        // Fill call
        call.set_vertices(this->vertex_buffer);
        call.set_normals(this->normal_buffer);
        call.set_indices(this->index_buffer);
    }

    return true;
}

} // namespace flowvis
} // namespace megamol
