#include "stdafx.h"
#include "STLDataSource.h"

#include "mmcore/Call.h"

#include "mesh/TriangleMeshCall.h"

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
namespace mesh {

    STLDataSource::STLDataSource() : ngmesh_output_slot("mesh_output", "Triangle mesh output") {
        // Remove parent output slot
        Module::SetSlotUnavailable(&(stdplugin::datatools::io::STLDataSource::mesh_output_slot));

        // Create output slot for triangle mesh data
        this->ngmesh_output_slot.SetCallback(
            TriangleMeshCall::ClassName(), "get_extent", &STLDataSource::get_extent_callback);
        this->ngmesh_output_slot.SetCallback(
            TriangleMeshCall::ClassName(), "get_data", &STLDataSource::get_mesh_data_callback);

        Module::MakeSlotAvailable(&this->ngmesh_output_slot);
    }

    STLDataSource::~STLDataSource() {
        this->Release();
    }

    bool STLDataSource::get_extent_callback(core::Call& caller) {
        // Get mesh call
        auto& call = dynamic_cast<TriangleMeshCall&>(caller);

        if (!get_input()) {
            return false;
        }

        call.set_dimension(TriangleMeshCall::dimension_t::THREE);
        call.set_bounding_box(vislib::math::Cuboid<float>(stdplugin::datatools::io::STLDataSource::min_x,
            stdplugin::datatools::io::STLDataSource::min_y, stdplugin::datatools::io::STLDataSource::min_z,
            stdplugin::datatools::io::STLDataSource::max_x, stdplugin::datatools::io::STLDataSource::max_y,
            stdplugin::datatools::io::STLDataSource::max_z));

        return true;
    }

    bool STLDataSource::get_mesh_data_callback(core::Call& caller) {
        // Get mesh call
        auto& call = dynamic_cast<TriangleMeshCall&>(caller);

        if (!get_extent_callback(caller)) {
            return false;
        }

        if (call.DataHash() != static_cast<SIZE_T>(stdplugin::datatools::io::STLDataSource::hash())) {
            call.SetDataHash(static_cast<SIZE_T>(stdplugin::datatools::io::STLDataSource::hash()));

            // Fill call
            call.set_vertices(stdplugin::datatools::io::STLDataSource::vertex_buffer);
            call.set_normals(stdplugin::datatools::io::STLDataSource::normal_buffer);
            call.set_indices(stdplugin::datatools::io::STLDataSource::index_buffer);
        }

        return true;
    }

} // namespace flowvis
} // namespace megamol
