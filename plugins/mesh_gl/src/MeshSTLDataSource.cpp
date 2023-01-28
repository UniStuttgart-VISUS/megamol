#include "MeshSTLDataSource.h"

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

namespace megamol::mesh_gl {

MeshSTLDataSource::MeshSTLDataSource() : ngmesh_output_slot("mesh_output", "Triangle mesh output") {
    // Remove parent output slot
    Module::SetSlotUnavailable(&(datatools_gl::io::STLDataSource::mesh_output_slot));

    // Create output slot for triangle mesh data
    this->ngmesh_output_slot.SetCallback(
        mesh::TriangleMeshCall::ClassName(), "get_extent", &MeshSTLDataSource::get_extent_callback);
    this->ngmesh_output_slot.SetCallback(
        mesh::TriangleMeshCall::ClassName(), "get_data", &MeshSTLDataSource::get_mesh_data_callback);

    Module::MakeSlotAvailable(&this->ngmesh_output_slot);
}

MeshSTLDataSource::~MeshSTLDataSource() {
    this->Release();
}

bool MeshSTLDataSource::get_extent_callback(core::Call& caller) {
    // Get mesh call
    auto& call = dynamic_cast<mesh::TriangleMeshCall&>(caller);

    if (!datatools_gl::io::STLDataSource::get_input()) {
        return false;
    }

    call.set_dimension(mesh::TriangleMeshCall::dimension_t::THREE);
    call.set_bounding_box(
        vislib::math::Cuboid<float>(datatools_gl::io::STLDataSource::min_x, datatools_gl::io::STLDataSource::min_y,
            datatools_gl::io::STLDataSource::min_z, datatools_gl::io::STLDataSource::max_x,
            datatools_gl::io::STLDataSource::max_y, datatools_gl::io::STLDataSource::max_z));

    return true;
}

bool MeshSTLDataSource::get_mesh_data_callback(core::Call& caller) {
    // Get mesh call
    auto& call = dynamic_cast<mesh::TriangleMeshCall&>(caller);

    if (!get_extent_callback(caller)) {
        return false;
    }

    if (call.DataHash() != static_cast<SIZE_T>(datatools_gl::io::STLDataSource::hash())) {
        call.SetDataHash(static_cast<SIZE_T>(datatools_gl::io::STLDataSource::hash()));

        // Fill call
        call.set_vertices(datatools_gl::io::STLDataSource::vertex_buffer);
        call.set_normals(datatools_gl::io::STLDataSource::normal_buffer);
        call.set_indices(datatools_gl::io::STLDataSource::index_buffer);
    }

    return true;
}

} // namespace megamol::mesh_gl
