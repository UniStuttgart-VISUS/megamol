#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"

//// CGAL
//#include "CGAL/Exact_predicates_exact_constructions_kernel.h"
//#include "CGAL/Exact_predicates_inexact_constructions_kernel.h"
//
//#include "CGAL/Delaunay_triangulation_3.h"
//#include "CGAL/Fixed_alpha_shape_3.h"
//#include "CGAL/Fixed_alpha_shape_cell_base_3.h"
//#include "CGAL/Fixed_alpha_shape_vertex_base_3.h"
//
//#include "CGAL/Min_sphere_d.h"
//#include "CGAL/Min_sphere_of_points_d_traits_d.h"
//
//#include "CGAL/Triangulation_vertex_base_with_info_3.h"
//// END CGAL

#include "SurfaceHelper.h"

#include "mesh/MeshCalls.h"

namespace megamol::thermodyn {
class ParticleSurface : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "ParticleSurface";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Determines the surface of MD data";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    ParticleSurface();

    virtual ~ParticleSurface();

protected:
    bool create() override;

    void release() override;

private:
    enum class surface_type : std::uint8_t { alpha_shape, gtim };

    bool is_dirty() {
        return _alpha_slot.IsDirty() || _type_slot.IsDirty() || _vert_type_slot.IsDirty() ||
               toggle_interface_slot_.IsDirty();
    }

    void reset_dirty() {
        _alpha_slot.ResetDirty();
        _type_slot.ResetDirty();
        _vert_type_slot.ResetDirty();
        toggle_interface_slot_.ResetDirty();
    }

    bool assert_data(core::moldyn::MultiParticleDataCall& call);

    struct particle_data {
        uint64_t idx;
        uint32_t id;
        float x, y, z;
        float i;
        float dx, dy, dz;
    };

    std::tuple<std::vector<float> /*vertices*/, std::vector<float> /*normals*/, std::vector<unsigned int> /*indices*/,
        std::vector<particle_data> /*point data*/>
    compute_alpha_shape(float alpha, core::moldyn::SimpleSphericalParticles const& parts);

    std::tuple<std::vector<float> /*vertices*/, std::vector<float> /*normals*/, std::vector<unsigned int> /*indices*/,
        std::vector<particle_data> /*point data*/>
    compute_alpha_shape(float alpha, std::list<std::pair<Point_3, std::size_t>> const& points,
        core::moldyn::SimpleSphericalParticles const& parts);

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool get_part_data_cb(core::Call& c);

    bool get_part_extent_cb(core::Call& c);

    bool write_info_cb(core::param::ParamSlot& p);

    core::CalleeSlot _out_mesh_slot;

    core::CalleeSlot _out_part_slot;

    core::CallerSlot _in_data_slot;

    core::CallerSlot _tf_slot;

    core::CallerSlot _flags_read_slot;

    core::param::ParamSlot _alpha_slot;

    core::param::ParamSlot _type_slot;

    core::param::ParamSlot _vert_type_slot;

    core::param::ParamSlot toggle_interface_slot_;

    core::param::ParamSlot info_filename_slot_;

    core::param::ParamSlot write_info_slot_;

    std::vector<std::vector<float>> _vertices;

    std::vector<std::vector<float>> _normals;

    std::vector<std::vector<float>> _colors;

    std::vector<std::vector<float>> _unsel_colors;

    std::vector<std::vector<std::uint32_t>> _indices;

    std::shared_ptr<mesh::MeshDataAccessCollection> _mesh_access_collection;

    unsigned int _frame_id = std::numeric_limits<unsigned int>::max();

    std::size_t _in_data_hash = std::numeric_limits<std::size_t>::max();

    std::size_t _out_data_hash = 0;

    std::vector<std::vector<particle_data>> _part_data;

    int _fcr_version = -1;

    std::vector<std::tuple<size_t, size_t>> _sel;

    std::size_t increment = 0;

    uint64_t total_tri_count_;
};
} // namespace megamol::thermodyn
