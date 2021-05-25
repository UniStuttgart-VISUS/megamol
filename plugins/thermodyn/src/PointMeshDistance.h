#pragma once

#include <list>
#include <vector>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "mesh/MeshCalls.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "CGAL/AABB_traits.h"
#include "CGAL/AABB_tree.h"
#include "CGAL/AABB_triangle_primitive.h"
#include "CGAL/Point_3.h"
#include "CGAL/Simple_cartesian.h"

namespace megamol::thermodyn {
class PointMeshDistance : public core::Module {
public:
    typedef CGAL::Simple_cartesian<double> K;
    typedef K::FT FT;
    /*typedef K::Ray_3 Ray;
    typedef K::Line_3 Line;*/
    typedef K::Point_3 Point;
    typedef K::Triangle_3 Triangle;
    typedef std::list<Triangle>::iterator Iterator;
    typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
    typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
    typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

    static const char* ClassName(void) {
        return "PointMeshDistance";
    }

    static const char* Description(void) {
        return "PointMeshDistance";
    }

    static bool IsAvailable(void) {
        return true;
    }

    PointMeshDistance();

    virtual ~PointMeshDistance();

protected:
    bool create() override;

    void release() override;

private:
    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(core::moldyn::MultiParticleDataCall& points, mesh::CallMesh& mesh);

    bool is_dirty() const {
        return toggle_threshold_slot_.IsDirty() || threshold_slot_.IsDirty() || toggle_invert_slot_.IsDirty();
    }

    void reset_dirty() {
        toggle_threshold_slot_.ResetDirty();
        threshold_slot_.ResetDirty();
        toggle_invert_slot_.ResetDirty();
    }

    core::CallerSlot in_points_slot_;

    core::CallerSlot in_mesh_slot_;

    core::CalleeSlot out_distances_slot_;

    core::param::ParamSlot toggle_threshold_slot_;

    core::param::ParamSlot threshold_slot_;

    core::param::ParamSlot toggle_invert_slot_;

    std::vector<std::vector<glm::vec3>> positions_;

    std::vector<std::vector<float>> distances_;

    std::vector<std::pair<float, float>> dist_minmax_;

    size_t out_data_hash_ = 0;

    int frame_id_ = -1;
};
} // namespace megamol::thermodyn
