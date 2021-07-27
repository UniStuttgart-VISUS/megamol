#pragma once

#include <vector>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "mesh/MeshCalls.h"

#include "mmstd_datatools/PointcloudHelpers.h"

#include "nanoflann.hpp"

namespace megamol::thermodyn {
class PointSurfaceElementsDistance : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "PointSurfaceElementsDistance";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "PointSurfaceElementsDistance";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    PointSurfaceElementsDistance();

    virtual ~PointSurfaceElementsDistance();

protected:
    bool create() override;

    void release() override;

private:
    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(
        mesh::CallMesh& in_part_mesh, mesh::CallMesh& in_inter_mesh, core::moldyn::MultiParticleDataCall& in_parts);

    core::CalleeSlot data_out_slot_;

    core::CallerSlot part_mesh_in_slot_;

    core::CallerSlot inter_mesh_in_slot_;

    core::CallerSlot parts_in_slot_;

    std::vector<std::shared_ptr<stdplugin::datatools::glmPointcloud>> inner_point_clouds_;

    std::vector<std::shared_ptr<stdplugin::datatools::glmPointcloud>> outer_point_clouds_;

    using kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, stdplugin::datatools::glmPointcloud>,
        stdplugin::datatools::glmPointcloud, 3>;

    std::vector<std::shared_ptr<kd_tree_t>> inner_kd_trees_;

    std::vector<std::shared_ptr<kd_tree_t>> outer_kd_trees_;

    std::vector<std::vector<float>> inter_pos_classes_;

    std::vector<std::pair<float, float>> inter_pos_classes_minmax_;

    /*std::size_t part_mesh_data_hash_ = std::numeric_limits<std::size_t>::max();

    std::size_t inter_mesh_data_hash_ = std::numeric_limits<std::size_t>::max();*/

    std::size_t parts_data_hash_ = std::numeric_limits<std::size_t>::max();

    std::size_t out_data_hash_ = 0;

    int frame_id_ = -1;
};
} // namespace megamol::thermodyn
