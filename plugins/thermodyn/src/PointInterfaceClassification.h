#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "mesh/MeshCalls.h"

#include "mmstd_datatools/PointcloudHelpers.h"

#include "nanoflann.hpp"

namespace megamol::thermodyn {

class PointInterfaceClassification : public core::Module {
public:
    static const char* ClassName(void) {
        return "PointInterfaceClassification";
    }

    static const char* Description(void) {
        return "PointInterfaceClassification";
    }

    static bool IsAvailable(void) {
        return true;
    }

    PointInterfaceClassification();

    virtual ~PointInterfaceClassification();

protected:
    bool create() override;

    void release() override;

private:
    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(core::moldyn::MultiParticleDataCall& points, mesh::CallMesh& mesh,
        core::moldyn::MultiParticleDataCall& mesh_points);

    core::CalleeSlot point_out_slot_;

    core::CallerSlot point_in_slot_;

    core::CallerSlot mesh_in_slot_;

    core::CallerSlot mesh_points_in_slot_;

    core::param::ParamSlot critical_temp_slot_;

    std::shared_ptr<stdplugin::datatools::genericPointcloud<float, 3>> point_cloud_;

    template<typename T, int DIM>
    using kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<T, stdplugin::datatools::genericPointcloud<T, DIM>>,
        stdplugin::datatools::genericPointcloud<T, DIM>, DIM>;

    std::shared_ptr<kd_tree_t<float, 3>> kd_tree_;

    std::vector<std::vector<glm::vec3>> positions_;

    std::vector<std::vector<float>> distances_;

    std::vector<std::pair<float, float>> dist_minmax_;

    std::vector<std::vector<float>> in_interface_;

    size_t out_data_hash_ = 0;

    int frame_id_ = -1;
};

} // namespace megamol::thermodyn
