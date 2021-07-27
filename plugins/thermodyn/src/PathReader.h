#pragma once

#include <memory>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/math/Cuboid.h"

#include "mesh/MeshCalls.h"

#include "glm/glm.hpp"

namespace megamol::thermodyn {
class PathReader : public core::Module {
public:
    static const char* ClassName(void) {
        return "PathReader";
    }
    static const char* Description(void) {
        return "PathReader";
    }
    static bool IsAvailable(void) {
        return true;
    }

    PathReader();

    virtual ~PathReader();

protected:
    bool create() override;

    void release() override;

private:
    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(int frame_id);

    core::CalleeSlot data_out_slot_;

    core::param::ParamSlot filename_slot_;

    std::shared_ptr<mesh::MeshDataAccessCollection> mesh_col_;

    uint32_t frame_count_;

    vislib::math::Cuboid<float> bbox_;

    vislib::math::Cuboid<float> cbox_;

    uint64_t out_data_hash_ = 0;

    int frame_id_ = -1;

    std::vector<std::vector<uint32_t>> indices_;

    std::vector<std::vector<glm::vec3>> positions_;

    std::vector<std::vector<glm::vec3>> colors_;
};
} // namespace megamol::thermodyn
