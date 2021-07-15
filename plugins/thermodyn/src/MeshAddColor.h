#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/view/CallGetTransferFunction.h"

#include "mesh/MeshCalls.h"

#include "glm/glm.hpp"

namespace megamol::thermodyn {
class MeshAddColor : public core::Module {
public:
    using color_con_t = std::vector<glm::vec4>;

    static const char* ClassName(void) {
        return "MeshAddColor";
    }

    static const char* Description(void) {
        return "MeshAddColor";
    }

    static bool IsAvailable(void) {
        return true;
    }

    MeshAddColor();

    virtual ~MeshAddColor();

protected:
    bool create() override;

    void release() override;

private:
    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(
        mesh::CallMesh& mesh, core::moldyn::MultiParticleDataCall& particles, core::view::CallGetTransferFunction* tf);

    core::CalleeSlot data_out_slot_;

    core::CallerSlot mesh_in_slot_;

    core::CallerSlot parts_in_slot_;

    core::CallerSlot tf_in_slot_;

    std::vector<color_con_t> colors_;

    std::shared_ptr<mesh::MeshDataAccessCollection> mesh_col_;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    unsigned int frame_id_ = 0;

    uint64_t version = 0;
};
} // namespace megamol::thermodyn
