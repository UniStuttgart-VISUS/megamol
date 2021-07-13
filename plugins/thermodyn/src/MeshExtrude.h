#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"

#include "mesh/MeshCalls.h"

#include "glm/glm.hpp"

namespace megamol::thermodyn {
class MeshExtrude : public core::Module {
public:
    using vertex_con_t = std::vector<glm::vec3>;

    static const char* ClassName(void) {
        return "MeshExtrude";
    }

    static const char* Description(void) {
        return "MeshExtrude";
    }

    static bool IsAvailable(void) {
        return true;
    }

    MeshExtrude();

    virtual ~MeshExtrude();

protected:
    bool create() override;

    void release() override;

private:
    bool is_dirty() const {
        return crit_temp_slot_.IsDirty();
    }

    void reset_dirty() {
        crit_temp_slot_.ResetDirty();
    }

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(mesh::CallMesh& mesh, core::moldyn::MultiParticleDataCall& particles);

    core::CalleeSlot data_out_slot_;

    core::CallerSlot mesh_in_slot_;

    core::CallerSlot parts_in_slot_;

    core::param::ParamSlot crit_temp_slot_;

    std::vector<vertex_con_t> vertices_;

    std::shared_ptr<mesh::MeshDataAccessCollection> mesh_col_;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    unsigned int frame_id_ = 0;

    uint64_t version = 0;
};
} // namespace megamol::thermodyn
