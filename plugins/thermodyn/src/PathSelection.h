#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/UniFlagCalls.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "mesh/MeshCalls.h"

namespace megamol::thermodyn {
class PathSelection : public core::Module {
public:
    static const char* ClassName(void) {
        return "PathSelection";
    }

    static const char* Description(void) {
        return "PathSelection";
    }

    static bool IsAvailable(void) {
        return true;
    }

    PathSelection();

    virtual ~PathSelection();

protected:
    bool create() override;

    void release() override;

private:
    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(
        mesh::CallMesh& mesh, core::moldyn::MultiParticleDataCall& particles, core::FlagCallRead_CPU& flags);

    core::CalleeSlot data_out_slot_;

    core::CallerSlot data_in_slot_;

    core::CallerSlot parts_in_slot_;

    core::CallerSlot flags_read_slot_;

    std::shared_ptr<mesh::MeshDataAccessCollection> mesh_col_;

    uint64_t out_data_hash_ = 0;
};
} // namespace megamol::thermodyn
