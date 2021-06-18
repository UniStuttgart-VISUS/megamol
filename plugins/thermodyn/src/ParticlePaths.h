#pragma once

#include <memory>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "mesh/MeshCalls.h"

#include "glm/glm.hpp"

namespace megamol::thermodyn {
class ParticlePaths : public core::Module {
public:
    static const char* ClassName(void) {
        return "ParticlePaths";
    }
    static const char* Description(void) {
        return "ParticlePaths";
    }
    static bool IsAvailable(void) {
        return true;
    }

    ParticlePaths();

    virtual ~ParticlePaths();

protected:
    bool create() override;

    void release() override;

private:
    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(core::moldyn::MultiParticleDataCall& in_parts);

    core::CalleeSlot data_out_slot_;

    core::CallerSlot data_in_slot_;

    std::vector<std::unordered_map<uint64_t /* particle ID */,
        std::pair<std::vector<glm::vec4> /* line */, std::vector<uint32_t> /* indices */>>>
        lines_;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    uint64_t out_data_hash_ = 0;

    std::shared_ptr<mesh::MeshDataAccessCollection> mesh_col_;
};
} // namespace megamol::thermodyn
