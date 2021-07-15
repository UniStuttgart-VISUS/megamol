#pragma once

#include <memory>
#include <optional>
#include <tuple>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"

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
    bool is_dirty() const {
        return all_frames_slot_.IsDirty() || min_frame_slot_.IsDirty() || max_frame_slot_.IsDirty();
    }

    void reset_dirty() {
        all_frames_slot_.ResetDirty();
        min_frame_slot_.ResetDirty();
        max_frame_slot_.ResetDirty();
    }

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(core::moldyn::MultiParticleDataCall& in_parts);

    std::optional<std::tuple<unsigned int /*min*/, unsigned int /*max*/, unsigned int /*count*/>> calc_frame_extent(
        unsigned int f_count);

    core::CalleeSlot data_out_slot_;

    core::CallerSlot data_in_slot_;

    core::param::ParamSlot all_frames_slot_;

    core::param::ParamSlot min_frame_slot_;

    core::param::ParamSlot max_frame_slot_;

    std::vector<std::unordered_map<uint64_t /* particle ID */,
        std::pair<std::vector<glm::vec3> /* line */, std::vector<uint32_t> /* indices */>>>
        lines_;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    uint64_t out_data_hash_ = 0;

    std::shared_ptr<mesh::MeshDataAccessCollection> mesh_col_;

    unsigned int frame_count_;

    int frame_id_ = 0;
};
} // namespace megamol::thermodyn
