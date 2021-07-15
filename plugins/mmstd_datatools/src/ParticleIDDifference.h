#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol::stdplugin::datatools {
class ParticleIDDifference : public core::Module {
public:
    static const char* ClassName(void) {
        return "ParticleIDDifference";
    }
    static const char* Description(void) {
        return "ParticleIDDifference";
    }
    static bool IsAvailable(void) {
        return true;
    }

    ParticleIDDifference();

    virtual ~ParticleIDDifference();

protected:
    bool create() override;

    void release() override;

private:
    struct particle_t {
        uint64_t id;
        float x, y, z;
        float i_col;
        float dx, dy, dz;
    };

    bool is_dirty() const {
        return threshold_slot_.IsDirty();
    }

    void reset_dirty() {
        threshold_slot_.ResetDirty();
    }

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(
        core::moldyn::MultiParticleDataCall& a_particles, core::moldyn::MultiParticleDataCall& b_particles);

    core::CalleeSlot data_out_slot_;

    core::CallerSlot a_particles_slot_;

    core::CallerSlot b_particles_slot_;

    core::param::ParamSlot threshold_slot_;

    std::vector<std::vector<particle_t>> data_;

    int frame_id_ = -1;

    uint64_t a_in_data_hash_ = std::numeric_limits<uint64_t>::max();

    uint64_t b_in_data_hash_ = std::numeric_limits<uint64_t>::max();

    uint64_t out_data_hash_ = 0;
};
} // namespace megamol::stdplugin::datatools
