#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"

#include "thermodyn/CallStatsInfo.h"

#include "SurfaceHelper.h"

namespace megamol::thermodyn {
class ParticleSpawner : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "ParticleSpawner";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Spawns random particles";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    ParticleSpawner();

    virtual ~ParticleSpawner();

protected:
    bool create() override;

    void release() override;

private:
    bool is_dirty() {
        return dx_slot_.IsDirty() || dy_slot_.IsDirty() || dz_slot_.IsDirty() || dmag_slot_.IsDirty() ||
               alpha_slot_.IsDirty();
    }

    void reset_dirty() {
        dx_slot_.ResetDirty();
        dy_slot_.ResetDirty();
        dz_slot_.ResetDirty();
        dmag_slot_.ResetDirty();
        alpha_slot_.ResetDirty();
    }

    bool assert_data(core::moldyn::MultiParticleDataCall& part_call, CallStatsInfo& stats_call);

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    core::CalleeSlot out_data_slot_;

    core::CallerSlot in_data_slot_;

    core::CallerSlot in_stats_slot_;

    core::param::ParamSlot dx_slot_;

    core::param::ParamSlot dy_slot_;

    core::param::ParamSlot dz_slot_;

    core::param::ParamSlot dmag_slot_;

    core::param::ParamSlot alpha_slot_;

    core::param::ParamSlot num_part_slot_;

    std::size_t in_data_hash_ = std::numeric_limits<std::size_t>::max();

    int frame_id_ = -1;

    std::vector<std::vector<float>> data_;

    std::size_t out_data_hash_ = 0;
};
} // namespace megamol::thermodyn
