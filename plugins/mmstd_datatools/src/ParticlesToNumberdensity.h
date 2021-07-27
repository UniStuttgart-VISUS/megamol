#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "mmcore/misc/VolumetricDataCall.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"

namespace megamol::stdplugin::datatools {
class ParticlesToNumberdensity : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "ParticlesToNumberdensity";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Computes a number density volume from particles";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    ParticlesToNumberdensity();

    virtual ~ParticlesToNumberdensity();

protected:
    bool create() override;

    void release() override;

private:
    bool is_dirty() const {
        return grid_x_res_slot_.IsDirty() || grid_y_res_slot_.IsDirty() || grid_z_res_slot_.IsDirty() ||
               surface_slot_.IsDirty();
    }

    void reset_dirty() {
        grid_x_res_slot_.ResetDirty();
        grid_y_res_slot_.ResetDirty();
        grid_z_res_slot_.ResetDirty();
        surface_slot_.ResetDirty();
    }

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool dummy_cb(core::Call& c);

    bool assert_data(core::moldyn::MultiParticleDataCall& parts);

    void modify_bbox(megamol::core::moldyn::MultiParticleDataCall& parts);

    core::CalleeSlot data_out_slot_;

    core::CallerSlot data_in_slot_;

    core::param::ParamSlot grid_x_res_slot_;
    core::param::ParamSlot grid_y_res_slot_;
    core::param::ParamSlot grid_z_res_slot_;

    core::param::ParamSlot surface_slot_;

    std::vector<float> vol_data_;

    int frame_id_ = -1;

    uint64_t out_data_hash_ = 0;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    core::misc::VolumetricDataCall::Metadata metadata;

    float min_dens_;

    float max_dens_;
};
} // namespace megamol::stdplugin::datatools
