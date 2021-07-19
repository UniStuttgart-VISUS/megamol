#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/UniFlagCalls.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol::stdplugin::datatools::masking {
class IColMasking : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "IColMasking";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "IColMasking";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    IColMasking();

    virtual ~IColMasking();

protected:
    bool create() override;

    void release() override;

private:
    bool is_dirty() const {
        return min_val_slot_.IsDirty() || max_val_slot_.IsDirty();
    }

    void reset_dirty() {
        min_val_slot_.ResetDirty();
        max_val_slot_.ResetDirty();
    }

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(core::moldyn::MultiParticleDataCall& particles, core::FlagCallRead_CPU& flags_read,
        core::FlagCallWrite_CPU& flags_write);

    core::CalleeSlot data_out_slot_;

    core::CallerSlot data_in_slot_;

    core::CallerSlot flags_read_slot_;

    core::CallerSlot flags_write_slot_;

    core::param::ParamSlot min_val_slot_;

    core::param::ParamSlot max_val_slot_;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    unsigned int frame_id_ = 0;
};
} // namespace megamol::stdplugin::datatools::masking
