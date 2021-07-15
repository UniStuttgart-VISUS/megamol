#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/UniFlagCalls.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "mmstd_datatools/masking/CallMaskOffsets.h"

namespace megamol::stdplugin::datatools::masking {
class CreateBaseMask : public core::Module {
public:
    static const char* ClassName(void) {
        return "CreateBaseMask";
    }

    static const char* Description(void) {
        return "CreateBaseMask";
    }

    static bool IsAvailable(void) {
        return true;
    }

    CreateBaseMask();

    virtual ~CreateBaseMask();

protected:
    bool create() override;

    void release() override;

private:
    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool get_offsets_cb(core::Call& c);

    bool dummy_cb(core::Call& c) {
        return true;
    }

    bool assert_data(core::moldyn::MultiParticleDataCall& particles, core::FlagCallRead_CPU& flags_read,
        core::FlagCallWrite_CPU& flags_write);

    core::CalleeSlot data_out_slot_;

    core::CalleeSlot offsets_out_slot_;

    core::CallerSlot data_in_slot_;

    core::CallerSlot flags_read_slot_;

    core::CallerSlot flags_write_slot_;

    std::shared_ptr<offsets_t> inclusive_sum_;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    unsigned int frame_id_ = 0;

    uint64_t offsets_version_ = 0;
};
} // namespace megamol::stdplugin::datatools::masking
