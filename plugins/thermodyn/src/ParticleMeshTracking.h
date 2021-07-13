#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"

#include "mmstd_datatools/table/TableDataCall.h"

namespace megamol::thermodyn {
class ParticleMeshTracking : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ParticleMeshTracking";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "ParticleMeshTracking";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    ParticleMeshTracking();

    virtual ~ParticleMeshTracking();

protected:
    bool create() override;

    void release() override;

private:
    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(core::moldyn::MultiParticleDataCall& particles);

    core::CalleeSlot data_out_slot_;

    core::CallerSlot parts_in_slot_;

    core::param::ParamSlot all_frames_slot_;

    core::param::ParamSlot min_frames_slot_;

    core::param::ParamSlot max_frames_slot_;

    std::vector<float> data_;

    std::vector<stdplugin::datatools::table::TableDataCall::ColumnInfo> infos_;

    uint64_t row_count_;

    uint64_t col_count_;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    uint64_t out_data_hash_ = 0;
};
} // namespace megamol::thermodyn
