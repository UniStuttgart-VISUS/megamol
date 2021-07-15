#pragma once

#include <memory>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"

#include "mmstd_datatools/StatisticsCall.h"

namespace megamol::stdplugin::datatools {
class IColStatistics : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "IColStatistics";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "IColStatistics";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    IColStatistics();

    virtual ~IColStatistics();

protected:
    bool create() override;

    void release() override;

private:
    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(core::moldyn::MultiParticleDataCall& in_parts);

    core::CalleeSlot data_out_slot_;

    core::CallerSlot data_in_slot_;

    core::param::ParamSlot num_buckets_slot_;

    std::vector<StatisticsData> data_;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    int frame_id_ = -1;

    uint64_t out_data_hash_ = 0;
};
} // namespace megamol::stdplugin::datatools
