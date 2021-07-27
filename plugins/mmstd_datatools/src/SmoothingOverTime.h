#pragma once

#include <array>
#include <vector>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol::stdplugin::datatools {
class SmoothingOverTime : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName(void) {
        return "SmoothingOverTime";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description(void) {
        return "Smooth ICol values over time";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable(void) {
        return true;
    }

    SmoothingOverTime();

    virtual ~SmoothingOverTime();

protected:
    bool create() override;

    void release() override;

private:
    bool is_dirty() const {
        return frame_count_slot_.IsDirty();
    }

    void reset_dirty() {
        frame_count_slot_.ResetDirty();
    }

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    core::CalleeSlot out_data_slot_;

    core::CallerSlot in_data_slot_;

    core::param::ParamSlot frame_count_slot_;

    core::param::ParamSlot frame_skip_slot_;

    std::size_t in_data_hash_ = std::numeric_limits<std::size_t>::max();

    std::size_t out_data_hash_ = 0;

    int frame_id_ = -1;

    std::vector<std::vector<float>> smoothed_icol_;

    std::vector<std::vector<std::uint64_t>> identity_;

    std::vector<std::vector<float>> weigths_;

    std::vector<std::array<float, 2>> minmax_;
};
} // namespace megamol::stdplugin::datatools
