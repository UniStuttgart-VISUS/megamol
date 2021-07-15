#pragma once

#include <unordered_map>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"

#include "mmstd_datatools/table/TableDataCall.h"

#include "glm/glm.hpp"

namespace megamol::thermodyn {

class AccumulateInterfacePresence2 : public core::Module {
public:
    static const char* ClassName(void) {
        return "AccumulateInterfacePresence2";
    }

    static const char* Description(void) {
        return "AccumulateInterfacePresence2";
    }

    static bool IsAvailable(void) {
        return true;
    }

    AccumulateInterfacePresence2();

    virtual ~AccumulateInterfacePresence2();

protected:
    bool create() override;

    void release() override;

private:
    bool is_dirty() const {
        return frame_count_slot_.IsDirty() || toggle_all_frames_slot_.IsDirty();
    }

    void reset_dirty() {
        frame_count_slot_.ResetDirty();
        toggle_all_frames_slot_.ResetDirty();
    }

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(core::moldyn::MultiParticleDataCall& points);

    core::CallerSlot data_in_slot_;

    core::CalleeSlot data_out_slot_;

    core::param::ParamSlot frame_count_slot_;

    core::param::ParamSlot toggle_all_frames_slot_;

    size_t in_data_hash_ = std::numeric_limits<size_t>::max();

    size_t out_data_hash_ = std::numeric_limits<size_t>::max();

    int frame_id_ = -1;

    int frame_count_ = 0;

    std::unordered_map<uint64_t, std::tuple<int /* state */, int /* start frame id*/, int /* end frame id */,
                                     std::array<float, 3> /* start pos */, std::array<float, 3> /* end pos */>>
        state_cache_;

    std::vector<std::tuple<uint64_t /* id */, int /* start frame id */, int /* end frame id */,
        std::array<float, 3> /* start pos */, std::array<float, 3> /* end pos */>>
        states_;

    std::vector<float> data_;

    std::vector<stdplugin::datatools::table::TableDataCall::ColumnInfo> infos_;
};

} // namespace megamol::thermodyn
