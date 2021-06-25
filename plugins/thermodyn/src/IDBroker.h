#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/FlagCollections.h"

namespace megamol::thermodyn {
class IDBroker : public core::Module {
public:
    static const char* ClassName(void) {
        return "IDBroker";
    }

    static const char* Description(void) {
        return "IDBroker";
    }

    static bool IsAvailable(void) {
        return true;
    }

    IDBroker();

    virtual ~IDBroker();

protected:
    bool create() override;

    void release() override;

private:
    bool flags_read_cb(core::Call& c);

    bool flags_read_meta_cb(core::Call& c);

    bool flags_write_cb(core::Call& c);

    bool flags_write_meta_cb(core::Call& c);

    bool out_frame_id_data_cb(core::Call& c);

    bool out_frame_id_extent_cb(core::Call& c);

    core::CalleeSlot out_flags_read_slot_;

    core::CalleeSlot out_flags_write_slot_;

    core::CalleeSlot out_frame_id_slot_;

    core::CallerSlot id_max_context_slot_;

    core::CallerSlot id_sub_context_slot_;

    core::CallerSlot in_flags_read_slot_;

    core::CallerSlot in_flags_write_slot_;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    uint64_t in_sub_data_hash_ = std::numeric_limits<uint64_t>::max();

    int in_frame_id_ = -1;

    int in_sub_frame_id_ = -1;

    int out_frame_id_ = -1;

    std::vector<std::unordered_map<uint64_t /* idx */, uint64_t /* id */>> idx_maps_;

    std::vector<std::unordered_map<uint64_t /* id */, uint64_t /* idx */>> id_maps_;

    std::shared_ptr<core::FlagCollection_CPU> flag_col_;

    uint32_t version_ = 0;

    std::vector<uint64_t> prefix_count_;
};
} // namespace megamol::thermodyn
