#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "mmstd_datatools/clustering/DBSCAN.h"

namespace megamol::probe {
class ProbeClustering : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ProbeClustering";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "...";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    ProbeClustering();

    virtual ~ProbeClustering();

protected:
    bool create() override;

    void release() override;

private:
    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool is_dirty() {
        return _eps_slot.IsDirty() || _minpts_slot.IsDirty() || _threshold_slot.IsDirty() || _handwaving_slot.IsDirty();
    }

    bool is_debug_dirty() {
        return _lhs_idx_slot.IsDirty() || _rhs_idx_slot.IsDirty();
    }

    void reset_dirty() {
        _eps_slot.ResetDirty();
        _minpts_slot.ResetDirty();
        _threshold_slot.ResetDirty();
        _handwaving_slot.ResetDirty();
    }

    void reset_debug_dirty() {
        _lhs_idx_slot.ResetDirty();
        _rhs_idx_slot.ResetDirty();
    }

    core::CalleeSlot _out_probes_slot;

    core::CallerSlot _in_probes_slot;

    core::CallerSlot _in_table_slot;

    core::param::ParamSlot _eps_slot;

    core::param::ParamSlot _minpts_slot;

    core::param::ParamSlot _threshold_slot;

    core::param::ParamSlot _handwaving_slot;

    core::param::ParamSlot _lhs_idx_slot;

    core::param::ParamSlot _rhs_idx_slot;

    std::shared_ptr<stdplugin::datatools::genericPointcloud<float, 3>> _points;

    std::shared_ptr<stdplugin::datatools::clustering::kd_tree_t<float, 3>> _kd_tree;

    unsigned int _frame_id = std::numeric_limits<unsigned int>::max();

    std::size_t _in_table_data_hash = std::numeric_limits<std::size_t>::max();

    std::size_t _out_data_hash = 0;
};
} // namespace megamol::probe
