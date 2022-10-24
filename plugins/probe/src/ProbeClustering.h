#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"

#include "datatools/clustering/DBSCAN.h"

#include "probe/ProbeCalls.h"

#include "glm/glm.hpp"

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
        return _eps_slot.IsDirty() || _minpts_slot.IsDirty() || _threshold_slot.IsDirty() ||
               _handwaving_slot.IsDirty() || _angle_threshold_slot.IsDirty();
    }

    bool is_debug_dirty() {
        return _lhs_idx_slot.IsDirty() || _rhs_idx_slot.IsDirty() || _print_debug_info_slot.IsDirty();
    }

    void reset_dirty() {
        _eps_slot.ResetDirty();
        _minpts_slot.ResetDirty();
        _threshold_slot.ResetDirty();
        _handwaving_slot.ResetDirty();
        _angle_threshold_slot.ResetDirty();
    }

    void reset_debug_dirty() {
        _lhs_idx_slot.ResetDirty();
        _rhs_idx_slot.ResetDirty();
        _print_debug_info_slot.ResetDirty();
    }

    bool print_debug_info(core::param::ParamSlot& p) {
        auto const lhs_idx = _lhs_idx_slot.Param<core::param::IntParam>()->Value();
        auto const rhs_idx = _rhs_idx_slot.Param<core::param::IntParam>()->Value();
        if (lhs_idx < _col_count && rhs_idx < _row_count) {
            auto const val = _sim_matrix[lhs_idx + rhs_idx * _col_count];
            core::utility::log::Log::DefaultLog.WriteInfo(
                "[ProbeClustering]: Similiarty val for %d:%d is %f", lhs_idx, rhs_idx, val);
            auto const angle = glm::degrees(
                std::acos(glm::dot(glm::normalize(_cur_dirs[lhs_idx]), glm::normalize(_cur_dirs[rhs_idx]))));
            core::utility::log::Log::DefaultLog.WriteInfo(
                "[ProbeClustering]: Angle between %d:%d is %f", lhs_idx, rhs_idx, angle);
            auto lhs_pos = *reinterpret_cast<glm::vec3 const*>(_points->get_position(lhs_idx));
            auto rhs_pos = *reinterpret_cast<glm::vec3 const*>(_points->get_position(rhs_idx));
            auto const dis = glm::distance(lhs_pos, rhs_pos);
            core::utility::log::Log::DefaultLog.WriteInfo(
                "[ProbeClustering]: Distance between %d:%d is %f", lhs_idx, rhs_idx, dis);
        }
        // reset_debug_dirty();

        if (lhs_idx < _probes->getProbeCount() && rhs_idx < _probes->getProbeCount()) {

            auto rhs_generic_probe = _probes->getGenericProbe(rhs_idx);
            auto lhs_generic_probe = _probes->getGenericProbe(lhs_idx);
            uint32_t rhs_cluster_id = std::visit(
                [](auto&& arg) -> uint32_t { return static_cast<uint32_t>(arg.m_cluster_id); }, rhs_generic_probe);
            uint32_t lhs_cluster_id = std::visit(
                [](auto&& arg) -> uint32_t { return static_cast<uint32_t>(arg.m_cluster_id); }, lhs_generic_probe);
            core::utility::log::Log::DefaultLog.WriteInfo("[ProbeClustering]: Assigned cluster IDs for %d:%d are %d:%d",
                lhs_idx, rhs_idx, lhs_cluster_id, rhs_cluster_id);
        }

        return true;
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

    core::param::ParamSlot _print_debug_info_slot;

    core::param::ParamSlot _toggle_reps_slot;

    core::param::ParamSlot _angle_threshold_slot;

    std::shared_ptr<datatools::genericPointcloud<float, 3>> _points;

    std::shared_ptr<datatools::clustering::kd_tree_t<float, 3>> _kd_tree;

    std::shared_ptr<ProbeCol> _probes = nullptr;

    float const* _sim_matrix = nullptr;

    std::vector<glm::vec3> _cur_dirs;

    datatools::clustering::cluster_result_t _cluster_res;

    unsigned int _frame_id = std::numeric_limits<unsigned int>::max();

    std::size_t _in_table_data_hash = std::numeric_limits<std::size_t>::max();

    std::size_t _out_data_hash = 0;

    std::size_t _col_count = 0;

    std::size_t _row_count = 0;
};
} // namespace megamol::probe
