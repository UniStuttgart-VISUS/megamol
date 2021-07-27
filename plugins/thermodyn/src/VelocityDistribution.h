#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"

#include "mmstd_datatools/table/TableDataCall.h"

#include "thermodyn/CallStatsInfo.h"

namespace megamol::thermodyn {
class VelocityDistribution : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "VelocityDistribution";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Determines the distribution of velocities";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    VelocityDistribution();

    virtual ~VelocityDistribution();

protected:
    bool create() override;

    void release() override;

private:
    enum class mode : std::uint8_t { icol, dir };

    core::CalleeSlot out_dist_slot_;

    core::CalleeSlot out_stat_slot_;

    core::CalleeSlot out_part_slot_;

    core::CallerSlot in_data_slot_;

    core::param::ParamSlot num_buckets_slot_;

    core::param::ParamSlot dump_histo_slot_;

    core::param::ParamSlot path_slot_;

    core::param::ParamSlot mode_slot_;

    bool is_dirty() {
        return num_buckets_slot_.IsDirty();
    }

    void reset_dirty() {
        num_buckets_slot_.ResetDirty();
    }

    bool assert_data(core::moldyn::MultiParticleDataCall& call);

    bool dump_histo(core::param::ParamSlot& p);

    void compute_statistics();

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool get_stats_data_cb(core::Call& c);

    bool get_stats_extent_cb(core::Call& c);

    bool get_parts_data_cb(core::Call& c);

    bool get_parts_extent_cb(core::Call& c);

    std::vector<std::vector<std::uint32_t>> histograms_;

    std::vector<std::vector<float>> domain_;

    std::vector<float> data_;

    std::vector<float> mean_;

    std::vector<float> stddev_;

    std::vector<std::vector<float>> part_data_;

    std::vector<stdplugin::datatools::table::TableDataCall::ColumnInfo> ci_;

    std::size_t row_cnt_ = 0;

    std::size_t col_cnt_ = 0;

    int frame_id_ = -1;

    std::size_t in_data_hash_ = std::numeric_limits<std::size_t>::max();

    std::size_t out_data_hash_ = 0;
};
} // namespace megamol::thermodyn
