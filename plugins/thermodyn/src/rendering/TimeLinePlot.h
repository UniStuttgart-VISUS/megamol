#pragma once

#include <optional>

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer2DModule.h"

#include "mmstd_datatools/table/TableDataCall.h"

#include "implot.h"

namespace megamol::thermodyn::rendering {
class TimeLinePlot : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "TimeLinePlot";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Plots data";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    TimeLinePlot();

    virtual ~TimeLinePlot();

protected:
    bool create() override;

    void release() override;

private:
    bool get_data_cb(core::Call& c);

    //bool GetExtents(core::view::CallRender2DGL& c) override;

    bool parse_data(stdplugin::datatools::table::TableDataCall& table);

    bool is_dirty() const {
        return x_axis_slot_.IsDirty() || y_axis_slot_.IsDirty();
    }

    void reset_dirty() {
        x_axis_slot_.ResetDirty();
        y_axis_slot_.ResetDirty();
    }

    void widget();

    std::optional<uint64_t> get_selected_bar(ImPlotPoint const& pt) const;

    core::CallerSlot in_table_slot_;

    core::CallerSlot flags_read_slot_;

    core::CallerSlot flags_write_slot_;

    core::CalleeSlot out_time_slot_;

    core::param::ParamSlot x_axis_slot_;

    core::param::ParamSlot y_axis_slot_;

    int frame_id_ = -1;

    uint64_t out_data_hash_ = 0;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    ImPlotContext* ctx_ = nullptr;

    std::vector<float> x_data_;

    std::vector<float> y_data_;

    std::array<float, 2> range_;

    float bar_width_ = 0.5f;

    uint64_t out_frame_id_ = 0;
};
} // namespace megamol::thermodyn::rendering
