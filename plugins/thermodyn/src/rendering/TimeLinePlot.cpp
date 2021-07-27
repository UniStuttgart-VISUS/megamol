#include "TimeLinePlot.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/UniFlagCalls.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/view/CallTime.h"


megamol::thermodyn::rendering::TimeLinePlot::TimeLinePlot()
        : in_table_slot_("inTable", "")
        , flags_read_slot_("flagsRead", "")
        , flags_write_slot_("flagsWrite", "")
        , out_time_slot_("outTime", "")
        , x_axis_slot_("x axis", "")
        , y_axis_slot_("y axis", "") {
    in_table_slot_.SetCompatibleCall<stdplugin::datatools::table::TableDataCallDescription>();
    MakeSlotAvailable(&in_table_slot_);

    flags_read_slot_.SetCompatibleCall<core::FlagCallRead_CPUDescription>();
    MakeSlotAvailable(&flags_read_slot_);

    flags_write_slot_.SetCompatibleCall<core::FlagCallWrite_CPUDescription>();
    MakeSlotAvailable(&flags_write_slot_);

    out_time_slot_.SetCallback(
        core::view::CallTime::ClassName(), core::view::CallTime::FunctionName(0), &TimeLinePlot::get_data_cb);
    MakeSlotAvailable(&out_time_slot_);

    x_axis_slot_ << new core::param::FlexEnumParam("undef");
    MakeSlotAvailable(&x_axis_slot_);

    y_axis_slot_ << new core::param::FlexEnumParam("undef");
    MakeSlotAvailable(&y_axis_slot_);
}


megamol::thermodyn::rendering::TimeLinePlot::~TimeLinePlot() {
    this->Release();
}


bool megamol::thermodyn::rendering::TimeLinePlot::create() {
    ctx_ = ImPlot::CreateContext();
    return true;
}


void megamol::thermodyn::rendering::TimeLinePlot::release() {
    ImPlot::DestroyContext(ctx_);
}


bool megamol::thermodyn::rendering::TimeLinePlot::get_data_cb(core::Call& c) {
    auto out_time = dynamic_cast<core::view::CallTime*>(&c);
    if (out_time == nullptr)
        return false;

    auto in_table = in_table_slot_.CallAs<stdplugin::datatools::table::TableDataCall>();
    if (in_table == nullptr)
        return false;

    in_table->SetFrameID(out_frame_id_);
    if (!(*in_table)(1))
        return false;
    if (!(*in_table)(0))
        return false;

    if (is_dirty() || frame_id_ != in_table->GetFrameID() || in_data_hash_ != in_table->DataHash()) {
        parse_data(*in_table);

        reset_dirty();
        frame_id_ = out_frame_id_;
        in_data_hash_ = in_table->DataHash();
        //++out_data_hash_;
    }

    ///*auto ctx = reinterpret_cast<ImGuiContext*>(this->GetCoreInstance()->GetCurrentImGuiContext());
    // if (ctx != nullptr)*/ {
    //    /*ImGui::SetCurrentContext(ctx);

    //    ImGuiIO& io = ImGui::GetIO();
    //    ImVec2 viewport = ImVec2(io.DisplaySize.x, io.DisplaySize.y);*/
    //    /*bool my_tool_active = true;
    //    ImGui::SetNextWindowPos(ImVec2(450, 20));
    //    ImGui::Begin("ProbeMenuButton", &my_tool_active,
    //        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground);
    //    if (ImGui::Button("Probe", ImVec2(75, 20))) {}
    //    ImGui::End();*/

    //    ImGui::SetNextWindowPos(ImVec2(450, 20), ImGuiCond_Appearing);

    //    ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_Appearing);
    //    bool plot_open = true;
    //    ImGui::Begin("Test Plot", &plot_open, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground);
    //    ImPlot::SetNextPlotLimits(0, y_data_.size(), range_[0], range_[1]);
    //
    //    if (ImPlot::BeginPlot("data", nullptr, nullptr, ImVec2(-1,0), ImPlotFlags_Query)) {
    //        ImPlot::PlotBars("histo", y_data_.data(), y_data_.size());
    //        if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(0) && ImGui::GetIO().KeyShift) {
    //            auto const pt = ImPlot::GetPlotMousePos();
    //            core::utility::log::Log::DefaultLog.WriteInfo(
    //                "[TimeLinePlot] Query information %f %f", pt.x, pt.y);
    //        }
    //        /*if (ImPlot::IsPlotQueried()) {
    //            auto const query = ImPlot::GetPlotQuery();
    //            core::utility::log::Log::DefaultLog.WriteInfo(
    //                "[TimeLinePlot] Query information %f %f %f %f", query.X.Min, query.X.Max, query.Y.Min,
    //                query.Y.Max);
    //        }*/
    //        ImPlot::EndPlot();
    //    }
    //    ImGui::End();
    //
    //}
    widget();

    out_time->setData(out_frame_id_, out_data_hash_);

    return true;
}


// bool megamol::thermodyn::rendering::TimeLinePlot::GetExtents(core::view::CallRender2DGL& c) {
//    auto in_table = in_table_slot_.CallAs<stdplugin::datatools::table::TableDataCall>();
//    if (in_table == nullptr)
//        return false;
//
//    in_table->SetFrameID(c.Time());
//    if (!(*in_table)(1))
//        return false;
//
//    c.SetTimeFramesCount(in_table->GetFrameCount());
//
//    return true;
//}


bool megamol::thermodyn::rendering::TimeLinePlot::parse_data(stdplugin::datatools::table::TableDataCall& table) {
    auto const column_count = table.GetColumnsCount();
    auto const row_count = table.GetRowsCount();

    auto const infos = table.GetColumnsInfos();
    auto const data = table.GetData();

    auto x_axis_param = x_axis_slot_.Param<core::param::FlexEnumParam>();
    auto y_axis_param = y_axis_slot_.Param<core::param::FlexEnumParam>();
    x_axis_param->ClearValues();
    y_axis_param->ClearValues();

    for (std::remove_cv_t<decltype(column_count)> i = 0; i < column_count; ++i) {
        x_axis_param->AddValue(infos[i].Name());
        y_axis_param->AddValue(infos[i].Name());
    }

    auto const x_axis_name = x_axis_slot_.Param<core::param::FlexEnumParam>()->Value();
    auto const y_axis_name = y_axis_slot_.Param<core::param::FlexEnumParam>()->Value();

    auto const x_fit = std::find_if(
        infos, infos + column_count, [&x_axis_name](auto const& info) { return info.Name() == x_axis_name; });
    auto const y_fit = std::find_if(
        infos, infos + column_count, [&y_axis_name](auto const& info) { return info.Name() == y_axis_name; });

    if (x_fit >= infos + column_count) {
        core::utility::log::Log::DefaultLog.WriteError("[TimeLinePlot] Could not find column {}", x_axis_name);
        return false;
    }

    if (y_fit >= infos + column_count) {
        core::utility::log::Log::DefaultLog.WriteError("[TimeLinePlot] Could not find column {}", y_axis_name);
        return false;
    }

    auto const x_idx = std::distance(infos, x_fit);
    auto const y_idx = std::distance(infos, y_fit);

    x_data_.resize(row_count);
    y_data_.resize(row_count);

    for (std::remove_cv_t<decltype(row_count)> i = 0; i < row_count; ++i) {
        x_data_[i] = table.GetData(x_idx, i);
        y_data_[i] = table.GetData(y_idx, i);
    }

    range_[0] = infos[y_idx].MinimumValue();
    range_[1] = infos[y_idx].MaximumValue();

    /*auto ctx = reinterpret_cast<ImGuiContext*>(this->GetCoreInstance()->GetCurrentImGuiContext());
    if (ctx != nullptr) {
        ImGui::SetCurrentContext(ctx);

        ImGuiIO& io = ImGui::GetIO();
        ImVec2 viewport = ImVec2(io.DisplaySize.x, io.DisplaySize.y);
        bool my_tool_active = true;
        ImGui::SetNextWindowPos(ImVec2(450, 20));
        ImGui::Begin("ProbeMenuButton", &my_tool_active,
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground);
        if (ImGui::Button("Probe", ImVec2(75, 20))) {

        }
        ImGui::End();

        ImGui::SetNextWindowPos(ImVec2(300, 20));

        ImGui::SetNextWindowSize(ImVec2(200, 200));
        ImGui::Begin("Test Plot");
        if (ImPlot::BeginPlot("data")) {
            ImPlot::PlotLine("line", x_data.data(), y_data.data(), row_count);
            ImPlot::EndPlot();
        }
        ImGui::End();
    }*/

    return true;
}


void megamol::thermodyn::rendering::TimeLinePlot::widget() {
    ImGui::SetNextWindowPos(ImVec2(450, 20), ImGuiCond_Appearing);

    ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_Appearing);
    bool plot_open = true;
    ImGui::Begin("Test Plot", &plot_open, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground);
    ImPlot::SetNextPlotLimits(
        -(bar_width_ * 0.5), static_cast<double>(y_data_.size()) - (1.0 - bar_width_ * 0.5), range_[0], range_[1]);

    if (ImPlot::BeginPlot("data", nullptr, nullptr, ImVec2(-1, 0), ImPlotFlags_Query)) {
        ImPlot::PlotBars("histo", y_data_.data(), y_data_.size(), bar_width_);
        if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(0) && ImGui::GetIO().KeyShift) {
            auto const pt = ImPlot::GetPlotMousePos();
            core::utility::log::Log::DefaultLog.WriteInfo("[TimeLinePlot] Query information %f %f", pt.x, pt.y);
            auto const res_idx = get_selected_bar(pt);
            if (res_idx.has_value()) {
                core::utility::log::Log::DefaultLog.WriteInfo(
                    "[TimeLinePlot] Selected value %f", y_data_[res_idx.value()]);
                out_frame_id_ = res_idx.value();
                ++out_data_hash_;
            }
        }
        auto draw_list = ImPlot::GetPlotDrawList();
        auto const lb = ImPlot::PlotToPixels(ImPlotPoint(out_frame_id_ - 0.6 * bar_width_, range_[0]));
        auto const rt = ImPlot::PlotToPixels(ImPlotPoint(out_frame_id_ + 0.6 * bar_width_, range_[1]));

        ImPlot::PushPlotClipRect();
        draw_list->AddRectFilled(lb, rt, IM_COL32(255, 0, 0, 64));
        ImPlot::PopPlotClipRect();
        /*if (ImPlot::IsPlotQueried()) {
            auto const query = ImPlot::GetPlotQuery();
            core::utility::log::Log::DefaultLog.WriteInfo(
                "[TimeLinePlot] Query information %f %f %f %f", query.X.Min, query.X.Max, query.Y.Min, query.Y.Max);
        }*/
        ImPlot::EndPlot();
    }
    ImGui::End();
}


std::optional<uint64_t> megamol::thermodyn::rendering::TimeLinePlot::get_selected_bar(ImPlotPoint const& pt) const {
    auto const res = std::nearbyintf(pt.x);
    if (res + 0.5f * bar_width_ >= pt.x && res - 0.5f * bar_width_ <= pt.x && res >= 0.0f &&
        res < static_cast<float>(y_data_.size())) {
        core::utility::log::Log::DefaultLog.WriteInfo("[TimeLinePlot] Selected Idx %f", res);
        return std::make_optional(static_cast<uint64_t>(res));
    }

    return std::nullopt;
}
