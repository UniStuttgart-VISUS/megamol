/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "SplitterWidget.h"
#include "gui_utils.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::SplitterWidget::SplitterWidget() : splitter_last_width(0.0f) {}


bool megamol::gui::SplitterWidget::Widget(const std::string& idstr, bool vertical, float length,
    FixedSplitterSide fixed_side, float& inout_range_left_top, float& inout_range_right_bottom,
    ImVec2 window_cursor_pos) {

    assert(ImGui::GetCurrentContext() != nullptr);

    const float splitter_width = this->GetWidth();
    float min_size = 1.0f; // >=1.0!

    float splitter_length = length;
    if (splitter_length == 0.0f) {
        splitter_length = (vertical) ? (ImGui::GetContentRegionAvail().y) : (ImGui::GetContentRegionAvail().x);
    }

    float split_range = (vertical) ? (ImGui::GetWindowSize().x - (2.0f * splitter_width))
                                   : (ImGui::GetWindowSize().y - (1.0f * splitter_width));

    inout_range_left_top =
        ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT_TOP) ? inout_range_left_top
                                                                     : (split_range - inout_range_right_bottom));
    inout_range_right_bottom =
        ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT_TOP) ? (split_range - inout_range_left_top)
                                                                     : inout_range_right_bottom);

    inout_range_left_top = std::max(inout_range_left_top, min_size);
    inout_range_right_bottom = std::max(inout_range_right_bottom, min_size);

    ImRect bb;
    if (fixed_side == SplitterWidget::FixedSplitterSide::LEFT_TOP) {
        bb.Min = window_cursor_pos +
                 (vertical ? ImVec2(inout_range_left_top + 1.0f, 0.0f) : ImVec2(0.0f, inout_range_left_top + 1.0f));
    } else if (fixed_side == SplitterWidget::FixedSplitterSide::RIGHT_BOTTOM) {
        bb.Min = window_cursor_pos + (vertical ? ImVec2((split_range - inout_range_right_bottom) + 1.0f, 0.0f)
                                               : ImVec2(0.0f, (split_range - inout_range_right_bottom) + 1.0f));
    }
    bb.Max = bb.Min + ImGui::CalcItemSize(vertical ? ImVec2(splitter_width / 2.0f, splitter_length)
                                                   : ImVec2(splitter_length, splitter_width / 2.0f),
                          0.0f, 0.0f);

    ImGuiID id = ImGui::GetID(idstr.c_str());
    bool retval = ImGui::SplitterBehavior(bb, id, vertical ? ImGuiAxis_X : ImGuiAxis_Y, &inout_range_left_top,
        &inout_range_right_bottom, min_size, min_size, 0.0f, 0.0f);

    /// XXX IO Why is left mouse button not recognized properly?
    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Right) && ImGui::IsItemHovered()) {
        float consider_width = ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT_TOP) ? inout_range_left_top
                                                                                            : inout_range_right_bottom);
        if (consider_width <= min_size) {
            inout_range_left_top = ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT_TOP)
                                        ? (this->splitter_last_width)
                                        : (split_range - this->splitter_last_width));
            inout_range_right_bottom =
                ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT_TOP) ? (split_range - this->splitter_last_width)
                                                                             : (this->splitter_last_width));
        } else {
            inout_range_left_top =
                ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT_TOP) ? (min_size) : (split_range - min_size));
            inout_range_right_bottom =
                ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT_TOP) ? (split_range - min_size) : (min_size));
            this->splitter_last_width = consider_width;
        }
    }

    return retval;
}


float megamol::gui::SplitterWidget::GetWidth() const {

    return (12.0f * megamol::gui::gui_scaling.Get());
}
