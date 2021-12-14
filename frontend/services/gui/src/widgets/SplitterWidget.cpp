/*
 * SplitterWidget.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "SplitterWidget.h"
#include "gui_utils.h"


using namespace megamol;
using namespace megamol::gui;


SplitterWidget::SplitterWidget() : splitter_last_width(0.0f) {}


bool megamol::gui::SplitterWidget::Widget(bool split_vertically, FixedSplitterSide fixed_side, float& size_left_top, float& size_right_bottom) {

    assert(ImGui::GetCurrentContext() != nullptr);

    const float thickness = (12.0f * megamol::gui::gui_scaling.Get());

    float min_size = 1.0f; // >=1.0!
    float splitter_long_axis_size = (split_vertically) ? (ImGui::GetContentRegionAvail().y) : (ImGui::GetContentRegionAvail().x);

    float avail_size = (split_vertically) ? (ImGui::GetWindowSize().x - (2.0f * thickness)) : (ImGui::GetWindowSize().y - (2.0f * thickness));

    size_left_top = ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT_TOP) ? size_left_top : (avail_size - size_right_bottom));
    size_right_bottom = ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT_TOP) ? (avail_size - size_left_top) : size_right_bottom);

    size_left_top = std::max(size_left_top, min_size);
    size_right_bottom = std::max(size_right_bottom, min_size);

    ImGuiWindow* window = ImGui::GetCurrentContext()->CurrentWindow;
    ImGuiID id = window->GetID("##Splitter");
    ImRect bb;
    if (fixed_side == SplitterWidget::FixedSplitterSide::LEFT_TOP) {
        bb.Min =
            window->DC.CursorPos + (split_vertically ? ImVec2(size_left_top + 1.0f, 0.0f) : ImVec2(0.0f, size_left_top + 1.0f));
    } else if (fixed_side == SplitterWidget::FixedSplitterSide::RIGHT_BOTTOM) {
        bb.Min = window->DC.CursorPos + (split_vertically ? ImVec2((avail_size - size_right_bottom) + 1.0f, 0.0f)
                                                          : ImVec2(0.0f, (avail_size - size_right_bottom) + 1.0f));
    }
    bb.Max = bb.Min + ImGui::CalcItemSize(split_vertically ? ImVec2(thickness / 2.0f, splitter_long_axis_size)
                                                           : ImVec2(splitter_long_axis_size, thickness / 2.0f),
                          0.0f, 0.0f);

    bool retval = ImGui::SplitterBehavior(
        bb, id, split_vertically ? ImGuiAxis_X : ImGuiAxis_Y, &size_left_top, &size_right_bottom, min_size, min_size, 0.0f, 0.0f);

    /// XXX Left mouse button is not recognized properly
    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Right) && ImGui::IsItemHovered()) {
        float consider_width = ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT_TOP) ? size_left_top : size_right_bottom);
        if (consider_width <= min_size) {
            size_left_top =
                ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT_TOP) ? (this->splitter_last_width)
                                                                         : (avail_size - this->splitter_last_width));
            size_right_bottom =
                ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT_TOP) ? (avail_size - this->splitter_last_width)
                                                                         : (this->splitter_last_width));
        } else {
            size_left_top =
                ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT_TOP) ? (min_size) : (avail_size - min_size));
            size_right_bottom =
                ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT_TOP) ? (avail_size - min_size) : (min_size));
            this->splitter_last_width = consider_width;
        }
    }

    return retval;
}
