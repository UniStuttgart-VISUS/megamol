/*
 * SplitterWidget.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "SplitterWidget.h"


using namespace megamol;
using namespace megamol::gui;


SplitterWidget::SplitterWidget(void) : splitter_last_width(0.0f) {}


bool megamol::gui::SplitterWidget::Widget(FixedSplitterSide fixed_side, float& size_left, float& size_right) {

    assert(ImGui::GetCurrentContext() != nullptr);

    const float thickness = (12.0f * megamol::gui::gui_scaling.Get());

    bool split_vertically = true;
    float min_size = 1.0f; // >=1.0!
    float splitter_long_axis_size = ImGui::GetContentRegionAvail().y;

    float width_avail = ImGui::GetWindowSize().x - (2.0f * thickness);

    size_left = ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT) ? size_left : (width_avail - size_right));
    size_right = ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT) ? (width_avail - size_left) : size_right);

    size_left = std::max(size_left, min_size);
    size_right = std::max(size_right, min_size);

    ImGuiWindow* window = ImGui::GetCurrentContext()->CurrentWindow;
    ImGuiID id = window->GetID("##Splitter");
    ImRect bb;
    if (fixed_side == SplitterWidget::FixedSplitterSide::LEFT) {
        bb.Min =
            window->DC.CursorPos + (split_vertically ? ImVec2(size_left + 1.0f, 0.0f) : ImVec2(0.0f, size_left + 1.0f));
    } else if (fixed_side == SplitterWidget::FixedSplitterSide::RIGHT) {
        bb.Min = window->DC.CursorPos + (split_vertically ? ImVec2((width_avail - size_right) + 1.0f, 0.0f)
                                                          : ImVec2(0.0f, (width_avail - size_right) + 1.0f));
    }
    bb.Max = bb.Min + ImGui::CalcItemSize(split_vertically ? ImVec2(thickness / 2.0f, splitter_long_axis_size)
                                                           : ImVec2(splitter_long_axis_size, thickness / 2.0f),
                          0.0f, 0.0f);

    bool retval = ImGui::SplitterBehavior(
        bb, id, split_vertically ? ImGuiAxis_X : ImGuiAxis_Y, &size_left, &size_right, min_size, min_size, 0.0f, 0.0f);

    /// XXX Left mouse button (= 0) is not recognized properly
    if (ImGui::IsMouseDoubleClicked(1) && ImGui::IsItemHovered()) {
        float consider_width = ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT) ? size_left : size_right);
        if (consider_width <= min_size) {
            size_left =
                ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT) ? (this->splitter_last_width)
                                                                         : (width_avail - this->splitter_last_width));
            size_right =
                ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT) ? (width_avail - this->splitter_last_width)
                                                                         : (this->splitter_last_width));
        } else {
            size_left =
                ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT) ? (min_size) : (width_avail - min_size));
            size_right =
                ((fixed_side == SplitterWidget::FixedSplitterSide::LEFT) ? (width_avail - min_size) : (min_size));
            this->splitter_last_width = consider_width;
        }
    }

    return retval;
}
