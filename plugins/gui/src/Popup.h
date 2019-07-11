/*
 * PopUp.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <string>

#include <imgui.h>


namespace megamol {
namespace gui {

/**
 * Utility class for popup-style widgets.
 */
class Popup {
public:
    Popup(void);

    ~Popup(void) = default;

    /**
     * Show tooltip on hover.
     *
     * @param text        The tooltip text.
     * @param id          The id of the imgui item the tooltip belongs (only needed for delayed appearance of tooltip).
     * @param time_start  The time delay to wait until the tooltip is shown for a hovered imgui item.
     * @param time_end    The time delay to wait until the tooltip is hidden for a hovered imgui item.
     */
    void HoverToolTip(std::string text, ImGuiID id = 0, float time_start = 0.0f, float time_end = 4.0f);

    /**
     * Show help marker text with tooltip on hover.
     *
     * @param text   The help tooltip text.
     * @param label  The visible text for which the tooltip is enabled.
     */
    void HelpMarkerToolTip(std::string text, std::string label = "(?)");

    /**
     * Open PopUp asking for user input.
     *
     * @param title    The popup title.
     * @param request  The descriptopn of the requested text input (e.g. file name).
     * @param open     The flag indicating that the popup should be opened.
     *
     * @return The captured text input.
     */
    std::string InputDialogPopUp(std::string title, std::string request, bool open);

private:
    /** Current tooltip hover time. */
    float tooltipTime;

    /** Current hovered tooltip item. */
    ImGuiID tooltipId;
};


} // namespace gui
} // namespace megamol