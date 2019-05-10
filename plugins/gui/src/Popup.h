/*
 * WidgetUtils.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#ifndef MEGAMOL_GUI_WIDGETUTILS_H_INCLUDED
#define MEGAMOL_GUI_WIDGETUTILS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

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
     * @param popup_name   The popup title.
     * @param request      The descriptopn of the requested text input (e.g. file name).
     * @param open         The flag indicating that the popup should be opened.
     *
     * @return The captured text input.
     */
    std::string InputDialogPopUp(std::string popup_name, std::string request, bool open);

private:
    /** Current tooltip hover time. */
    float tooltip_time;

    /** Current hovered tooltip item. */
    ImGuiID tooltip_id;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_WIDGETUTILS_H_INCLUDED
