/*
 * GUIUtils.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GUIUTILS_INCLUDED
#define MEGAMOL_GUI_GUIUTILS_INCLUDED


#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm> // search
#include <cctype>    // toupper
#include <string>


namespace megamol {
namespace gui {


// Utility functions ----------------------------------------------------------

/**
 * Utility class for GUIUtils-style widgets.
 */
class GUIUtils {
public:
    GUIUtils(void);

    ~GUIUtils(void) = default;

    // Tool tip widgets -------------------------------------------------------

    /**
     * Show tooltip on hover.
     *
     * @param text        The tooltip text.
     * @param id          The id of the imgui item the tooltip belongs (only needed for delayed appearance of tooltip).
     * @param time_start  The time delay to wait until the tooltip is shown for a hovered imgui item.
     * @param time_end    The time delay to wait until the tooltip is hidden for a hovered imgui item.
     */
    void HoverToolTip(const std::string& text, ImGuiID id = 0, float time_start = 0.0f, float time_end = 4.0f);

    /**
     * Show help marker text with tooltip on hover.
     *
     * @param text   The help tooltip text.
     * @param label  The visible text for which the tooltip is enabled.
     */
    void HelpMarkerToolTip(const std::string& text, std::string label = "(?)");

    // UTF8 String En-/Decoding -----------------------------------------------

    /** Decode string from UTF-8. */
    bool Utf8Decode(std::string& str) const;

    /** Encode string into UTF-8. */
    bool Utf8Encode(std::string& str) const;

    // String search widget ---------------------------------------------------

    /** Show string serach widget. */
    void StringSearch(const std::string& label, const std::string& help);

    /**
     * Returns true if search string is found in source as a case insensitive substring.
     *
     * @param source   The string to search in.
     * @param search   The string to search for in the source.
     */
    inline bool FindCaseInsensitiveSubstring(const std::string& source, const std::string& search) const {

        auto it = std::search(source.begin(), source.end(), search.begin(), search.end(),
            [](char ch1, char ch2) { return std::toupper(ch1) == std::toupper(ch2); });
        return (it != source.end());
    }

    /** Set keyboard focus to search text input. */
    inline void SetSearchFocus(bool focus) { this->searchFocus = focus; }

    /** Set keyboard focus to search text input. */
    inline std::string GetSearchString(void) const { return this->searchString; }

    // Other utility functions ------------------------------------------------

    /**
     * Returns width of text drawn as widget.
     */
    float TextWidgetWidth(const std::string& text) const;

    /**
     * Draw draggable splitter between two child windows.
     * https://github.com/ocornut/imgui/issues/319
     */
    bool VerticalSplitter(float thickness, float* size_left, float* size_right);

private:
    /** Current tooltip hover time. */
    float tooltipTime;

    /** Current hovered tooltip item. */
    ImGuiID tooltipId;

    /** Set focus to search text input. */
    bool searchFocus;

    /** Current search string. */
    std::string searchString;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUIUTILS_INCLUDED