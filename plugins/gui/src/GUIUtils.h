/*
 * GUIUtils.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GUIUTILS_INCLUDED
#define MEGAMOL_GUI_GUIUTILS_INCLUDED

#include "mmcore/view/Input.h"

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
#include <tuple>

/// CMake exeption for the cluster "stampede2" running CentOS. (C++ filesystem support is not working?)
#ifdef GUI_USE_FILESYSTEM
#    include "FileUtils.h"
#endif // GUI_USE_FILESYSTEM


namespace megamol {
namespace gui {

#define GUI_INVALID_ID (-1)


/** Type for holding data of hotkeys*/
typedef std::tuple<megamol::core::view::KeyCode, bool> HotkeyData;


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
    inline void SetSearchFocus(bool focus) { this->search_focus = focus; }

    /** Set keyboard focus to search text input. */
    inline std::string GetSearchString(void) const { return this->search_string; }

    // Other utility functions ------------------------------------------------

    /**
     * Returns width of text drawn as widget.
     */
    float TextWidgetWidth(const std::string& text) const;

    /**
     * Draw draggable splitter between two child windows, relative to parent window size.
     * https://github.com/ocornut/imgui/issues/319
     */
    bool VerticalSplitter(float* size_left, float* size_right);

#ifdef GUI_USE_FILESYSTEM

    enum FileBrowserFlag { SAVE, LOAD };
    bool FileBrowserPopUp(FileBrowserFlag flag, bool open_popup, const std::string& label, std::string& inout_filename);

#endif // GUI_USE_FILESYSTEM

private:
    /** Current tooltip hover time. */
    float tooltip_time;

    /** Current hovered tooltip item. */
    ImGuiID tooltip_id;

    /** Set focus to search text input. */
    bool search_focus;

    /** Current search string. */
    std::string search_string;

    /** File Browser */
    std::string file_name_str;
    std::string file_path_str;
    size_t additional_lines;

#ifdef GUI_USE_FILESYSTEM
    bool splitPath(const fsns::path& in_file_path, std::string& out_path, std::string& out_file);
#endif // GUI_USE_FILESYSTEM
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUIUTILS_INCLUDED